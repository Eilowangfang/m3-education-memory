from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from m3_edu_memory.annotations import get_attempt_image
from m3_edu_memory.control import build_memory_report
from m3_edu_memory.database import connect
from m3_edu_memory.ingestion import process_next_job
from m3_edu_memory.image_preprocessing import ImagePreparationPolicy
from m3_edu_memory.embeddings import create_embedding_client
from m3_edu_memory.providers import create_vision_client
from m3_edu_memory.retrieval import build_memory_index
from m3_edu_memory.server import MemoryApi, _handler
from m3_edu_memory.vlm import OpenAICompatibleVisionClient


def _json_request(url: str, *, method: str = "GET", body: dict | None = None):
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=encoded,
        method=method,
        headers={"Content-Type": "application/json"} if encoded else {},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = json.loads(exc.read().decode("utf-8"))
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def run(args) -> dict:
    started = time.perf_counter()
    if args.provider_profile:
        client = create_vision_client(
            args.provider_config, profile_name=args.provider_profile
        )
    else:
        if not args.base_url:
            raise ValueError(
                "--base-url is required when --provider-profile is disabled"
            )
        client = OpenAICompatibleVisionClient(
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            max_retries=1,
            provider=args.provider,
            image_detail=args.image_detail,
            image_policy=ImagePreparationPolicy(
                max_long_edge=args.image_max_long_edge,
                max_pixels=args.image_max_pixels,
                max_bytes=args.image_max_bytes,
                jpeg_quality=args.image_jpeg_quality,
                min_jpeg_quality=args.image_min_jpeg_quality,
            ),
            request_options=getattr(args, "request_options", {}),
        )
    model = client.model
    provider = client.provider
    embedding_client = (
        create_embedding_client(
            args.embedding_config, profile_name=args.embedding_profile
        )
        if args.embedding_config and args.embedding_profile
        else None
    )
    source = connect(args.source_db)
    source_attempt = source.execute(
        """SELECT attempt_id,orig_q,domain_code,subdomain_code,grade
           FROM attempts WHERE attempt_id=?""",
        (args.source_attempt_id,),
    ).fetchone()
    source.close()
    if source_attempt is None:
        raise KeyError(f"Unknown source attempt: {args.source_attempt_id}")
    image_bytes, mime_type = get_attempt_image(
        args.source_db, attempt_id=args.source_attempt_id
    )
    domain_names = {
        "clc": "微积分", "alg": "代数", "art": "算术", "mgm": "几何",
        "pst": "概率统计", "trg": "三角", "apt": "综合能力",
    }
    domain_name = domain_names.get(source_attempt["domain_code"], "数学")
    memory_request = f"帮我整理过去在{domain_name}上的作答"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_root) / f"{args.run_prefix}-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    db_path = root / "memory.db"
    api = MemoryApi(
        db_path,
        default_model=model,
        object_root=root / "objects",
        embedding_client=embedding_client,
        vector_weight=1.0 if embedding_client else 0.65,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(api))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        upload_started = time.perf_counter()
        upload_status, upload = _json_request(
            base + "/v1/attempts",
            method="POST",
            body={
                "image_base64": (
                    f"data:{mime_type};base64,"
                    + base64.b64encode(image_bytes).decode("ascii")
                ),
                "question_text": source_attempt["orig_q"],
                "domain_code": source_attempt["domain_code"],
                "subdomain_code": source_attempt["subdomain_code"],
                "grade": source_attempt["grade"],
                "attempted_at": datetime.now(timezone.utc).isoformat(),
                "idempotency_key": f"{args.run_prefix}:{stamp}:{args.source_attempt_id}",
                "model": model,
                "max_attempts": 1,
            },
        )
        upload_seconds = time.perf_counter() - upload_started

        inference_started = time.perf_counter()
        worker = process_next_job(
            db_path,
            client=client,
            worker_id=f"{provider}-e2e-worker",
            retry_base_seconds=0,
        )
        inference_seconds = time.perf_counter() - inference_started
        if worker is None or worker["status"] != "completed":
            raise RuntimeError(f"E2E diagnosis failed: {worker}")

        index_result = build_memory_index(
            db_path,
            model=model,
            embedding_client=embedding_client,
        )

        attempt_id = upload["attempt_id"]
        status_code, attempt = _json_request(base + f"/v1/attempts/{attempt_id}")
        query_status, query = _json_request(
            base + "/v1/memory/query",
            method="POST",
            body={
                "request": memory_request,
                "model": model,
                "limit": 10,
            },
        )
        weakness_status, weaknesses = _json_request(
            base + f"/v1/memory/weaknesses?domain={source_attempt['domain_code']}&model={model}"
        )
        report = build_memory_report(
            db_path,
            request=memory_request,
            model=model,
            output_dir=root / "memory-report",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    connection = connect(db_path)
    counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "attempts", "analysis_runs", "diagnoses", "diagnosis_steps",
            "corrections", "correction_versions", "episodic_memories",
            "semantic_memories", "mastery_states", "memory_documents",
            "memory_embeddings", "ingestion_jobs",
        )
    }
    run = connection.execute(
        """SELECT provider,model,prompt_version,status,created_at
           FROM analysis_runs ORDER BY created_at DESC LIMIT 1"""
    ).fetchone()
    connection.close()
    diagnosis = worker["diagnosis"]
    result = {
        "result": "passed",
        "paid_vlm_calls": 1,
        "paid_embedding_calls": 1 if embedding_client else 0,
        "provider_profile": args.provider_profile,
        "embedding_profile": args.embedding_profile,
        "source_attempt_id": args.source_attempt_id,
        "memory_request": memory_request,
        "uploaded_attempt_id": upload["attempt_id"],
        "job_id": upload["job_id"],
        "http_statuses": {
            "upload": upload_status,
            "attempt": status_code,
            "memory_query": query_status,
            "weaknesses": weakness_status,
        },
        "analysis_run": dict(run),
        "diagnosis": {
            "has_error": diagnosis["has_error"],
            "error_type": diagnosis["error_type"],
            "first_error_step": diagnosis["first_error_step"],
            "confidence": diagnosis["confidence"],
            "requires_review": diagnosis["requires_review"],
            "knowledge_points": diagnosis["knowledge_points"],
            "correction_verification": (
                diagnosis["verification"]["status"]
                if diagnosis.get("correction") else None
            ),
        },
        "memory_query": {
            "returned": len(query.get("evidence", [])),
            "tool_trace_length": len(query.get("tool_trace", [])),
            "search_backend": query.get("search_backend"),
        },
        "memory_index": index_result,
        "weakness_count": len(weaknesses.get("weaknesses", [])),
        "database_counts": counts,
        "timing_seconds": {
            "upload": round(upload_seconds, 3),
            "vlm_and_memory_write": round(inference_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "artifacts": {
            "root": str(root.resolve()),
            "database": str(db_path.resolve()),
            "report_json": report["report_json"],
            "gallery_html": report["gallery_html"],
        },
    }
    (root / "e2e-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["artifacts"]["summary"] = str((root / "e2e-summary.json").resolve())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", default="data/education_memory.db")
    parser.add_argument(
        "--source-attempt-id", default="fermat-00002-00010-000116"
    )
    parser.add_argument("--output-root", default="outputs/e2e-validation")
    parser.add_argument("--provider-config", default="config/vlm_providers.json")
    parser.add_argument(
        "--provider-profile", default="volcengine-seed-2.1-turbo"
    )
    parser.add_argument("--embedding-config", default="config/embedding_providers.json")
    parser.add_argument(
        "--embedding-profile", default="volcengine-doubao-embedding-vision"
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--provider", default="aliyun-bailian")
    parser.add_argument("--run-prefix", default="seed-e2e")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--image-detail", choices=("low", "high", "xhigh"), default="high")
    parser.add_argument("--image-max-long-edge", type=int, default=2048)
    parser.add_argument("--image-max-pixels", type=int, default=3_000_000)
    parser.add_argument("--image-max-bytes", type=int, default=2_000_000)
    parser.add_argument("--image-jpeg-quality", type=int, default=88)
    parser.add_argument("--image-min-jpeg-quality", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
