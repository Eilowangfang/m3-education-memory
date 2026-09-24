from __future__ import annotations

import json
import hashlib
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import connect, initialize
from .media import image_bytes_from_source, image_mime_type
from .image_preprocessing import ImagePreparationPolicy, prepare_image_for_vlm
from .memory_writer import rebuild_mastery_states
from .vlm import PROMPT_VERSION, VisionClient
from .verification import verify_correction


_WRITE_LOCK = threading.Lock()


def diagnose_attempt(
    db_path: str | Path,
    *,
    attempt_id: str,
    client: VisionClient,
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    attempt = connection.execute(
        """SELECT attempt_id,orig_q,image_source_path,Time,domain_code,subdomain_code
           FROM attempts WHERE attempt_id=?""",
        (attempt_id,),
    ).fetchone()
    if attempt is None:
        connection.close()
        raise KeyError(f"Unknown attempt_id: {attempt_id}")
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    preprocessing: dict = {}
    provider_metadata: dict = {}
    latency_ms: int | None = None
    inference_bytes = b""
    try:
        source_bytes = image_bytes_from_source(attempt["image_source_path"])
        policy = getattr(client, "image_policy", ImagePreparationPolicy())
        inference_bytes, preprocessing = prepare_image_for_vlm(
            source_bytes, policy=policy
        )
        call_started = time.perf_counter()
        response = client.analyze(
            image_bytes=inference_bytes,
            question_text=attempt["orig_q"],
        )
        latency_ms = round((time.perf_counter() - call_started) * 1000)
        if len(response) == 3:
            raw, parsed, provider_metadata = response
        else:
            raw, parsed = response
        correction = parsed.get("correction")
        verification = verify_correction(correction)
        with _WRITE_LOCK, connection:
            connection.execute(
                """INSERT INTO analysis_runs(
                     run_id,attempt_id,provider,model,prompt_version,status,
                     raw_response,parsed_json,error_message,input_image_bytes,
                     input_width,input_height,input_mime_type,preprocessing_json,
                     latency_ms,provider_usage_json,request_attempts,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, attempt_id, client.provider, client.model,
                    PROMPT_VERSION, "completed", raw,
                    json.dumps(parsed, ensure_ascii=False), None,
                    len(inference_bytes), preprocessing.get("output_width"),
                    preprocessing.get("output_height"), image_mime_type(inference_bytes),
                    json.dumps(preprocessing, ensure_ascii=False), latency_ms,
                    json.dumps(provider_metadata.get("usage", {}), ensure_ascii=False),
                    int(provider_metadata.get("request_attempts", 1)), created_at,
                ),
            )
            diagnosis_id = f"diagnosis:{run_id}"
            connection.execute(
                """INSERT INTO diagnoses(
                     diagnosis_id,run_id,attempt_id,question_transcription,
                     solution_transcription,has_error_pred,error_type_pred,
                     first_error_step,error_explanation,knowledge_points_json,
                     confidence,requires_review)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    diagnosis_id, run_id, attempt_id,
                    parsed["question_transcription"], parsed["solution_transcription"],
                    None if parsed["has_error"] is None else int(parsed["has_error"]),
                    parsed["error_type"], parsed["first_error_step"],
                    parsed["error_explanation"],
                    json.dumps(parsed["knowledge_points"], ensure_ascii=False),
                    float(parsed["confidence"]), int(bool(parsed["requires_review"])),
                ),
            )
            for step in parsed["steps"]:
                connection.execute(
                    """INSERT INTO diagnosis_steps(
                         run_id,step_index,transcription,normalized_latex,bbox_json,
                         is_error,confidence) VALUES(?,?,?,?,?,?,?)""",
                    (
                        run_id, int(step["step_index"]), step["transcription"],
                        step["normalized_latex"],
                        None if step["bbox"] is None else json.dumps(step["bbox"]),
                        None if step["is_error"] is None else int(step["is_error"]),
                        float(step["confidence"]),
                    ),
                )
            correction_id = None
            if correction is not None:
                correction_id = f"correction:{run_id}"
                connection.execute(
                    """INSERT INTO corrections(
                         correction_id,run_id,attempt_id,corrected_solution,
                         corrected_steps_json,final_answer,verification_expression,
                         verification_status,verification_method,
                         verification_details_json,confidence,rendered_path,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",
                    (
                        correction_id, run_id, attempt_id,
                        correction["corrected_solution"],
                        json.dumps(correction["corrected_steps"], ensure_ascii=False),
                        correction["final_answer"],
                        correction["verification_expression"],
                        verification["status"], verification["method"],
                        json.dumps(verification["details"], ensure_ascii=False),
                        float(correction["confidence"]), created_at,
                    ),
                )
                parent = connection.execute(
                    """SELECT version_id FROM correction_versions
                       WHERE attempt_id=? AND model=? AND status='active'
                       ORDER BY created_at DESC,version_id DESC LIMIT 1""",
                    (attempt_id, client.model),
                ).fetchone()
                connection.execute(
                    """UPDATE correction_versions SET status='superseded'
                       WHERE attempt_id=? AND model=? AND status='active'""",
                    (attempt_id, client.model),
                )
                version_id = f"correction-version:{run_id}"
                connection.execute(
                    """INSERT INTO correction_versions(
                         version_id,attempt_id,run_id,model,source,parent_version_id,
                         corrected_solution,corrected_steps_json,final_answer,
                         verification_expression,verification_status,
                         verification_method,verification_details_json,confidence,
                         status,created_by,rendered_path,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'vlm-worker',NULL,?)""",
                    (
                        version_id, attempt_id, run_id, client.model, "vlm",
                        parent["version_id"] if parent else None,
                        correction["corrected_solution"],
                        json.dumps(correction["corrected_steps"], ensure_ascii=False),
                        correction["final_answer"],
                        correction["verification_expression"],
                        verification["status"], verification["method"],
                        json.dumps(verification["details"], ensure_ascii=False),
                        float(correction["confidence"]), "active", created_at,
                    ),
                )
            source = f"vlm:{client.provider}:{client.model}:{PROMPT_VERSION}"
            memory_id = f"episode:{attempt_id}:{run_id}"
            content = (
                f"At simulated Time {attempt['Time']}, an image-only VLM diagnosis "
                f"classified the attempt as has_error={parsed['has_error']}, "
                f"type={parsed['error_type']}. {parsed['error_explanation']}"
            )
            connection.execute(
                """INSERT INTO episodic_memories(
                     memory_id,attempt_id,content,error_type,status,source,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    memory_id, attempt_id, content, parsed["error_type"],
                    "hypothesis_requires_review" if parsed["requires_review"] else "hypothesis",
                    source, created_at,
                ),
            )
            connection.execute(
                """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
                   VALUES(?,?,?,?)""",
                (
                    memory_id, "ErrorEpisode", parsed["error_type"],
                    json.dumps({"source": source, "run_id": run_id}, ensure_ascii=False),
                ),
            )
            connection.execute(
                """INSERT INTO memory_edges(source_id,relation,target_id)
                   VALUES(?,?,?)""",
                (f"attempt:{attempt_id}", "HAS_VLM_DIAGNOSIS", memory_id),
            )
            if correction_id:
                connection.execute(
                    """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
                       VALUES(?,?,?,?)""",
                    (
                        correction_id, "Correction",
                        correction.get("final_answer") or "corrected solution",
                        json.dumps(
                            {
                                "run_id": run_id,
                                "verification_status": verification["status"],
                                "verification_method": verification["method"],
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_edges(source_id,relation,target_id) VALUES(?,?,?)",
                    (correction_id, "CORRECTS", memory_id),
                )
                connection.execute(
                    "INSERT INTO memory_edges(source_id,relation,target_id) VALUES(?,?,?)",
                    (f"attempt:{attempt_id}", "HAS_CORRECTION", correction_id),
                )
                version_node_id = f"correction-version-node:{version_id}"
                connection.execute(
                    """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
                       VALUES(?,?,?,?)""",
                    (
                        version_node_id, "CorrectionVersion",
                        correction.get("final_answer") or "corrected solution",
                        json.dumps(
                            {"source": "vlm", "status": "active", "run_id": run_id},
                            ensure_ascii=False,
                        ),
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_edges(source_id,relation,target_id) VALUES(?,?,?)",
                    (f"attempt:{attempt_id}", "HAS_CORRECTION_VERSION", version_node_id),
                )
            rebuild_mastery_states(
                connection,
                model=client.model,
                prompt_version=PROMPT_VERSION,
                change_reason=f"diagnosis:{run_id}",
            )
        result = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            **parsed,
            "verification": verification,
            "inference": {
                "preprocessing": preprocessing,
                "latency_ms": latency_ms,
                "provider": provider_metadata,
            },
        }
    except Exception as exc:
        if latency_ms is None and "call_started" in locals():
            latency_ms = round((time.perf_counter() - call_started) * 1000)
        provider_metadata["request_attempts"] = getattr(
            exc, "request_attempts", provider_metadata.get("request_attempts", 1)
        )
        with _WRITE_LOCK, connection:
            connection.execute(
                """INSERT INTO analysis_runs(
                     run_id,attempt_id,provider,model,prompt_version,status,
                     raw_response,parsed_json,error_message,input_image_bytes,
                     input_width,input_height,input_mime_type,preprocessing_json,
                     latency_ms,provider_usage_json,request_attempts,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, attempt_id, client.provider, client.model,
                    PROMPT_VERSION, "failed", None, None, str(exc),
                    len(inference_bytes) if inference_bytes else None,
                    preprocessing.get("output_width"), preprocessing.get("output_height"),
                    image_mime_type(inference_bytes) if inference_bytes else None,
                    json.dumps(preprocessing, ensure_ascii=False) if preprocessing else None,
                    latency_ms,
                    json.dumps(provider_metadata.get("usage", {}), ensure_ascii=False),
                    int(provider_metadata.get("request_attempts", 1)), created_at,
                ),
            )
        connection.close()
        raise
    connection.close()
    try:
        from .retrieval import index_memory_attempt

        result["index_update"] = index_memory_attempt(
            db_path, model=client.model, attempt_id=attempt_id
        )
    except Exception as index_exc:
        # Diagnosis is durable even if the derived retrieval index needs repair.
        result["index_error"] = str(index_exc)
    return result


def analysis_run_metrics(
    db_path: str | Path, *, model: str | None = None
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    where = "WHERE model=?" if model else ""
    params = (model,) if model else ()
    rows = connection.execute(
        f"""SELECT status,input_image_bytes,preprocessing_json,latency_ms,
                   provider_usage_json,request_attempts
            FROM analysis_runs {where} ORDER BY created_at""",
        params,
    ).fetchall()
    connection.close()
    latencies = sorted(
        int(row["latency_ms"]) for row in rows if row["latency_ms"] is not None
    )
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] == "failed"]
    processed = 0
    source_bytes = output_bytes = 0
    token_totals: dict[str, int] = {}
    for row in rows:
        prep = json.loads(row["preprocessing_json"] or "{}")
        processed += int(bool(prep.get("processed")))
        source_bytes += int(prep.get("source_bytes") or 0)
        output_bytes += int(prep.get("output_bytes") or 0)
        for key, value in json.loads(row["provider_usage_json"] or "{}").items():
            if isinstance(value, int):
                token_totals[key] = token_totals.get(key, 0) + value
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1) if latencies else 0
    return {
        "model": model,
        "run_count": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "success_rate": len(completed) / len(rows) if rows else None,
        "preprocessed_count": processed,
        "source_image_bytes": source_bytes,
        "inference_image_bytes": output_bytes,
        "byte_reduction_rate": (
            1 - output_bytes / source_bytes if source_bytes else None
        ),
        "average_latency_ms": (
            round(sum(latencies) / len(latencies)) if latencies else None
        ),
        "p95_latency_ms": latencies[p95_index] if latencies else None,
        "request_attempts": sum(int(row["request_attempts"] or 0) for row in rows),
        "provider_usage_totals": token_totals,
    }


def select_attempt_ids(
    db_path: str | Path,
    *,
    domain_code: str | None = None,
    limit: int = 1,
    only_unprocessed_model: str | None = None,
) -> list[str]:
    connection = connect(db_path)
    where = []
    params: list[object] = []
    if domain_code:
        where.append("a.domain_code=?")
        params.append(domain_code)
    if only_unprocessed_model:
        where.append(
            "NOT EXISTS (SELECT 1 FROM analysis_runs r WHERE r.attempt_id=a.attempt_id "
            "AND r.model=? AND r.prompt_version=? AND r.status='completed')"
        )
        params.extend((only_unprocessed_model, PROMPT_VERSION))
    clause = " WHERE " + " AND ".join(where) if where else ""
    rows = connection.execute(
        "SELECT a.attempt_id FROM attempts a" + clause + " ORDER BY a.Time,a.attempt_id LIMIT ?",
        [*params, limit],
    ).fetchall()
    connection.close()
    return [row["attempt_id"] for row in rows]


def select_stratified_attempt_ids(
    db_path: str | Path,
    *,
    limit: int,
    seed: str,
    model: str | None = None,
) -> list[str]:
    """Select a deterministic evaluation pilot without exposing truth to the VLM."""
    connection = connect(db_path)
    rows = connection.execute(
        """SELECT a.attempt_id,a.domain_code,a.grade,b.has_error
           FROM attempts a JOIN benchmark_truth b USING(attempt_id)"""
    ).fetchall()
    processed: set[str] = set()
    if model:
        processed = {
            row[0]
            for row in connection.execute(
                """SELECT attempt_id FROM analysis_runs
                   WHERE model=? AND prompt_version=? AND status='completed'""",
                (model, PROMPT_VERSION),
            ).fetchall()
        }
    connection.close()
    buckets: dict[tuple[int, str, str], list[str]] = {}
    for row in rows:
        if row["attempt_id"] in processed:
            continue
        key = (row["has_error"], row["domain_code"], row["grade"])
        buckets.setdefault(key, []).append(row["attempt_id"])
    for key, values in buckets.items():
        values.sort(
            key=lambda attempt_id: hashlib.sha256(
                f"{seed}:{attempt_id}".encode("utf-8")
            ).hexdigest()
        )
    def pick(label: int, target: int) -> list[str]:
        chosen: list[str] = []
        keys = [key for key in sorted(buckets) if key[0] == label]
        while len(chosen) < target and keys:
            next_keys = []
            for key in keys:
                values = buckets[key]
                if values and len(chosen) < target:
                    chosen.append(values.pop())
                if values:
                    next_keys.append(key)
            keys = next_keys
        return chosen

    correct = pick(0, limit // 2)
    errors = pick(1, limit - len(correct))
    if len(correct) + len(errors) < limit:
        correct.extend(pick(0, limit - len(correct) - len(errors)))
    selected = []
    for index in range(max(len(correct), len(errors))):
        if index < len(errors):
            selected.append(errors[index])
        if index < len(correct):
            selected.append(correct[index])
    return selected[:limit]
