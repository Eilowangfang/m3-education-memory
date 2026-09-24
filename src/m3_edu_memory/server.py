from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .annotations import (
    annotation_progress,
    get_attempt_image,
    list_evaluation_annotations,
    list_evaluation_cases,
    submit_evaluation_annotation,
)
from .annotation_ui import annotation_ui_html
from .control_tools import EducationControlAgent, generate_correction, get_evidence
from .correction_service import list_correction_versions, submit_correction_version
from .diagnosis import analysis_run_metrics
from .review import submit_diagnosis_review
from .review_ui import review_ui_html
from .student_ui import student_ui_html
from .retrieval import hybrid_search, memory_index_stats
from .embeddings import create_embedding_client
from .ingestion import (
    decode_image_base64,
    enqueue_uploaded_attempt,
    get_attempt_processing_status,
    get_ingestion_job,
    ingestion_metrics,
    retry_ingestion_job,
)
from .weakness import extract_vlm_weaknesses, mastery_history


class MemoryApi:
    def __init__(
        self,
        db_path: str | Path,
        *,
        default_model: str,
        object_root: str | Path | None = None,
        embedding_client=None,
        vector_weight: float = 0.65,
    ):
        self.db_path = Path(db_path)
        self.default_model = default_model
        self.object_root = Path(object_root or self.db_path.parent / "objects")
        self.embedding_client = embedding_client
        self.vector_weight = vector_weight

    def dispatch(
        self,
        method: str,
        target: str,
        body: dict | None = None,
    ) -> tuple[int, dict]:
        parsed = urlparse(target)
        params = parse_qs(parsed.query)
        segments = [unquote(item) for item in parsed.path.strip("/").split("/")]
        model = params.get("model", [self.default_model])[0]
        if method == "GET" and parsed.path == "/health":
            return HTTPStatus.OK, {"status": "ok", "model": self.default_model}
        if method == "GET" and parsed.path == "/v1/memory/weaknesses":
            domain = params.get("domain", [None])[0]
            return HTTPStatus.OK, extract_vlm_weaknesses(
                self.db_path, model=model, domain_code=domain
            )
        if method == "GET" and parsed.path == "/v1/memory/mastery-history":
            try:
                return HTTPStatus.OK, mastery_history(
                    self.db_path,
                    model=model,
                    domain_code=params.get("domain", [None])[0],
                    knowledge_point=params.get("knowledge_point", [None])[0],
                    prompt_version=params.get("prompt_version", [None])[0],
                    limit=int(params.get("limit", [100])[0]),
                )
            except (TypeError, ValueError) as exc:
                return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        if method == "GET" and parsed.path == "/v1/memory/index":
            return HTTPStatus.OK, memory_index_stats(self.db_path, model=model)
        if method == "GET" and parsed.path == "/v1/analysis/metrics":
            return HTTPStatus.OK, analysis_run_metrics(
                self.db_path, model=model if "model" in params else None
            )
        if (
            method == "GET" and len(segments) == 4
            and segments[:2] == ["v1", "evaluation-suites"]
            and segments[3] == "cases"
        ):
            try:
                return HTTPStatus.OK, list_evaluation_cases(
                    self.db_path,
                    suite=segments[2],
                    annotation_status=params.get("annotation_status", [None])[0],
                    limit=int(params.get("limit", [100])[0]),
                    offset=int(params.get("offset", [0])[0]),
                )
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
            except (TypeError, ValueError) as exc:
                return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        if (
            method == "GET" and len(segments) == 4
            and segments[:2] == ["v1", "evaluation-suites"]
            and segments[3] == "progress"
        ):
            try:
                return HTTPStatus.OK, annotation_progress(
                    self.db_path, suite=segments[2]
                )
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
        if (
            method == "GET" and len(segments) == 5
            and segments[:2] == ["v1", "evaluation-suites"]
            and segments[3] == "annotations"
        ):
            try:
                return HTTPStatus.OK, list_evaluation_annotations(
                    self.db_path, suite=segments[2], attempt_id=segments[4]
                )
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
        if method == "GET" and parsed.path.startswith("/v1/errors/") and parsed.path.endswith("/corrections"):
            attempt_id = parsed.path.removeprefix("/v1/errors/").removesuffix("/corrections").strip("/")
            return HTTPStatus.OK, list_correction_versions(
                self.db_path, attempt_id=attempt_id, model=model
            )
        if method == "GET" and parsed.path.startswith("/v1/attempts/"):
            attempt_id = parsed.path.removeprefix("/v1/attempts/")
            try:
                try:
                    evidence = get_evidence(
                        self.db_path, model=model, attempt_id=attempt_id
                    )
                    evidence["processing"] = get_attempt_processing_status(
                        self.db_path, attempt_id=attempt_id
                    )
                    return HTTPStatus.OK, evidence
                except KeyError:
                    return HTTPStatus.ACCEPTED, get_attempt_processing_status(
                        self.db_path, attempt_id=attempt_id
                    )
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
        if method == "GET" and parsed.path == "/v1/jobs/metrics":
            return HTTPStatus.OK, ingestion_metrics(self.db_path)
        if method == "GET" and parsed.path.startswith("/v1/jobs/"):
            job_id = parsed.path.removeprefix("/v1/jobs/")
            try:
                return HTTPStatus.OK, get_ingestion_job(
                    self.db_path, job_id=job_id
                )
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
        if method == "POST" and parsed.path == "/v1/attempts":
            body = body or {}
            try:
                result = enqueue_uploaded_attempt(
                    self.db_path,
                    object_root=self.object_root,
                    image_bytes=decode_image_base64(str(body.get("image_base64", ""))),
                    question_text=str(body.get("question_text", "")),
                    domain_code=str(body.get("domain_code", "")),
                    subdomain_code=body.get("subdomain_code"),
                    grade=str(body.get("grade") or "c00"),
                    attempted_at=body.get("attempted_at"),
                    idempotency_key=body.get("idempotency_key"),
                    requested_model=str(body.get("model") or self.default_model),
                    max_attempts=int(body.get("max_attempts", 3)),
                )
                return HTTPStatus.OK if result["idempotent_replay"] else HTTPStatus.ACCEPTED, result
            except (TypeError, ValueError) as exc:
                return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        if method == "POST" and parsed.path.startswith("/v1/jobs/") and parsed.path.endswith("/retry"):
            job_id = parsed.path.removeprefix("/v1/jobs/").removesuffix("/retry").strip("/")
            try:
                return HTTPStatus.OK, retry_ingestion_job(
                    self.db_path, job_id=job_id
                )
            except (KeyError, ValueError) as exc:
                return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        if method == "POST" and parsed.path == "/v1/memory/query":
            body = body or {}
            request = str(body.get("request", "")).strip()
            if not request:
                return HTTPStatus.BAD_REQUEST, {"error": "request is required"}
            request_model = str(body.get("model") or self.default_model)
            limit = int(body.get("limit", 10))
            return HTTPStatus.OK, EducationControlAgent(
                self.db_path,
                model=request_model,
                embedding_client=self.embedding_client,
                vector_weight=self.vector_weight,
            ).query(request, default_limit=limit)
        if method == "POST" and parsed.path == "/v1/memory/search":
            body = body or {}
            query = str(body.get("query", "")).strip()
            if not query:
                return HTTPStatus.BAD_REQUEST, {"error": "query is required"}
            return HTTPStatus.OK, hybrid_search(
                self.db_path,
                model=str(body.get("model") or self.default_model),
                query=query,
                candidate_attempt_ids=body.get("candidate_attempt_ids"),
                limit=int(body.get("limit", 20)),
                embedding_client=self.embedding_client,
                vector_weight=self.vector_weight,
            )
        if (
            method == "POST" and len(segments) == 5
            and segments[:2] == ["v1", "evaluation-suites"]
            and segments[3] == "annotations"
        ):
            body = body or {}
            try:
                return HTTPStatus.CREATED, submit_evaluation_annotation(
                    self.db_path,
                    suite=segments[2],
                    attempt_id=segments[4],
                    annotator=str(body.get("annotator", "")),
                    status=str(body.get("status", "draft")),
                    has_error=body.get("has_error"),
                    first_error_step=body.get("first_error_step"),
                    error_bbox=body.get("error_bbox"),
                    knowledge_points=body.get("knowledge_points"),
                    corrected_final_answer=body.get("corrected_final_answer"),
                    correction_steps=body.get("correction_steps"),
                    notes=str(body.get("notes", "")),
                    parent_annotation_id=body.get("parent_annotation_id"),
                )
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
            except (TypeError, ValueError) as exc:
                return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        if method == "POST" and parsed.path.startswith("/v1/errors/") and parsed.path.endswith("/review"):
            attempt_id = parsed.path.removeprefix("/v1/errors/").removesuffix("/review").strip("/")
            body = body or {}
            try:
                result = submit_diagnosis_review(
                    self.db_path,
                    attempt_id=attempt_id,
                    model=str(body.get("model") or self.default_model),
                    verdict=str(body.get("verdict", "")),
                    has_error=body.get("has_error"),
                    error_type=body.get("error_type"),
                    first_error_step=body.get("first_error_step"),
                    error_bbox=body.get("error_bbox"),
                    knowledge_points=body.get("knowledge_points"),
                    error_explanation=body.get("error_explanation"),
                    reviewer=str(body.get("reviewer", "local-teacher")),
                    notes=str(body.get("notes", "")),
                    run_id=body.get("run_id"),
                )
                if body.get("render"):
                    result["rendered"] = generate_correction(
                        self.db_path,
                        model=str(body.get("model") or self.default_model),
                        attempt_id=attempt_id,
                        output_dir=self.object_root.parent / "correction-audits",
                        force_render=True,
                    )
                return HTTPStatus.CREATED, result
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
            except (TypeError, ValueError) as exc:
                return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        if method == "POST" and parsed.path.startswith("/v1/errors/") and parsed.path.endswith("/correction"):
            attempt_id = parsed.path.removeprefix("/v1/errors/").removesuffix("/correction").strip("/")
            body = body or {}
            request_model = str(body.get("model") or self.default_model)
            try:
                result = submit_correction_version(
                    self.db_path,
                    attempt_id=attempt_id,
                    model=request_model,
                    corrected_solution=str(body.get("corrected_solution", "")),
                    corrected_steps=body.get("corrected_steps"),
                    final_answer=body.get("final_answer"),
                    verification_expression=body.get("verification_expression"),
                    confidence=float(body.get("confidence", 1.0)),
                    source=str(body.get("source", "human")),
                    created_by=str(body.get("created_by", "api-reviewer")),
                    activate=bool(body.get("activate", True)),
                    parent_version_id=body.get("parent_version_id"),
                )
                if body.get("render") and result["status"] == "active":
                    result["rendered"] = generate_correction(
                        self.db_path,
                        model=request_model,
                        attempt_id=attempt_id,
                        output_dir=self.object_root.parent / "correction-audits",
                    )
                return HTTPStatus.CREATED, result
            except KeyError as exc:
                return HTTPStatus.NOT_FOUND, {"error": str(exc)}
            except (TypeError, ValueError) as exc:
                return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        return HTTPStatus.NOT_FOUND, {"error": "route not found"}


def _handler(api: MemoryApi):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_content(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/student", "/student/"):
                self._send_content(
                    HTTPStatus.OK,
                    student_ui_html().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            if parsed.path in ("/annotation", "/annotation/"):
                self._send_content(
                    HTTPStatus.OK,
                    annotation_ui_html().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            if parsed.path in ("/review", "/review/"):
                self._send_content(
                    HTTPStatus.OK,
                    review_ui_html().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            segments = [unquote(item) for item in parsed.path.strip("/").split("/")]
            if (
                len(segments) == 4 and segments[:2] == ["v1", "attempts"]
                and segments[3] == "image"
            ):
                try:
                    content, mime_type = get_attempt_image(
                        api.db_path, attempt_id=segments[2]
                    )
                    self._send_content(HTTPStatus.OK, content, mime_type)
                except KeyError as exc:
                    self._send(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                except (OSError, RuntimeError, ValueError) as exc:
                    self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
                return
            if (
                len(segments) == 4 and segments[:2] == ["v1", "attempts"]
                and segments[3] == "audit"
            ):
                try:
                    model = parse_qs(parsed.query).get("model", [api.default_model])[0]
                    result = generate_correction(
                        api.db_path,
                        model=model,
                        attempt_id=segments[2],
                        output_dir=api.object_root.parent / "correction-audits",
                    )
                    artifact = result.get("audit_artifact")
                    if not artifact or not Path(artifact).exists():
                        raise KeyError("audit image is not available")
                    self._send_content(
                        HTTPStatus.OK, Path(artifact).read_bytes(), "image/png"
                    )
                except KeyError as exc:
                    self._send(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                except (OSError, RuntimeError, ValueError) as exc:
                    self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
                return
            status, payload = api.dispatch("GET", self.path)
            self._send(status, payload)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 30 * 1024 * 1024:
                    self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request exceeds 30 MiB"})
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                status, payload = api.dispatch("POST", self.path, body)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                status, payload = HTTPStatus.BAD_REQUEST, {"error": str(exc)}
            self._send(status, payload)

        def log_message(self, format: str, *args) -> None:
            print(f"[memory-api] {format % args}")

    return Handler


def serve(
    db_path: str | Path,
    *,
    model: str,
    host: str,
    port: int,
    object_root: str | Path | None = None,
    embedding_client=None,
    vector_weight: float = 0.65,
) -> None:
    api = MemoryApi(
        db_path,
        default_model=model,
        object_root=object_root,
        embedding_client=embedding_client,
        vector_weight=vector_weight,
    )
    server = ThreadingHTTPServer((host, port), _handler(api))
    print(json.dumps({"status": "listening", "url": f"http://{host}:{port}", "model": model}))
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="m3-edu-memory-api")
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--object-root")
    parser.add_argument("--embedding-config")
    parser.add_argument("--embedding-profile")
    parser.add_argument("--vector-weight", type=float, default=0.65)
    args = parser.parse_args(argv)
    if bool(args.embedding_config) != bool(args.embedding_profile):
        parser.error("--embedding-config and --embedding-profile must be used together")
    embedding_client = (
        create_embedding_client(
            args.embedding_config, profile_name=args.embedding_profile
        )
        if args.embedding_config else None
    )
    serve(
        args.db, model=args.model, host=args.host, port=args.port,
        object_root=args.object_root,
        embedding_client=embedding_client,
        vector_weight=args.vector_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
