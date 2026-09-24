import base64
import io
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from m3_edu_memory.control_tools import (
    EducationControlAgent,
    expand_concepts,
    filter_attempts,
    generate_correction,
    get_evidence,
    search_memories,
)
from m3_edu_memory.database import connect, initialize
from m3_edu_memory.diagnosis import diagnose_attempt
from m3_edu_memory.server import MemoryApi
from m3_edu_memory.review_ui import review_ui_html
from m3_edu_memory.ingestion import (
    get_ingestion_job,
    ingestion_metrics,
    process_next_job,
    recover_expired_jobs,
)
from m3_edu_memory.retrieval import (
    build_memory_index,
    hybrid_search,
    index_memory_attempt,
    memory_index_stats,
)
from m3_edu_memory.vlm import FixtureVisionClient


FIXTURE = {
    "question_transcription": "Simplify 1/3 + 1/6",
    "solution_transcription": "1/3 + 1/6 = 2/9",
    "steps": [{
        "step_index": 0,
        "transcription": "1/3 + 1/6 = 2/9",
        "normalized_latex": "1/3+1/6=2/9",
        "bbox": [0.1, 0.2, 0.8, 0.5],
        "is_error": True,
        "confidence": 0.9,
    }],
    "knowledge_points": ["fraction addition"],
    "has_error": True,
    "error_type": "arithmetic",
    "first_error_step": 0,
    "error_explanation": "分母不能直接相加。",
    "correction": {
        "corrected_solution": "1/3 + 1/6 = 1/2",
        "corrected_steps": [{
            "step_index": 0,
            "latex": "1/3+1/6=1/2",
            "explanation": "通分。",
        }],
        "final_answer": "1/2",
        "verification_expression": "1/3 + 1/6 = 1/2",
        "confidence": 0.95,
    },
    "confidence": 0.9,
    "requires_review": False,
}


class FailingVisionClient:
    provider = "fixture"
    model = "failing-vlm"

    def analyze(self, *, image_bytes: bytes, question_text: str):
        del image_bytes, question_text
        raise RuntimeError("simulated provider failure")


class ControlToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Path(self.temp_dir.name) / "memory.db"
        connection = connect(self.db)
        initialize(connection)
        self.attempt_id = "fermat-control-000001"
        connection.execute(
            "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.attempt_id, "rev", "shard.parquet", 1, "custom", 1,
                "hash", "parquet:///tmp/a.parquet#row=1", "c12",
                "2025-06-01T00:00:00+08:00", "simulated_from_grade", "c12",
                "v1", "alg", "epr", "question", 1, 1, "0", 0,
            ),
        )
        connection.execute(
            "INSERT INTO memory_nodes(node_id,node_type,label) VALUES(?,?,?)",
            (f"attempt:{self.attempt_id}", "Attempt", self.attempt_id),
        )
        connection.commit()
        connection.close()
        with patch("m3_edu_memory.diagnosis.image_bytes_from_source", return_value=b"image"):
            diagnose_attempt(
                self.db,
                attempt_id=self.attempt_id,
                client=FixtureVisionClient(FIXTURE),
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_control_tools_preserve_evidence_ids(self):
        filtered = filter_attempts(
            self.db, model="fixture-math-vlm", domain_code="alg", errors_only=True
        )
        self.assertEqual(filtered["attempts"][0]["attempt_id"], self.attempt_id)
        expanded = expand_concepts(
            self.db, query="代数错题", model="fixture-math-vlm"
        )
        self.assertEqual(expanded["domain_code"], "alg")
        searched = search_memories(
            self.db,
            model="fixture-math-vlm",
            query="fraction addition",
            candidate_attempt_ids=[self.attempt_id],
        )
        self.assertEqual(searched["results"][0]["attempt_id"], self.attempt_id)
        evidence = get_evidence(
            self.db, model="fixture-math-vlm", attempt_id=self.attempt_id
        )
        self.assertEqual(evidence["correction"]["final_answer"], "1/2")
        correction = generate_correction(
            self.db, model="fixture-math-vlm", attempt_id=self.attempt_id
        )
        self.assertEqual(correction["status"], "verified")

    def test_control_agent_records_tool_trace(self):
        result = EducationControlAgent(
            self.db, model="fixture-math-vlm"
        ).query("帮我整理代数错题")
        self.assertEqual(result["plan"]["domain_code"], "alg")
        self.assertEqual(result["evidence"][0]["attempt_id"], self.attempt_id)
        self.assertEqual(
            result["weakness_summary"]["source"],
            "derived_from_vlm_diagnoses",
        )
        self.assertIn("大模型诊断", result["weakness_summary"]["text"])
        self.assertEqual(
            [entry["tool"] for entry in result["tool_trace"]],
            ["expand_concepts", "filter_attempts", "search_memories", "get_evidence"],
        )
        recent = EducationControlAgent(
            self.db, model="fixture-math-vlm"
        ).query("帮我整理最近三个月的代数错题")
        self.assertEqual(recent["plan"]["recent_months"], 3)
        self.assertTrue(recent["resolved_time_window"]["time_from"].startswith("2025-04-01"))
        self.assertEqual(len(recent["evidence"]), 1)

    def test_control_agent_ranks_complete_filtered_candidate_set(self):
        second_id = "fermat-control-000002"
        connection = connect(self.db)
        connection.execute(
            "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                second_id, "rev", "shard.parquet", 2, "custom-2", 2,
                "hash-2", "parquet:///tmp/a.parquet#row=2", "c12",
                "2025-06-02T00:00:00+08:00", "simulated_from_grade", "c12",
                "v1", "alg", "epr", "question 2", 1, 1, "0", 0,
            ),
        )
        connection.execute(
            "INSERT INTO memory_nodes(node_id,node_type,label) VALUES(?,?,?)",
            (f"attempt:{second_id}", "Attempt", second_id),
        )
        connection.commit()
        connection.close()
        with patch("m3_edu_memory.diagnosis.image_bytes_from_source", return_value=b"image"):
            diagnose_attempt(
                self.db,
                attempt_id=second_id,
                client=FixtureVisionClient(FIXTURE),
            )
        result = EducationControlAgent(
            self.db, model="fixture-math-vlm"
        ).query("帮我整理1道代数错题")
        filter_trace = next(
            item for item in result["tool_trace"] if item["tool"] == "filter_attempts"
        )
        self.assertEqual(filter_trace["result_count"], 2)
        self.assertEqual(len(result["evidence"]), 1)

    def test_hybrid_index_and_control_backend(self):
        indexed = build_memory_index(self.db, model="fixture-math-vlm")
        self.assertEqual(indexed["indexed_documents"], 1)
        self.assertEqual(
            memory_index_stats(self.db, model="fixture-math-vlm")["documents"], 1
        )
        search = hybrid_search(
            self.db,
            model="fixture-math-vlm",
            query="fraction addition",
        )
        self.assertEqual(search["results"][0]["attempt_id"], self.attempt_id)
        self.assertIn("vector_score", search["results"][0])
        result = EducationControlAgent(
            self.db, model="fixture-math-vlm"
        ).query("帮我整理代数错题")
        self.assertEqual(result["search_backend"], "sqlite_fts5+local_hash_vector")

    def test_incremental_index_refresh(self):
        refreshed = index_memory_attempt(
            self.db, model="fixture-math-vlm", attempt_id=self.attempt_id
        )
        self.assertEqual(refreshed["attempt_id"], self.attempt_id)
        self.assertEqual(
            memory_index_stats(self.db, model="fixture-math-vlm")["documents"], 1
        )

    def test_http_api_dispatch(self):
        api = MemoryApi(self.db, default_model="fixture-math-vlm")
        status, health = api.dispatch("GET", "/health")
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(health["status"], "ok")
        status, metrics = api.dispatch(
            "GET", "/v1/analysis/metrics?model=fixture-math-vlm"
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(metrics["completed"], 1)
        status, history = api.dispatch(
            "GET",
            "/v1/memory/mastery-history?model=fixture-math-vlm&domain=alg",
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(history["total"], 1)
        self.assertEqual(history["snapshots"][0]["change_reason"].split(":")[0], "diagnosis")
        status, result = api.dispatch(
            "POST", "/v1/memory/query", {"request": "代数错题", "limit": 5}
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["evidence"][0]["attempt_id"], self.attempt_id)
        status, attempt = api.dispatch(
            "GET", f"/v1/attempts/{self.attempt_id}"
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(attempt["attempt_id"], self.attempt_id)
        self.assertIn("analysis_observability", attempt)
        self.assertIn("教师修订错题报告", review_ui_html())
        self.assertIn("/v1/errors/", review_ui_html())

    def test_review_override_changes_effective_memory_without_erasing_model_output(self):
        api = MemoryApi(self.db, default_model="fixture-math-vlm")
        status, review = api.dispatch(
            "POST",
            f"/v1/errors/{self.attempt_id}/review",
            {
                "verdict": "rejected",
                "reviewer": "teacher-1",
                "notes": "人工复核：原作答正确。",
            },
        )
        self.assertEqual(status, HTTPStatus.CREATED)
        self.assertEqual(review["has_error_override"], False)
        evidence = get_evidence(
            self.db, model="fixture-math-vlm", attempt_id=self.attempt_id
        )
        self.assertEqual(evidence["has_error_pred"], 1)
        self.assertEqual(evidence["has_error_effective"], 0)
        self.assertEqual(evidence["review"]["verdict"], "rejected")
        self.assertEqual(evidence["review"]["reviewer"], "teacher-1")
        filtered = filter_attempts(
            self.db, model="fixture-math-vlm", domain_code="alg", errors_only=True
        )
        self.assertEqual(filtered["returned"], 0)

    def test_teacher_can_replace_location_knowledge_and_explanation(self):
        api = MemoryApi(self.db, default_model="fixture-math-vlm")
        status, review = api.dispatch(
            "POST",
            f"/v1/errors/{self.attempt_id}/review",
            {
                "verdict": "modified",
                "reviewer": "teacher-2",
                "has_error": True,
                "error_type": "transcription",
                "first_error_step": 1,
                "error_bbox": [0.2, 0.3, 0.7, 0.6],
                "knowledge_points": ["Polynomial subtraction"],
                "error_explanation": "教师修订：变量被誊写错误。",
                "notes": "查询报告人工修订",
            },
        )
        self.assertEqual(status, HTTPStatus.CREATED)
        self.assertEqual(review["first_error_step_override"], 1)
        evidence = get_evidence(
            self.db, model="fixture-math-vlm", attempt_id=self.attempt_id
        )
        self.assertEqual(evidence["display_source"], "teacher_override")
        self.assertEqual(evidence["first_error_step_effective"], 1)
        self.assertEqual(evidence["error_bbox_effective"], [0.2, 0.3, 0.7, 0.6])
        self.assertEqual(evidence["knowledge_points"], ["Polynomial subtraction"])
        self.assertEqual(evidence["knowledge_points_pred"], ["fraction addition"])
        self.assertEqual(evidence["error_explanation"], "教师修订：变量被誊写错误。")
        self.assertNotEqual(
            evidence["error_explanation"], evidence["error_explanation_pred"]
        )

    def test_versioned_human_correction_becomes_effective_without_overwriting_vlm(self):
        api = MemoryApi(self.db, default_model="fixture-math-vlm")
        status, created = api.dispatch(
            "POST",
            f"/v1/errors/{self.attempt_id}/correction",
            {
                "source": "human",
                "created_by": "teacher-1",
                "corrected_solution": "Use a common denominator: 1/3 + 1/6 = 1/2",
                "corrected_steps": [{
                    "step_index": 0,
                    "latex": "1/3+1/6=1/2",
                    "explanation": "通分后相加。",
                }],
                "final_answer": "1/2",
                "verification_expression": "1/3 + 1/6 = 1/2",
                "activate": True,
            },
        )
        self.assertEqual(status, HTTPStatus.CREATED)
        self.assertEqual(created["verification"]["status"], "verified")
        evidence = get_evidence(
            self.db, model="fixture-math-vlm", attempt_id=self.attempt_id
        )
        self.assertEqual(evidence["correction"]["source"], "human")
        self.assertEqual(evidence["correction"]["version_id"], created["version_id"])
        connection = connect(self.db)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0], 1)
        connection.close()
        status, history = api.dispatch(
            "GET", f"/v1/errors/{self.attempt_id}/corrections?model=fixture-math-vlm"
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(len(history["versions"]), 2)
        self.assertEqual(
            {item["status"] for item in history["versions"]},
            {"active", "superseded"},
        )

        status, draft = api.dispatch(
            "POST",
            f"/v1/errors/{self.attempt_id}/correction",
            {
                "source": "rule",
                "corrected_solution": "draft alternative",
                "activate": False,
            },
        )
        self.assertEqual(status, HTTPStatus.CREATED)
        self.assertEqual(draft["status"], "draft")
        evidence = get_evidence(
            self.db, model="fixture-math-vlm", attempt_id=self.attempt_id
        )
        self.assertEqual(evidence["correction"]["version_id"], created["version_id"])

    def test_uploaded_attempt_is_idempotent_and_processed_by_worker(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (80, 60), "white").save(image_buffer, format="PNG")
        payload = {
            "image_base64": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
            "question_text": "Simplify 1/3 + 1/6",
            "domain_code": "alg",
            "subdomain_code": "epr",
            "grade": "c12",
            "idempotency_key": "upload-test-1",
        }
        api = MemoryApi(
            self.db,
            default_model="fixture-math-vlm",
            object_root=Path(self.temp_dir.name) / "objects",
        )
        status, queued = api.dispatch("POST", "/v1/attempts", payload)
        self.assertEqual(status, HTTPStatus.ACCEPTED)
        self.assertEqual(queued["status"], "queued")
        status, replay = api.dispatch("POST", "/v1/attempts", payload)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["attempt_id"], queued["attempt_id"])

        processed = process_next_job(
            self.db, client=FixtureVisionClient(FIXTURE)
        )
        self.assertEqual(processed["status"], "completed")
        job = get_ingestion_job(self.db, job_id=queued["job_id"])
        self.assertEqual(job["status"], "completed")
        status, attempt = api.dispatch(
            "GET", f"/v1/attempts/{queued['attempt_id']}"
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(attempt["processing"]["event_time_source"], "uploaded_or_provided")
        self.assertTrue(Path(attempt["processing"]["object_path"]).exists())

    def test_worker_retries_then_dead_letters_and_can_be_manually_retried(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(image_buffer, format="PNG")
        api = MemoryApi(
            self.db,
            default_model="failing-vlm",
            object_root=Path(self.temp_dir.name) / "objects",
        )
        status, queued = api.dispatch(
            "POST",
            "/v1/attempts",
            {
                "image_base64": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
                "question_text": "q",
                "domain_code": "alg",
                "idempotency_key": "failing-job",
                "max_attempts": 2,
            },
        )
        self.assertEqual(status, HTTPStatus.ACCEPTED)
        first = process_next_job(
            self.db, client=FailingVisionClient(), retry_base_seconds=0
        )
        self.assertEqual(first["status"], "retry_scheduled")
        second = process_next_job(
            self.db, client=FailingVisionClient(), retry_base_seconds=0
        )
        self.assertEqual(second["status"], "failed")
        metrics = ingestion_metrics(self.db)
        self.assertEqual(metrics["dead_lettered_jobs"], 1)
        metrics_status, api_metrics = api.dispatch("GET", "/v1/jobs/metrics")
        self.assertEqual(metrics_status, HTTPStatus.OK)
        self.assertEqual(api_metrics["dead_lettered_jobs"], 1)
        status, retried = api.dispatch(
            "POST", f"/v1/jobs/{queued['job_id']}/retry", {}
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(retried["attempt_count"], 0)

    def test_expired_worker_lease_is_recovered(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(image_buffer, format="PNG")
        api = MemoryApi(
            self.db,
            default_model="fixture-math-vlm",
            object_root=Path(self.temp_dir.name) / "objects",
        )
        _, queued = api.dispatch(
            "POST",
            "/v1/attempts",
            {
                "image_base64": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
                "question_text": "q",
                "domain_code": "alg",
                "idempotency_key": "expired-lease-job",
            },
        )
        connection = connect(self.db)
        with connection:
            connection.execute(
                """UPDATE ingestion_jobs SET status='processing',attempt_count=1,
                          lease_owner='dead-worker',lease_expires_at='2000-01-01T00:00:00+00:00'
                   WHERE job_id=?""",
                (queued["job_id"],),
            )
        connection.close()
        recovered = recover_expired_jobs(self.db)
        self.assertEqual(recovered["requeued"], 1)
        self.assertEqual(
            get_ingestion_job(self.db, job_id=queued["job_id"])["status"], "queued"
        )


if __name__ == "__main__":
    unittest.main()
