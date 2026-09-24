import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from m3_edu_memory.database import connect, initialize
from m3_edu_memory.control import plan_memory_query
from m3_edu_memory.diagnosis import analysis_run_metrics, diagnose_attempt
from m3_edu_memory.image_preprocessing import (
    ImagePreparationPolicy,
    prepare_image_for_vlm,
)
from m3_edu_memory.memory_writer import rebuild_mastery_states
from m3_edu_memory.query import query_vlm_memories
from m3_edu_memory.weakness import extract_vlm_weaknesses
from m3_edu_memory.verification import verify_correction
from m3_edu_memory.vlm import (
    PROMPT_VERSION,
    FixtureVisionClient,
    parse_json_object,
    validate_diagnosis,
)


FIXTURE = {
    "question_transcription": "q",
    "solution_transcription": "s",
    "steps": [{
        "step_index": 0,
        "transcription": "step",
        "normalized_latex": "x=1",
        "bbox": [0.1, 0.2, 0.8, 0.9],
        "is_error": True,
        "confidence": 0.8,
    }],
    "knowledge_points": ["limits"],
    "has_error": True,
    "error_type": "conceptual",
    "first_error_step": 0,
    "error_explanation": "explanation",
    "correction": {
        "corrected_solution": "1/3 + 1/6 = 1/2",
        "corrected_steps": [{
            "step_index": 0,
            "latex": "\\frac{1}{3}+\\frac{1}{6}=\\frac{1}{2}",
            "explanation": "use a common denominator",
        }],
        "final_answer": "1/2",
        "verification_expression": "1/3 + 1/6 = 1/2",
        "confidence": 0.9,
    },
    "confidence": 0.8,
    "requires_review": False,
}


class VlmTests(unittest.TestCase):
    def test_image_preprocessing_respects_provider_budget(self):
        source = io.BytesIO()
        Image.new("RGB", (1800, 1200), "white").save(source, format="BMP")
        prepared, metadata = prepare_image_for_vlm(
            source.getvalue(),
            policy=ImagePreparationPolicy(
                max_long_edge=1400,
                max_pixels=1_500_000,
                max_bytes=500_000,
                jpeg_quality=90,
                min_jpeg_quality=70,
            ),
        )
        self.assertTrue(metadata["processed"])
        self.assertLessEqual(metadata["output_width"], 1400)
        self.assertLessEqual(
            metadata["output_width"] * metadata["output_height"], 1_500_000
        )
        self.assertLessEqual(len(prepared), 500_000)
        self.assertEqual(metadata["output_format"], "jpeg")
        self.assertTrue(metadata["normalized_bbox_preserved"])

    def test_image_preprocessing_can_be_disabled_per_model(self):
        source = io.BytesIO()
        Image.new("RGB", (320, 200), "white").save(source, format="PNG")
        original = source.getvalue()
        prepared, metadata = prepare_image_for_vlm(
            original, policy=ImagePreparationPolicy(enabled=False)
        )
        self.assertEqual(prepared, original)
        self.assertEqual(metadata["reason"], "disabled")

    def test_chinese_memory_request_is_planned(self):
        plan = plan_memory_query("帮我整理过去在微积分上的10道错题")
        self.assertEqual(plan.domain_code, "clc")
        self.assertTrue(plan.errors_only)
        self.assertEqual(plan.limit, 10)

    def test_recent_months_are_parsed_from_chinese_request(self):
        plan = plan_memory_query("帮我整理过去三月的微积分错题")
        self.assertEqual(plan.domain_code, "clc")
        self.assertEqual(plan.recent_months, 3)
        self.assertTrue(plan.errors_only)

    def test_mastery_identity_includes_subdomain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "memory.db"
            connection = connect(db)
            initialize(connection)
            cases = (
                ("lim", "shared point"),
                ("der", "shared point"),
                ("int", "Custom Point"),
                ("int", "custom point"),
            )
            for index, (subdomain, point) in enumerate(cases, start=1):
                attempt_id = f"fermat-test-{index:06d}"
                connection.execute(
                    "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        attempt_id, "rev", "shard.parquet", index, "custom", index,
                        f"hash-{index}", f"parquet:///tmp/a.parquet#row={index}",
                        "c12", f"2025-0{index}-01T00:00:00+08:00",
                        "simulated_from_grade", "c12", "v1", "clc", subdomain,
                        "question", 1, 1, "0", 0,
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_nodes(node_id,node_type,label) VALUES(?,?,?)",
                    (f"attempt:{attempt_id}", "Attempt", attempt_id),
                )
                run_id = f"run-{index}"
                connection.execute(
                    """INSERT INTO analysis_runs(
                         run_id,attempt_id,provider,model,prompt_version,status,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        run_id, attempt_id, "fixture", "model", PROMPT_VERSION,
                        "completed", f"2025-0{index}-01",
                    ),
                )
                connection.execute(
                    """INSERT INTO diagnoses(
                         diagnosis_id,run_id,attempt_id,question_transcription,
                         solution_transcription,has_error_pred,error_type_pred,
                         error_explanation,knowledge_points_json,confidence,
                         requires_review)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        f"diagnosis-{index}", run_id, attempt_id, "q", "s", 1,
                        "conceptual", "e", json.dumps([point]), 0.8, 0,
                    ),
                )
            rebuild_mastery_states(
                connection, model="model", prompt_version=PROMPT_VERSION
            )
            ids = [
                row[0]
                for row in connection.execute(
                    "SELECT mastery_id FROM mastery_states ORDER BY mastery_id"
                )
            ]
            self.assertEqual(len(ids), 3)
            self.assertEqual(len(set(ids)), 3)
            merged = connection.execute(
                """SELECT attempt_count FROM mastery_states
                   WHERE subdomain_code='int'"""
            ).fetchone()
            self.assertEqual(merged[0], 2)
            connection.close()

    def test_exact_numeric_correction_verification(self):
        result = verify_correction(FIXTURE["correction"])
        self.assertEqual(result["status"], "verified")

    def test_inconsistent_numeric_correction_is_rejected(self):
        result = verify_correction({"verification_expression": "1/3 + 1/6 = 2/3"})
        self.assertEqual(result["status"], "inconsistent")

    def test_symbolic_identity_correction_verification(self):
        result = verify_correction(
            {"verification_expression": "(x+1)^2=x^2+2*x+1"}
        )
        self.assertEqual(result["status"], "verified")
        self.assertIn(
            result["method"],
            {"sympy_symbolic_identity", "rational_probe_identity"},
        )

    def test_matrix_correction_verification(self):
        result = verify_correction({
            "verification_expression": (
                "matrix([[0,-1],[0,2]]) * matrix([[3,5],[0,0]]) "
                "= matrix([[0,0],[0,0]])"
            )
        })
        self.assertEqual(result["status"], "verified")
        self.assertIn(result["method"], {"fraction_matrix_exact", "sympy_exact"})

    def test_symbolic_derivative_and_antiderivative_verification(self):
        derivative = verify_correction(
            {"verification_expression": "diff(x^3,x)=3*x^2"}
        )
        antiderivative = verify_correction(
            {"verification_expression": "integrate(2*x,x)=x^2+7"}
        )
        self.assertEqual(derivative["status"], "verified")
        self.assertIn(
            derivative["method"], {"sympy_derivative", "rational_probe_derivative"}
        )
        self.assertEqual(antiderivative["status"], "verified")
        self.assertIn(
            antiderivative["method"],
            {"sympy_antiderivative", "rational_probe_antiderivative"},
        )

    def test_compound_limit_correction_verification(self):
        try:
            import sympy
        except ImportError:
            self.skipTest("SymPy is unavailable")
        if not hasattr(sympy, "limit"):
            self.skipTest("SymPy runtime is unavailable")
        result = verify_correction({
            "verification_expression": (
                "limit((x^15-1)/(x^10-1),x,1)=3/2,"
                "limit((sqrt(1+x)-1)/x,x,0)=1/2"
            )
        })

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["method"], "compound_claims")
        self.assertEqual(
            [claim["method"] for claim in result["details"]["claims"]],
            ["sympy_limit", "sympy_limit"],
        )

    def test_unsafe_symbolic_expression_is_not_executed(self):
        result = verify_correction(
            {"verification_expression": "__import__(os)=1"}
        )
        self.assertEqual(result["status"], "unverifiable")

    def test_parses_fenced_json(self):
        parsed = parse_json_object("```json\n" + json.dumps(FIXTURE) + "\n```")
        self.assertEqual(validate_diagnosis(parsed)["error_type"], "conceptual")

    def test_repairs_unescaped_latex_backslashes(self):
        parsed = parse_json_object(
            r'{"latex":"\frac{x}{y}","note":"\theta+\operatorname{sin}(x)"}'
        )
        self.assertEqual(parsed["latex"], r"\frac{x}{y}")
        self.assertEqual(parsed["note"], r"\theta+\operatorname{sin}(x)")

    def test_partial_correction_is_retained_for_review(self):
        partial = json.loads(json.dumps(FIXTURE))
        partial["requires_review"] = False
        partial["correction"] = {
            "corrected_steps": [{
                "step_index": 0,
                "latex": "1/3+1/6=1/2",
                "explanation": "通分。",
            }],
            "confidence": 0.7,
        }
        validated = validate_diagnosis(partial)
        self.assertTrue(validated["requires_review"])
        self.assertIsNone(validated["correction"]["final_answer"])
        self.assertIn(
            "1/3+1/6=1/2", validated["correction"]["corrected_solution"]
        )

    def test_invalid_bbox_and_null_explanation_require_review_without_dropping_run(self):
        payload = json.loads(json.dumps(FIXTURE))
        payload["steps"][0]["bbox"] = [25, 50, 600, 900]
        payload["error_explanation"] = None
        payload["requires_review"] = False
        validated = validate_diagnosis(payload)
        self.assertIsNone(validated["steps"][0]["bbox"])
        self.assertEqual(validated["error_explanation"], "模型未提供可用的错误说明。")
        self.assertTrue(validated["requires_review"])

    def test_fixture_diagnosis_is_versioned_and_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "memory.db"
            connection = connect(db)
            initialize(connection)
            attempt_id = "fermat-test-000001"
            values = (
                attempt_id, "rev", "shard.parquet", 1, "custom", 1, "hash",
                "parquet:///tmp/a.parquet#row=1", "c12",
                "2025-06-01T00:00:00+08:00", "simulated_from_grade", "c12", "v1",
                "clc", "lim", "question", 1, 1, "0", 0,
            )
            connection.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
            connection.execute(
                "INSERT INTO memory_nodes(node_id,node_type,label) VALUES(?,?,?)",
                (f"attempt:{attempt_id}", "Attempt", attempt_id),
            )
            connection.commit()
            connection.close()

            image_buffer = io.BytesIO()
            Image.new("RGB", (640, 480), "white").save(image_buffer, format="PNG")
            with patch(
                "m3_edu_memory.diagnosis.image_bytes_from_source",
                return_value=image_buffer.getvalue(),
            ):
                result = diagnose_attempt(
                    db,
                    attempt_id=attempt_id,
                    client=FixtureVisionClient(FIXTURE),
                )
            self.assertEqual(result["error_type"], "conceptual")
            connection = connect(db)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM diagnosis_steps").fetchone()[0], 1)
            correction = connection.execute("SELECT * FROM corrections").fetchone()
            self.assertEqual(correction["verification_status"], "verified")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mastery_states").fetchone()[0], 1)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM mastery_state_history"
                ).fetchone()[0],
                1,
            )
            rebuild_mastery_states(
                connection,
                model="fixture-math-vlm",
                prompt_version=PROMPT_VERSION,
                change_reason="repeat-without-change",
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM mastery_state_history"
                ).fetchone()[0],
                1,
            )
            run = connection.execute("SELECT * FROM analysis_runs").fetchone()
            self.assertIsNotNone(run["preprocessing_json"])
            self.assertIsNotNone(run["latency_ms"])
            self.assertEqual(run["input_width"], 640)
            self.assertEqual(run["input_height"], 480)
            self.assertEqual(
                connection.execute(
                    "SELECT source FROM episodic_memories WHERE attempt_id=?", (attempt_id,)
                ).fetchone()[0],
                f"vlm:fixture:fixture-math-vlm:{PROMPT_VERSION}",
            )
            connection.close()
            metrics = analysis_run_metrics(db, model="fixture-math-vlm")
            self.assertEqual(metrics["completed"], 1)
            self.assertEqual(metrics["failed"], 0)
            self.assertIsNotNone(metrics["average_latency_ms"])
            retrieved = query_vlm_memories(
                db,
                model="fixture-math-vlm",
                domain_code="clc",
                errors_only=True,
            )
            self.assertEqual(retrieved["total"], 1)
            self.assertEqual(retrieved["attempts"][0]["first_error_step"], 0)
            self.assertEqual(retrieved["attempts"][0]["steps"][0]["bbox"], [0.1, 0.2, 0.8, 0.9])
            self.assertEqual(
                retrieved["attempts"][0]["correction"]["final_answer"], "1/2"
            )
            weakness = extract_vlm_weaknesses(
                db, model="fixture-math-vlm", domain_code="clc"
            )
            self.assertEqual(len(weakness["weaknesses"]), 1)
            self.assertEqual(weakness["weaknesses"][0]["knowledge_point"], "limits")


if __name__ == "__main__":
    unittest.main()
