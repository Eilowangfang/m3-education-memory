import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

from PIL import Image

from m3_edu_memory.annotation_ui import annotation_ui_html
from m3_edu_memory.annotations import (
    annotation_progress,
    get_attempt_image,
    list_evaluation_annotations,
    list_evaluation_cases,
    submit_evaluation_annotation,
)
from m3_edu_memory.database import connect, initialize
from m3_edu_memory.evaluation import (
    answers_equivalent,
    compare_evaluation_runs,
    create_evaluation_suite,
    evaluate_model,
    get_evaluation_suite,
)
from m3_edu_memory.providers import create_vision_client, load_provider_profiles
from m3_edu_memory.server import MemoryApi, _handler
from m3_edu_memory.vlm import PROMPT_VERSION


class EvaluationTests(unittest.TestCase):
    def test_final_answers_use_safe_mathematical_equivalence(self):
        self.assertTrue(answers_equivalent("1/2", r"\frac{2}{4}"))
        self.assertTrue(answers_equivalent("75.46", r"75.46\,\mathrm{m}^2"))
        self.assertTrue(answers_equivalent("x^2+2*x+1", "(x+1)^2"))
        self.assertFalse(answers_equivalent("1/3", r"\frac{1}{2}"))

    def _database(self, directory: str) -> Path:
        db = Path(directory) / "memory.db"
        connection = connect(db)
        initialize(connection)
        for index in range(4):
            attempt_id = f"eval-{index}"
            truth = int(index % 2 == 0)
            connection.execute(
                """INSERT INTO attempts(
                     attempt_id,dataset_revision,shard,row_index,custom_id,img_id,
                     image_sha256,image_source_path,grade,Time,time_source,time_bucket,
                     simulation_seed_version,domain_code,subdomain_code,orig_q,
                     handwriting_style,image_quality,rotation,is_table)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    attempt_id, "rev", "shard", index, attempt_id, index,
                    f"hash-{index}", f"file-{index}.png", "c12",
                    f"2025-06-0{index + 1}T00:00:00+08:00",
                    "simulated_from_grade", "c12", "v1", "clc", "der",
                    f"unique question {index}", 1, 1, "0", 0,
                ),
            )
            connection.execute(
                """INSERT INTO benchmark_truth(
                     attempt_id,has_error,orig_a,pert_a,pert_reasoning,
                     reference_error_type) VALUES(?,?,?,?,?,?)""",
                (
                    attempt_id, truth, "2" if truth else "answer", "perturbed", "reason",
                    "arithmetic" if truth else "no_actual_error",
                ),
            )
            for model in ("model-good", "model-bad"):
                run_id = f"{model}-{index}"
                prediction = truth if model == "model-good" else 1 - truth
                error_type = "arithmetic" if prediction else "no_actual_error"
                connection.execute(
                    """INSERT INTO analysis_runs(
                         run_id,attempt_id,provider,model,prompt_version,status,
                         created_at) VALUES(?,?,?,?,?,'completed',?)""",
                    (run_id, attempt_id, "fixture", model, PROMPT_VERSION,
                     f"2026-01-0{index + 1}T00:00:00+00:00"),
                )
                connection.execute(
                    """INSERT INTO diagnoses(
                         diagnosis_id,run_id,attempt_id,question_transcription,
                         solution_transcription,has_error_pred,error_type_pred,
                         first_error_step,error_explanation,knowledge_points_json,
                         confidence,requires_review)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        f"diagnosis-{run_id}", run_id, attempt_id, "q", "s",
                        prediction, error_type, 0 if prediction else None, "e",
                        '["derivative"]', 0.9, 0,
                    ),
                )
                if prediction:
                    connection.execute(
                        """INSERT INTO diagnosis_steps(
                             run_id,step_index,transcription,normalized_latex,
                             bbox_json,is_error,confidence)
                           VALUES(?,0,'step','x=1','[0.1,0.1,0.5,0.5]',1,0.9)""",
                        (run_id,),
                    )
                    connection.execute(
                        """INSERT INTO corrections(
                             correction_id,run_id,attempt_id,corrected_solution,
                             corrected_steps_json,final_answer,verification_expression,
                             verification_status,verification_method,
                             verification_details_json,confidence,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            f"correction-{run_id}", run_id, attempt_id, "1+1=2",
                            "[]", "2", "1+1=2", "verified", "sympy",
                            "{}", 0.9, "2026-01-01T00:00:00+00:00",
                        ),
                    )
        connection.commit()
        connection.close()
        return db

    def test_fixed_suite_metrics_are_persisted_and_compared(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = self._database(temp_dir)
            suite = create_evaluation_suite(
                db, name="pilot", limit=4, domain_code="clc", seed="stable"
            )
            self.assertEqual(suite["attempt_count"], 4)
            self.assertEqual(suite["error_count"], 2)
            self.assertEqual(get_evaluation_suite(db, suite="pilot")["attempt_count"], 4)

            for attempt_id in ("eval-0", "eval-2"):
                annotation = submit_evaluation_annotation(
                    db,
                    suite="pilot",
                    attempt_id=attempt_id,
                    annotator="teacher-1",
                    status="approved",
                    has_error=True,
                    first_error_step=0,
                    error_bbox=[0.1, 0.1, 0.5, 0.5],
                    knowledge_points=["derivative"],
                    corrected_final_answer="2",
                    correction_steps=[{"step_index": 0, "latex": "1+1=2"}],
                )
                self.assertEqual(annotation["status"], "approved")

            good = evaluate_model(db, model="model-good", suite="pilot", persist=True)
            bad = evaluate_model(db, model="model-bad", suite="pilot", persist=True)
            self.assertEqual(good["metrics"]["has_error_accuracy"], 1.0)
            self.assertEqual(good["metrics"]["error_bbox_coverage"], 1.0)
            self.assertEqual(good["metrics"]["verified_correction_rate"], 1.0)
            self.assertEqual(good["metrics"]["reference_answer_exact_match"], 1.0)
            self.assertEqual(
                good["metrics"]["reference_answer_equivalent_match"], 1.0
            )
            self.assertEqual(
                good["metrics"]["ground_truth_source"], "FERMAT benchmark_truth"
            )
            self.assertNotIn("approved_annotation_coverage", good["metrics"])
            self.assertNotIn("teacher_has_error_accuracy", good["metrics"])
            self.assertEqual(bad["metrics"]["has_error_accuracy"], 0.0)
            comparison = compare_evaluation_runs(db, suite="pilot")
            self.assertEqual(len(comparison["models"]), 2)
            self.assertEqual(annotation_progress(db, suite="pilot")["counts"]["approved"], 2)
            self.assertEqual(
                list_evaluation_cases(
                    db, suite="pilot", annotation_status="unannotated"
                )["total"],
                2,
            )
            history = list_evaluation_annotations(
                db, suite="pilot", attempt_id="eval-0"
            )
            self.assertEqual(len(history["versions"]), 1)

            # Teacher annotations remain available as review history, but they
            # never replace FERMAT ground truth in offline model evaluation.
            submit_evaluation_annotation(
                db, suite="pilot", attempt_id="eval-1", annotator="teacher-2",
                status="approved", has_error=True, first_error_step=3,
                error_bbox=[0.2, 0.2, 0.8, 0.8], knowledge_points=["wrong-label"],
                corrected_final_answer="wrong",
            )
            unchanged = evaluate_model(db, model="model-good", suite="pilot")
            self.assertEqual(unchanged["metrics"]["has_error_accuracy"], 1.0)
            self.assertEqual(unchanged["metrics"]["reference_answer_exact_match"], 1.0)

    def test_annotation_http_api_validates_approved_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = self._database(temp_dir)
            create_evaluation_suite(db, name="api-suite", limit=4)
            api = MemoryApi(db, default_model="model-good")
            status, invalid = api.dispatch(
                "POST",
                "/v1/evaluation-suites/api-suite/annotations/eval-0",
                {"annotator": "teacher", "status": "approved", "has_error": True},
            )
            self.assertEqual(status, 400)
            self.assertIn("first_error_step", invalid["error"])

            status, created = api.dispatch(
                "POST",
                "/v1/evaluation-suites/api-suite/annotations/eval-1",
                {"annotator": "teacher", "status": "approved", "has_error": False},
            )
            self.assertEqual(status, 201)
            self.assertEqual(created["status"], "approved")
            status, progress = api.dispatch(
                "GET", "/v1/evaluation-suites/api-suite/progress"
            )
            self.assertEqual(status, 200)
            self.assertEqual(progress["counts"]["approved"], 1)

    def test_annotation_ui_and_attempt_image_are_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = self._database(temp_dir)
            image_path = Path(temp_dir) / "attempt.png"
            Image.new("RGB", (40, 30), "white").save(image_path)
            connection = connect(db)
            connection.execute(
                "UPDATE attempts SET image_source_path=? WHERE attempt_id='eval-0'",
                (image_path.resolve().as_uri(),),
            )
            connection.commit()
            connection.close()
            content, mime_type = get_attempt_image(db, attempt_id="eval-0")
            self.assertTrue(content.startswith(b"\x89PNG"))
            self.assertEqual(mime_type, "image/png")
            image_path.unlink()
            cached, cached_mime = get_attempt_image(db, attempt_id="eval-0")
            self.assertEqual(cached, content)
            self.assertEqual(cached_mime, "image/png")
            html = annotation_ui_html()
            self.assertIn("手写数学教师标注台", html)
            self.assertIn("bboxCanvas", html)
            self.assertIn("/v1/evaluation-suites/", html)

            api = MemoryApi(db, default_model="model-good")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(api))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/annotation") as response:
                    self.assertIn("text/html", response.headers["Content-Type"])
                with urlopen(base + "/review?attempt_id=eval-0&model=model-good") as response:
                    self.assertIn("text/html", response.headers["Content-Type"])
                    self.assertIn("教师修订错题报告", response.read().decode("utf-8"))
                with urlopen(base + "/student") as response:
                    student_html = response.read().decode("utf-8")
                    self.assertIn("数学记忆", student_html)
                    self.assertIn("帮我整理过去三月的微积分错题", student_html)
                    self.assertIn("mathjax@3", student_html.casefold())
                    self.assertIn("mathRich", student_html)
                    self.assertIn("typesetMath()", student_html)
                    self.assertNotIn("grade 映射", student_html)
                with urlopen(base + "/v1/attempts/eval-0/image") as response:
                    self.assertEqual(response.headers["Content-Type"], "image/png")
                    self.assertTrue(response.read().startswith(b"\x89PNG"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_provider_profiles_reference_environment_variables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "providers.json"
            path.write_text(json.dumps({"profiles": {"demo": {
                "base_url": "https://example.test/v1",
                "model": "math-vlm",
                "api_key_env": "DEMO_API_KEY",
                "provider": "demo-provider",
                "request_options": {"thinking": {"type": "disabled"}},
            }}}), encoding="utf-8")
            profiles = load_provider_profiles(path)
            self.assertEqual(profiles["demo"]["api_key_env"], "DEMO_API_KEY")
            client = create_vision_client(path, profile_name="demo")
            self.assertEqual(client.provider, "demo-provider")
            self.assertEqual(client.model, "math-vlm")
            self.assertEqual(client.image_detail, "high")
            self.assertEqual(client.image_policy.max_pixels, 4_014_080)
            self.assertEqual(
                client.request_options, {"thinking": {"type": "disabled"}}
            )

            path.write_text(json.dumps({"profiles": {"unsafe": {
                "base_url": "https://example.test/v1",
                "model": "m", "api_key_env": "KEY", "api_key": "secret"
            }}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_provider_profiles(path)


if __name__ == "__main__":
    unittest.main()
