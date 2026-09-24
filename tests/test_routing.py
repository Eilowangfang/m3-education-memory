from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m3_edu_memory.database import connect, initialize
from m3_edu_memory.routing import (
    RoutingPolicy,
    decide_route,
    evaluate_routing_policy,
    materialize_routing_view,
    record_routing_decision,
    select_unrouted_attempt_ids,
)
from m3_edu_memory.query import query_vlm_memories


class RoutingTests(unittest.TestCase):
    def test_unmaterialized_decision_remains_retryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            db_path = Path(temporary) / "routing.db"
            connection = connect(db_path)
            initialize(connection)
            connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "retry-1", "rev", "shard", 1, "retry-1", 1, "hash",
                    "file.png", "c12", "2025-06-01T00:00:00+08:00",
                    "simulated_from_grade", "c12", "v1", "alg", "epr",
                    "question", 1, 1, "0", 0,
                ),
            )
            connection.execute(
                """INSERT INTO analysis_runs(
                     run_id,attempt_id,provider,model,prompt_version,status,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                ("run-1", "retry-1", "provider", "turbo", "prompt", "completed", "now"),
            )
            connection.execute(
                """INSERT INTO routing_decisions(
                     routing_decision_id,attempt_id,policy_name,policy_version,
                     initial_model,fallback_model,initial_run_id,selected_run_id,
                     selected_model,escalated,reasons_json,features_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "decision-1", "retry-1", "router", "v1", "turbo", "pro",
                    "run-1", "run-1", "turbo", 0, "[]", "{}", "now",
                ),
            )
            connection.commit()
            connection.close()
            policy = RoutingPolicy(
                policy_name="router", policy_version="v1",
                initial_profile="turbo-profile", fallback_profile="pro-profile",
                initial_model="turbo", fallback_model="pro",
                confidence_threshold=0.9,
            )
            self.assertEqual(
                select_unrouted_attempt_ids(db_path, policy=policy, limit=10),
                ["retry-1"],
            )

    def test_reviewed_initial_result_escalates_and_policy_metrics_use_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            db_path = Path(temporary) / "routing.db"
            connection = connect(db_path)
            initialize(connection)
            for index, truth in enumerate((0, 1), start=1):
                attempt_id = f"route-{index}"
                connection.execute(
                    "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        attempt_id, "rev", "shard", index, attempt_id, index,
                        f"hash-{index}", f"file-{index}.png", "c12",
                        f"2025-06-0{index}T00:00:00+08:00",
                        "simulated_from_grade", "c12", "v1", "alg", "epr",
                        "question", 1, 1, "0", 0,
                    ),
                )
                connection.execute(
                    "INSERT INTO benchmark_truth VALUES(?,?,?,?,?,?)",
                    (attempt_id, truth, "answer", "answer", "reason", "no_actual_error"),
                )
                connection.execute(
                    "INSERT INTO memory_nodes(node_id,node_type,label) VALUES(?,?,?)",
                    (f"attempt:{attempt_id}", "Attempt", attempt_id),
                )
            suite_id = "suite-routing"
            connection.execute(
                "INSERT INTO evaluation_suites VALUES(?,?,?,?,?,?)",
                (
                    suite_id, "routing-suite", "routing test", "seed",
                    json.dumps(["route-1", "route-2"]), "2025-01-01",
                ),
            )

            def insert_diagnosis(run_id, attempt_id, model, prediction, review, confidence):
                connection.execute(
                    """INSERT INTO analysis_runs(
                         run_id,attempt_id,provider,model,prompt_version,status,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (run_id, attempt_id, "provider", model, "prompt", "completed", run_id),
                )
                connection.execute(
                    """INSERT INTO diagnoses(
                         diagnosis_id,run_id,attempt_id,question_transcription,
                         solution_transcription,has_error_pred,error_type_pred,
                         error_explanation,knowledge_points_json,confidence,requires_review)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        f"diagnosis:{run_id}", run_id, attempt_id, "q", "s",
                        prediction, "conceptual" if prediction else "no_actual_error",
                        "explanation", "[]", confidence, int(review),
                    ),
                )

            insert_diagnosis("turbo-1", "route-1", "turbo", 1, True, 0.95)
            insert_diagnosis("pro-1", "route-1", "pro", 0, False, 0.95)
            insert_diagnosis("turbo-2", "route-2", "turbo", 1, False, 0.97)
            connection.execute(
                """INSERT INTO corrections(
                     correction_id,run_id,attempt_id,corrected_solution,
                     corrected_steps_json,final_answer,verification_expression,
                     verification_status,verification_method,
                     verification_details_json,confidence,rendered_path,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "correction:pro-1", "pro-1", "route-1", "corrected",
                    "[]", "42", "40+2=42", "verified", "numeric", "{}",
                    0.95, None, "2025-01-01",
                ),
            )
            connection.commit()
            connection.close()

            policy = RoutingPolicy(
                policy_name="router", policy_version="v1",
                initial_profile="turbo-profile", fallback_profile="pro-profile",
                initial_model="turbo", fallback_model="pro",
                confidence_threshold=0.9,
            )
            first = decide_route(db_path, attempt_id="route-1", policy=policy)
            second = decide_route(db_path, attempt_id="route-2", policy=policy)
            self.assertTrue(first["escalate"])
            self.assertEqual(first["selected_model"], "pro")
            self.assertFalse(second["escalate"])
            self.assertEqual(second["selected_model"], "turbo")

            metrics = evaluate_routing_policy(
                db_path, suite="routing-suite", policy=policy
            )
            self.assertEqual(metrics["coverage"], 1.0)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["escalation_rate"], 0.5)

            recorded = record_routing_decision(
                db_path,
                attempt_id="route-1",
                policy=policy,
                decision=first,
                selected_run_id="pro-1",
                fallback_run_id="pro-1",
            )
            self.assertEqual(recorded["selected_model"], "pro")
            materialized = materialize_routing_view(
                db_path,
                routing_decision_id=recorded["routing_decision_id"],
                policy=policy,
            )
            self.assertEqual(materialized["model_alias"], "seed-education-router-v1")
            replayed = record_routing_decision(
                db_path,
                attempt_id="route-1",
                policy=policy,
                decision=first,
                selected_run_id="pro-1",
                fallback_run_id="pro-1",
            )
            materialize_routing_view(
                db_path,
                routing_decision_id=replayed["routing_decision_id"],
                policy=policy,
            )
            routed = query_vlm_memories(
                db_path, model="seed-education-router-v1", limit=10
            )
            self.assertEqual(routed["total"], 1)
            self.assertEqual(routed["attempts"][0]["attempt_id"], "route-1")
            connection = connect(db_path)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM routing_decisions").fetchone()[0],
                2,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) FROM episodic_memories
                       WHERE attempt_id='route-1' AND source='routing:router:v1'"""
                ).fetchone()[0],
                1,
            )
            connection.close()


if __name__ == "__main__":
    unittest.main()
