from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m3_edu_memory.database import connect, initialize
from m3_edu_memory.operations import is_fatal_provider_error, routing_batch_status
from scripts.finalize_seed_main_memory import build_acceptance


class OperationsTests(unittest.TestCase):
    def test_full_demo_acceptance_requires_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = {}
            for name in ("calculus", "algebra", "geometry", "trigonometry", "probability"):
                gallery = root / name / "gallery.html"
                gallery.parent.mkdir(parents=True)
                gallery.write_text("demo", encoding="utf-8")
                reports[name] = {
                    "matched": 4,
                    "returned": 4,
                    "gallery_html": str(gallery),
                }
            acceptance = build_acceptance(
                coverage={
                    "total_attempts": 2244,
                    "materialized_attempts": 2244,
                    "remaining_attempts": 0,
                },
                index={
                    "documents": 2244,
                    "embeddings": [
                        {"embedding_model": "vision", "count": 2244}
                    ],
                },
                embedding_model="vision",
                retrieval={"query_count": 14, "evaluated_query_count": 14},
                reports=reports,
                expected_query_count=14,
            )
            self.assertTrue(acceptance["passed"])
            self.assertEqual(acceptance["status"], "passed")

            reports["geometry"]["returned"] = 0
            failed = build_acceptance(
                coverage={
                    "total_attempts": 2244,
                    "materialized_attempts": 2243,
                    "remaining_attempts": 1,
                },
                index={"documents": 2243, "embeddings": []},
                embedding_model="vision",
                retrieval={"query_count": 13, "evaluated_query_count": 13},
                reports=reports,
                expected_query_count=14,
            )
            self.assertFalse(failed["passed"])
            self.assertFalse(failed["checks"]["full_memory_coverage"]["passed"])
            self.assertFalse(failed["checks"]["subject_reports"]["passed"])

    def test_provider_wide_billing_and_auth_errors_trip_circuit_breaker(self):
        self.assertTrue(is_fatal_provider_error("VLM HTTP 403: AccountOverdueError"))
        self.assertTrue(is_fatal_provider_error("VLM HTTP 401: invalid api key"))
        self.assertFalse(is_fatal_provider_error("VLM request timed out after 240 seconds"))
        self.assertFalse(is_fatal_provider_error("response did not contain JSON"))

    def test_empty_batch_status_is_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = root / "memory.db"
            connection = connect(db)
            initialize(connection)
            connection.close()
            config = root / "routing.json"
            config.write_text(
                json.dumps(
                    {
                        "policy_name": "router",
                        "policy_version": "v1",
                        "model_alias": "routed-model",
                    }
                ),
                encoding="utf-8",
            )
            result = routing_batch_status(db, routing_config=config)
            self.assertEqual(result["total_attempts"], 0)
            self.assertEqual(result["materialized_attempts"], 0)
            self.assertEqual(result["coverage"], 0.0)
            self.assertEqual(result["retryable_unmaterialized_decisions"], 0)


if __name__ == "__main__":
    unittest.main()
