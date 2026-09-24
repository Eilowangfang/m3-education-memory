from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m3_edu_memory.database import connect, initialize
from m3_edu_memory.operations import is_fatal_provider_error, routing_batch_status


class OperationsTests(unittest.TestCase):
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
