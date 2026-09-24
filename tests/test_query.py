from pathlib import Path
import tempfile
import unittest

from m3_edu_memory.database import connect, initialize
from m3_edu_memory.query import MemoryQuery, query_attempts
from m3_edu_memory.weakness import extract_weaknesses


class QueryTests(unittest.TestCase):
    def test_exact_domain_and_error_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "memory.db"
            connection = connect(db)
            initialize(connection)
            base = (
                "fermat-test-000001", "rev", "shard.parquet", 1, "custom", 1,
                "hash", "parquet:///tmp/a.parquet#row=1", "c12",
                "2025-06-01T00:00:00+08:00", "simulated_from_grade", "c12", "v1",
                "clc", "derivative", "question", 1, 1, "0", 0,
            )
            connection.execute(
                """INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                base,
            )
            connection.execute(
                "INSERT INTO benchmark_truth VALUES(?,?,?,?,?,?)",
                (base[0], 1, "correct", "wrong", "reason", "conceptual"),
            )
            connection.commit()
            connection.close()

            result = query_attempts(db, MemoryQuery(domain_code="clc", errors_only=True))
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["attempts"][0]["attempt_id"], base[0])
            self.assertEqual(result["source"], "benchmark_reference")

            weakness = extract_weaknesses(db, domain_code="clc")["weaknesses"][0]
            self.assertEqual(weakness["subdomain_code"], "derivative")
            self.assertEqual(weakness["attempt_count"], 1)
            self.assertEqual(weakness["error_count"], 1)
            self.assertGreater(weakness["weakness_score"], 0)


if __name__ == "__main__":
    unittest.main()
