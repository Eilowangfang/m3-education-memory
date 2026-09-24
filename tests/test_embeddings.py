from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m3_edu_memory.embeddings import create_embedding_client, load_embedding_profiles
from m3_edu_memory.retrieval import _embed_document, _embed_query
from m3_edu_memory.retrieval_evaluation import _metrics


class EmbeddingTests(unittest.TestCase):
    def test_config_requires_environment_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "bad": {
                                "endpoint": "https://example.test",
                                "model": "model",
                                "api_key_env": "KEY",
                                "api_key": "must-not-be-here",
                                "dimension": 2,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "api_key"):
                load_embedding_profiles(path)

    def test_client_loads_profile_without_reading_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "ark": {
                                "endpoint": "https://example.test",
                                "model": "embedding-model",
                                "api_key_env": "ARK_TEST_KEY",
                                "dimension": 1024,
                                "document_instructions": "document",
                                "query_instructions": "query",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            client = create_embedding_client(path, profile_name="ark")
            self.assertEqual(client.model, "embedding-model")
            self.assertEqual(client.dimension, 1024)
            self.assertEqual(client.api_key_env, "ARK_TEST_KEY")

    def test_retrieval_uses_distinct_document_and_query_roles(self):
        class FakeClient:
            model = "fake"
            dimension = 2

            def __init__(self):
                self.calls = []

            def embed_document(self, text):
                self.calls.append(("document", text))
                return [1.0, 0.0]

            def embed_query(self, text):
                self.calls.append(("query", text))
                return [0.0, 1.0]

        client = FakeClient()
        self.assertEqual(_embed_document(client, "memory"), [1.0, 0.0])
        self.assertEqual(_embed_query(client, "question"), [0.0, 1.0])
        self.assertEqual(
            client.calls,
            [("document", "memory"), ("query", "question")],
        )

    def test_retrieval_metrics_are_computed_at_k(self):
        metrics = _metrics(["a", "x", "b"], {"a", "b", "c", "d"}, k=3)
        self.assertAlmostEqual(metrics["precision@3"], 2 / 3)
        self.assertAlmostEqual(metrics["recall@3"], 0.5)
        self.assertEqual(metrics["hit@3"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 1.0)
        self.assertGreater(metrics["ndcg@3"], 0.0)


if __name__ == "__main__":
    unittest.main()
