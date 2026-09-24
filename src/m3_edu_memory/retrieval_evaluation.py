from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .database import connect, initialize
from .retrieval import EmbeddingClient, hybrid_search


def load_retrieval_queries(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(queries, list) or not queries:
        raise ValueError("Retrieval evaluation config requires a non-empty queries list")
    normalized = []
    for index, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Retrieval query {index} must be an object")
        query = str(item.get("query", "")).strip()
        domain = str(item.get("relevant_domain", "")).strip()
        if not query or not domain:
            raise ValueError(
                f"Retrieval query {index} requires query and relevant_domain"
            )
        normalized.append(
            {
                "query_id": str(item.get("query_id") or f"query-{index:02d}"),
                "query": query,
                "relevant_domain": domain,
            }
        )
    return normalized


def _relevant_attempt_ids(db_path: str | Path, *, model: str, domain: str) -> set[str]:
    connection = connect(db_path)
    initialize(connection)
    rows = connection.execute(
        "SELECT attempt_id FROM memory_documents WHERE model=? AND domain_code=?",
        (model, domain),
    ).fetchall()
    connection.close()
    return {row["attempt_id"] for row in rows}


def _metrics(ranked_ids: list[str], relevant: set[str], *, k: int) -> dict:
    top = ranked_ids[:k]
    hits = [1 if attempt_id in relevant else 0 for attempt_id in top]
    hit_count = sum(hits)
    first_rank = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        "relevant_count": len(relevant),
        "retrieved_count": len(top),
        "relevant_retrieved": hit_count,
        f"precision@{k}": hit_count / len(top) if top else 0.0,
        f"recall@{k}": hit_count / len(relevant) if relevant else 0.0,
        f"hit@{k}": 1.0 if hit_count else 0.0,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def evaluate_retrieval(
    db_path: str | Path,
    *,
    model: str,
    queries_path: str | Path,
    k: int = 10,
    embedding_client: EmbeddingClient | None = None,
    vector_weight: float = 0.65,
    max_workers: int = 1,
) -> dict:
    k = max(1, min(int(k), 100))
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    def evaluate_one(item: dict) -> dict:
        relevant = _relevant_attempt_ids(
            db_path, model=model, domain=item["relevant_domain"]
        )
        result = hybrid_search(
            db_path,
            model=model,
            query=item["query"],
            limit=k,
            embedding_client=embedding_client,
            vector_weight=vector_weight,
        )
        ranked_ids = [row["attempt_id"] for row in result["results"]]
        metrics = _metrics(ranked_ids, relevant, k=k)
        return {
            **item,
            **metrics,
            "ranked_attempt_ids": ranked_ids,
        }

    queries = load_retrieval_queries(queries_path)
    if max_workers == 1:
        query_results = [evaluate_one(item) for item in queries]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            query_results = list(executor.map(evaluate_one, queries))
    evaluated = [item for item in query_results if item["relevant_count"]]
    metric_names = (
        f"precision@{k}", f"recall@{k}", f"hit@{k}",
        "reciprocal_rank", f"ndcg@{k}",
    )
    macro = {
        name: (
            sum(item[name] for item in evaluated) / len(evaluated)
            if evaluated else 0.0
        )
        for name in metric_names
    }
    return {
        "model": model,
        "embedding_model": (
            embedding_client.model if embedding_client is not None
            else "local-hash-embedding-v1"
        ),
        "relevance_source": "memory_document_domain_code",
        "k": k,
        "vector_weight": vector_weight,
        "max_workers": max_workers,
        "query_count": len(query_results),
        "evaluated_query_count": len(evaluated),
        "macro_metrics": macro,
        "queries": query_results,
    }
