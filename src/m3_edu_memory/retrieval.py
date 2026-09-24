from __future__ import annotations

import hashlib
import math
import re
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .database import connect, initialize
from .query import query_vlm_memories
from .knowledge import canonicalize_knowledge_points


DOMAIN_LABELS = {
    "alg": "代数 algebra",
    "art": "算术 arithmetic",
    "apt": "综合能力 aptitude",
    "clc": "微积分 calculus",
    "mgm": "几何 geometry",
    "pst": "概率 统计 probability statistics",
    "trg": "三角 trigonometry",
}


class EmbeddingClient(Protocol):
    model: str
    dimension: int

    def embed(self, text: str) -> list[float]: ...


def _embed_document(client: EmbeddingClient, text: str) -> list[float]:
    method = getattr(client, "embed_document", None)
    return method(text) if method else client.embed(text)


def _embed_query(client: EmbeddingClient, text: str) -> list[float]:
    method = getattr(client, "embed_query", None)
    return method(text) if method else client.embed(text)


def _features(text: str) -> list[str]:
    normalized = text.casefold()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized)
    compact = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]", normalized))
    ngrams = [compact[index:index + 3] for index in range(max(0, len(compact) - 2))]
    return words + ngrams


@dataclass(frozen=True)
class HashEmbeddingClient:
    """Offline deterministic baseline; swap with a semantic embedding provider."""

    dimension: int = 384
    model: str = "local-hash-embedding-v1"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for feature in _features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            vector[index] += -1.0 if value & (1 << 63) else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dimension: int) -> tuple[float, ...]:
    return struct.unpack(f"<{dimension}f", blob)


def _cosine(left: list[float], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _initialize_fts(connection) -> bool:
    try:
        connection.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                 document_id UNINDEXED, model UNINDEXED, content, tokenize='unicode61')"""
        )
        return True
    except Exception:
        return False


def _document_content(item: dict) -> str:
    correction = item.get("correction") or {}
    review = item.get("review") or {}
    canonical_points = canonicalize_knowledge_points(
        list(item["knowledge_points"]),
        domain_code=item.get("domain_code"),
        subdomain_code=item.get("subdomain_code"),
    )
    return "\n".join(
        part for part in (
            DOMAIN_LABELS.get(item["domain_code"], item["domain_code"]),
            item.get("subdomain_code") or "",
            item["question_transcription"],
            item["solution_transcription"],
            " ".join(dict.fromkeys([*item["knowledge_points"], *canonical_points])),
            item.get("error_type_effective") or item["error_type_pred"],
            item["error_explanation"],
            correction.get("corrected_solution", ""),
            correction.get("final_answer", ""),
            review.get("notes", ""),
        ) if part
    )


def build_memory_index(
    db_path: str | Path,
    *,
    model: str,
    embedding_client: EmbeddingClient | None = None,
    max_workers: int = 1,
) -> dict:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    client = embedding_client or HashEmbeddingClient()
    memories = query_vlm_memories(db_path, model=model, limit=100_000)
    connection = connect(db_path)
    initialize(connection)
    fts_enabled = _initialize_fts(connection)
    now = datetime.now(timezone.utc).isoformat()
    documents = [_document_payload(item, model=model) for item in memories["attempts"]]
    existing_rows = connection.execute(
        "SELECT document_id,content_hash FROM memory_documents WHERE model=?",
        (model,),
    ).fetchall()
    existing_hashes = {row["document_id"]: row["content_hash"] for row in existing_rows}
    embedded_rows = connection.execute(
        """SELECT e.document_id FROM memory_embeddings e
           JOIN memory_documents d USING(document_id)
           WHERE d.model=? AND e.embedding_model=? AND e.dimension=?""",
        (model, client.model, client.dimension),
    ).fetchall()
    embedded_ids = {row["document_id"] for row in embedded_rows}
    pending = [
        document for document in documents
        if not (
            existing_hashes.get(document["document_id"]) == document["content_hash"]
            and document["document_id"] in embedded_ids
        )
    ]
    failures = []

    def embed(document: dict) -> tuple[dict, list[float]]:
        return document, _embed_document(client, document["content"])

    embedded_count = 0
    if max_workers == 1:
        for document in pending:
            try:
                completed_document, vector = embed(document)
                with connection:
                    _store_document(
                        connection,
                        document=completed_document,
                        vector=vector,
                        client=client,
                        fts_enabled=fts_enabled,
                        now=now,
                    )
                embedded_count += 1
            except Exception as exc:
                failures.append({
                    "attempt_id": document["item"]["attempt_id"],
                    "error": str(exc),
                })
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(embed, document): document for document in pending}
            for future in as_completed(futures):
                document = futures[future]
                try:
                    completed_document, vector = future.result()
                    with connection:
                        _store_document(
                            connection,
                            document=completed_document,
                            vector=vector,
                            client=client,
                            fts_enabled=fts_enabled,
                            now=now,
                        )
                    embedded_count += 1
                except Exception as exc:
                    failures.append({
                        "attempt_id": document["item"]["attempt_id"],
                        "error": str(exc),
                    })
    desired_ids = {document["document_id"] for document in documents}
    if not failures:
        stale_ids = set(existing_hashes) - desired_ids
        with connection:
            for document_id in stale_ids:
                if fts_enabled:
                    connection.execute(
                        "DELETE FROM memory_fts WHERE document_id=?", (document_id,)
                    )
                connection.execute(
                    "DELETE FROM memory_documents WHERE document_id=?", (document_id,)
                )
    count = len(documents)
    connection.close()
    if failures:
        preview = "; ".join(
            f"{item['attempt_id']}: {item['error']}" for item in failures[:5]
        )
        raise RuntimeError(
            f"Embedding failed for {len(failures)} of {len(pending)} documents; "
            f"successful vectors were retained for resume. {preview}"
        )
    return {
        "model": model,
        "indexed_documents": count,
        "embedded_documents": embedded_count,
        "reused_embeddings": count - len(pending),
        "embedding_model": client.model,
        "dimension": client.dimension,
        "fts_enabled": fts_enabled,
        "max_workers": max_workers,
    }


def _document_payload(item: dict, *, model: str) -> dict:
    content = _document_content(item)
    return {
        "item": item,
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "document_id": f"memory-document:{model}:{item['attempt_id']}",
        "model": model,
    }


def _store_document(
    connection,
    *,
    document: dict,
    vector: list[float],
    client,
    fts_enabled: bool,
    now: str,
) -> str:
    item = document["item"]
    document_id = document["document_id"]
    previous = connection.execute(
        "SELECT content_hash FROM memory_documents WHERE document_id=?",
        (document_id,),
    ).fetchone()
    if previous is not None and previous["content_hash"] != document["content_hash"]:
        connection.execute(
            "DELETE FROM memory_embeddings WHERE document_id=?", (document_id,)
        )
    if fts_enabled:
        connection.execute("DELETE FROM memory_fts WHERE document_id=?", (document_id,))
    connection.execute(
        """INSERT INTO memory_documents(
             document_id,attempt_id,run_id,model,prompt_version,domain_code,
             subdomain_code,content,content_hash,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(document_id) DO UPDATE SET
             attempt_id=excluded.attempt_id,
             run_id=excluded.run_id,
             model=excluded.model,
             prompt_version=excluded.prompt_version,
             domain_code=excluded.domain_code,
             subdomain_code=excluded.subdomain_code,
             content=excluded.content,
             content_hash=excluded.content_hash,
             updated_at=excluded.updated_at""",
        (
            document_id, item["attempt_id"], item["run_id"], document["model"],
            item["prompt_version"], item["domain_code"], item["subdomain_code"],
            document["content"], document["content_hash"], now,
        ),
    )
    connection.execute(
        """INSERT INTO memory_embeddings(
             document_id,embedding_model,dimension,vector_blob,created_at)
           VALUES(?,?,?,?,?)
           ON CONFLICT(document_id,embedding_model) DO UPDATE SET
             dimension=excluded.dimension,
             vector_blob=excluded.vector_blob,
             created_at=excluded.created_at""",
        (document_id, client.model, len(vector), _pack(vector), now),
    )
    if fts_enabled:
        connection.execute(
            "INSERT INTO memory_fts(document_id,model,content) VALUES(?,?,?)",
            (document_id, document["model"], document["content"]),
        )
    return document_id


def _write_document(connection, *, item: dict, model: str, client, fts_enabled: bool, now: str) -> str:
    document = _document_payload(item, model=model)
    vector = _embed_document(client, document["content"])
    return _store_document(
        connection,
        document=document,
        vector=vector,
        client=client,
        fts_enabled=fts_enabled,
        now=now,
    )


def index_memory_attempt(
    db_path: str | Path,
    *,
    model: str,
    attempt_id: str,
    embedding_client: EmbeddingClient | None = None,
) -> dict:
    """Incrementally refresh one latest memory after diagnosis or review."""
    client = embedding_client or HashEmbeddingClient()
    result = query_vlm_memories(
        db_path, model=model, attempt_id=attempt_id, limit=1
    )
    if not result["attempts"]:
        raise KeyError(f"No completed {model} diagnosis for {attempt_id}")
    connection = connect(db_path)
    initialize(connection)
    fts_enabled = _initialize_fts(connection)
    now = datetime.now(timezone.utc).isoformat()
    with connection:
        document_id = _write_document(
            connection,
            item=result["attempts"][0],
            model=model,
            client=client,
            fts_enabled=fts_enabled,
            now=now,
        )
    connection.close()
    return {
        "document_id": document_id,
        "attempt_id": attempt_id,
        "model": model,
        "embedding_model": client.model,
        "fts_enabled": fts_enabled,
    }


def memory_index_stats(db_path: str | Path, *, model: str) -> dict:
    connection = connect(db_path)
    initialize(connection)
    row = connection.execute(
        """SELECT COUNT(*) AS documents,MAX(updated_at) AS updated_at
           FROM memory_documents WHERE model=?""",
        (model,),
    ).fetchone()
    embeddings = connection.execute(
        """SELECT embedding_model,COUNT(*) AS count
           FROM memory_embeddings e JOIN memory_documents d USING(document_id)
           WHERE d.model=? GROUP BY embedding_model""",
        (model,),
    ).fetchall()
    connection.close()
    return {
        "model": model,
        "documents": row["documents"],
        "updated_at": row["updated_at"],
        "embeddings": [dict(item) for item in embeddings],
    }


def hybrid_search(
    db_path: str | Path,
    *,
    model: str,
    query: str,
    candidate_attempt_ids: list[str] | None = None,
    limit: int = 20,
    embedding_client: EmbeddingClient | None = None,
    vector_weight: float = 0.65,
) -> dict:
    if not 0.0 <= vector_weight <= 1.0:
        raise ValueError("vector_weight must be between 0 and 1")
    client = embedding_client or HashEmbeddingClient()
    connection = connect(db_path)
    initialize(connection)
    candidates = set(candidate_attempt_ids or [])
    rows = connection.execute(
        """SELECT d.document_id,d.attempt_id,d.content,e.dimension,e.vector_blob
           FROM memory_documents d
           JOIN memory_embeddings e USING(document_id)
           WHERE d.model=? AND e.embedding_model=?""",
        (model, client.model),
    ).fetchall()
    if candidates:
        rows = [row for row in rows if row["attempt_id"] in candidates]
    if not rows:
        connection.close()
        return {"query": query, "backend": "unavailable", "results": []}

    query_vector = _embed_query(client, query)
    vector_scores = {
        row["document_id"]: max(0.0, _cosine(query_vector, _unpack(row["vector_blob"], row["dimension"])))
        for row in rows
    }
    fts_scores: dict[str, float] = {}
    if _initialize_fts(connection):
        terms = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", query.casefold())
        if terms:
            expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
            try:
                fts_rows = connection.execute(
                    """SELECT document_id,bm25(memory_fts) AS rank
                       FROM memory_fts WHERE memory_fts MATCH ? AND model=?
                       ORDER BY rank LIMIT 500""",
                    (expression, model),
                ).fetchall()
                permitted = {row["document_id"] for row in rows}
                ordered = [row for row in fts_rows if row["document_id"] in permitted]
                fts_scores = {
                    row["document_id"]: 1.0 / (1.0 + index)
                    for index, row in enumerate(ordered)
                }
            except Exception:
                fts_scores = {}
    ranked = []
    for row in rows:
        vector_score = vector_scores[row["document_id"]]
        fulltext_score = fts_scores.get(row["document_id"], 0.0)
        score = vector_weight * vector_score + (1.0 - vector_weight) * fulltext_score
        ranked.append(
            {
                "attempt_id": row["attempt_id"],
                "score": round(score, 6),
                "vector_score": round(vector_score, 6),
                "fulltext_score": round(fulltext_score, 6),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["attempt_id"]))
    connection.close()
    return {
        "query": query,
        "backend": (
            "sqlite_fts5+local_hash_vector"
            if client.model == "local-hash-embedding-v1"
            else "sqlite_fts5+dense_vector"
        ),
        "embedding_model": client.model,
        "vector_weight": vector_weight,
        "results": ranked[: max(1, min(limit, 500))],
    }
