from __future__ import annotations

from pathlib import Path

from .database import connect, initialize


def database_stats(db_path: str | Path) -> dict:
    connection = connect(db_path)
    initialize(connection)
    scalar_queries = {
        "attempts": "SELECT COUNT(*) FROM attempts",
        "errors": "SELECT COUNT(*) FROM benchmark_truth WHERE has_error=1",
        "calculus_attempts": "SELECT COUNT(*) FROM attempts WHERE domain_code='clc'",
        "calculus_errors": (
            "SELECT COUNT(*) FROM attempts a JOIN benchmark_truth b USING(attempt_id) "
            "WHERE a.domain_code='clc' AND b.has_error=1"
        ),
        "episodic_memories": "SELECT COUNT(*) FROM episodic_memories",
        "semantic_memories": "SELECT COUNT(*) FROM semantic_memories",
        "graph_nodes": "SELECT COUNT(*) FROM memory_nodes",
        "graph_edges": "SELECT COUNT(*) FROM memory_edges",
    }
    result = {
        name: connection.execute(statement).fetchone()[0]
        for name, statement in scalar_queries.items()
    }
    result["grade_time_buckets"] = [
        dict(row)
        for row in connection.execute(
            """SELECT grade,substr(Time,6,2) AS month,COUNT(*) AS count
               FROM attempts GROUP BY grade,month ORDER BY month"""
        ).fetchall()
    ]
    result["memory_sources"] = [
        dict(row)
        for row in connection.execute(
            "SELECT source,COUNT(*) AS count FROM episodic_memories GROUP BY source"
        ).fetchall()
    ]
    connection.close()
    return result
