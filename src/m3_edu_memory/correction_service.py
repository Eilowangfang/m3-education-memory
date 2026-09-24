from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import connect, initialize
from .memory_writer import rebuild_mastery_states
from .retrieval import index_memory_attempt
from .verification import verify_correction


CORRECTION_SOURCES = {"human", "rule"}


def submit_correction_version(
    db_path: str | Path,
    *,
    attempt_id: str,
    model: str,
    corrected_solution: str,
    corrected_steps: list[dict] | None = None,
    final_answer: str | None = None,
    verification_expression: str | None = None,
    confidence: float = 1.0,
    source: str = "human",
    created_by: str = "local-reviewer",
    activate: bool = True,
    parent_version_id: str | None = None,
) -> dict:
    if source not in CORRECTION_SOURCES:
        raise ValueError("source must be human or rule")
    if not corrected_solution.strip():
        raise ValueError("corrected_solution is required")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be between 0 and 1")
    steps = corrected_steps or []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not all(
            key in step for key in ("step_index", "latex", "explanation")
        ):
            raise ValueError(
                f"corrected_steps[{index}] requires step_index, latex and explanation"
            )

    connection = connect(db_path)
    initialize(connection)
    run = connection.execute(
        """SELECT run_id,prompt_version FROM analysis_runs
           WHERE attempt_id=? AND model=? AND status='completed'
           ORDER BY created_at DESC,run_id DESC LIMIT 1""",
        (attempt_id, model),
    ).fetchone()
    if run is None:
        connection.close()
        raise KeyError(f"No completed {model} diagnosis for {attempt_id}")
    if parent_version_id is None:
        parent = connection.execute(
            """SELECT version_id FROM correction_versions
               WHERE attempt_id=? AND model=? AND status='active'
               ORDER BY created_at DESC,version_id DESC LIMIT 1""",
            (attempt_id, model),
        ).fetchone()
        parent_version_id = parent["version_id"] if parent else None
    elif connection.execute(
        """SELECT 1 FROM correction_versions
           WHERE version_id=? AND attempt_id=? AND model=?""",
        (parent_version_id, attempt_id, model),
    ).fetchone() is None:
        connection.close()
        raise ValueError("parent_version_id does not belong to this attempt and model")

    correction = {
        "corrected_solution": corrected_solution.strip(),
        "corrected_steps": steps,
        "final_answer": final_answer,
        "verification_expression": verification_expression,
        "confidence": float(confidence),
    }
    verification = verify_correction(correction)
    version_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    status = "active" if activate else "draft"
    with connection:
        if activate:
            connection.execute(
                """UPDATE correction_versions SET status='superseded'
                   WHERE attempt_id=? AND model=? AND status='active'""",
                (attempt_id, model),
            )
        connection.execute(
            """INSERT INTO correction_versions(
                 version_id,attempt_id,run_id,model,source,parent_version_id,
                 corrected_solution,corrected_steps_json,final_answer,
                 verification_expression,verification_status,verification_method,
                 verification_details_json,confidence,status,created_by,
                 rendered_path,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",
            (
                version_id, attempt_id, run["run_id"], model, source,
                parent_version_id, correction["corrected_solution"],
                json.dumps(steps, ensure_ascii=False), final_answer,
                verification_expression, verification["status"],
                verification["method"],
                json.dumps(verification["details"], ensure_ascii=False),
                float(confidence), status, created_by, created_at,
            ),
        )
        node_id = f"correction-version-node:{version_id}"
        connection.execute(
            """INSERT INTO memory_nodes(node_id,node_type,label,properties_json)
               VALUES(?,?,?,?)""",
            (
                node_id, "CorrectionVersion", final_answer or "corrected solution",
                json.dumps(
                    {
                        "source": source,
                        "status": status,
                        "verification_status": verification["status"],
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        connection.execute(
            """INSERT OR IGNORE INTO memory_edges(source_id,relation,target_id)
               VALUES(?,?,?)""",
            (f"attempt:{attempt_id}", "HAS_CORRECTION_VERSION", node_id),
        )
        if parent_version_id:
            parent_node = f"correction-version-node:{parent_version_id}"
            if connection.execute(
                "SELECT 1 FROM memory_nodes WHERE node_id=?", (parent_node,)
            ).fetchone():
                connection.execute(
                    """INSERT OR IGNORE INTO memory_edges(source_id,relation,target_id)
                       VALUES(?,?,?)""",
                    (node_id, "REVISES", parent_node),
                )
        if activate:
            rebuild_mastery_states(
                connection,
                model=model,
                prompt_version=run["prompt_version"],
                change_reason=f"correction:{version_id}",
            )
    connection.close()
    result = {
        "version_id": version_id,
        "attempt_id": attempt_id,
        "run_id": run["run_id"],
        "model": model,
        "source": source,
        "parent_version_id": parent_version_id,
        "status": status,
        **correction,
        "verification": verification,
        "created_by": created_by,
        "created_at": created_at,
    }
    if activate:
        try:
            result["index_update"] = index_memory_attempt(
                db_path, model=model, attempt_id=attempt_id
            )
        except Exception as exc:
            result["index_error"] = str(exc)
    return result


def list_correction_versions(
    db_path: str | Path, *, attempt_id: str, model: str
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    rows = connection.execute(
        """SELECT * FROM correction_versions
           WHERE attempt_id=? AND model=?
           ORDER BY created_at DESC,version_id DESC""",
        (attempt_id, model),
    ).fetchall()
    connection.close()
    versions = []
    for row in rows:
        item = dict(row)
        item["corrected_steps"] = json.loads(item.pop("corrected_steps_json"))
        item["verification_details"] = json.loads(
            item.pop("verification_details_json")
        )
        versions.append(item)
    return {"attempt_id": attempt_id, "model": model, "versions": versions}
