from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import connect, initialize
from .memory_writer import rebuild_mastery_states
from .taxonomy import ERROR_TYPES


REVIEW_VERDICTS = {"confirmed", "rejected", "modified"}


def submit_diagnosis_review(
    db_path: str | Path,
    *,
    attempt_id: str,
    model: str,
    verdict: str,
    has_error: bool | None = None,
    error_type: str | None = None,
    first_error_step: int | None = None,
    error_bbox: list[float] | None = None,
    knowledge_points: list[str] | None = None,
    error_explanation: str | None = None,
    reviewer: str = "local-teacher",
    notes: str = "",
    run_id: str | None = None,
) -> dict:
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(f"Unsupported verdict: {verdict}")
    if error_type is not None and error_type not in ERROR_TYPES:
        raise ValueError(f"Unsupported error_type: {error_type}")
    if has_error is not None and not isinstance(has_error, bool):
        raise ValueError("has_error must be true, false, or null")
    if first_error_step is not None:
        if isinstance(first_error_step, bool) or not isinstance(first_error_step, int):
            raise ValueError("first_error_step must be an integer")
        if first_error_step < 0:
            raise ValueError("first_error_step must be zero or greater")
    if error_bbox is not None:
        if not isinstance(error_bbox, list) or len(error_bbox) != 4:
            raise ValueError("error_bbox must contain four normalized coordinates")
        error_bbox = [float(item) for item in error_bbox]
        if any(item < 0 or item > 1 for item in error_bbox):
            raise ValueError("error_bbox values must be between 0 and 1")
        if error_bbox[0] >= error_bbox[2] or error_bbox[1] >= error_bbox[3]:
            raise ValueError("error_bbox must satisfy x1 < x2 and y1 < y2")
    if knowledge_points is not None and not isinstance(knowledge_points, list):
        raise ValueError("knowledge_points must be an array")
    points = (
        [str(item).strip() for item in knowledge_points if str(item).strip()]
        if knowledge_points is not None else None
    )
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    if verdict == "rejected":
        has_error = False if has_error is None else has_error
        error_type = error_type or "no_actual_error"
        first_error_step = None
        error_bbox = None
    if verdict == "modified" and all(
        value is None for value in (
            has_error, error_type, first_error_step, error_bbox,
            knowledge_points, error_explanation,
        )
    ):
        raise ValueError("modified review requires at least one diagnosis override")

    connection = connect(db_path)
    initialize(connection)
    if run_id:
        run = connection.execute(
            """SELECT run_id,prompt_version FROM analysis_runs
               WHERE run_id=? AND attempt_id=? AND model=? AND status='completed'""",
            (run_id, attempt_id, model),
        ).fetchone()
    else:
        run = connection.execute(
            """SELECT run_id,prompt_version FROM analysis_runs
               WHERE attempt_id=? AND model=? AND status='completed'
               ORDER BY created_at DESC,run_id DESC LIMIT 1""",
            (attempt_id, model),
        ).fetchone()
    if run is None:
        connection.close()
        raise KeyError(f"No completed {model} diagnosis for {attempt_id}")

    review_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with connection:
        connection.execute(
            """INSERT INTO diagnosis_reviews(
                 review_id,run_id,attempt_id,verdict,has_error_override,
                 error_type_override,first_error_step_override,
                 error_bbox_override_json,knowledge_points_override_json,
                 error_explanation_override,reviewer,notes,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                review_id, run["run_id"], attempt_id, verdict,
                None if has_error is None else int(has_error),
                error_type, first_error_step,
                None if error_bbox is None else json.dumps(error_bbox),
                None if points is None else json.dumps(points, ensure_ascii=False),
                None if error_explanation is None else str(error_explanation).strip(),
                reviewer.strip(), notes, created_at,
            ),
        )
        rebuild_mastery_states(
            connection,
            model=model,
            prompt_version=run["prompt_version"],
            change_reason=f"review:{review_id}",
        )
    connection.close()
    result = {
        "review_id": review_id,
        "run_id": run["run_id"],
        "attempt_id": attempt_id,
        "model": model,
        "verdict": verdict,
        "has_error_override": has_error,
        "error_type_override": error_type,
        "first_error_step_override": first_error_step,
        "error_bbox_override": error_bbox,
        "knowledge_points_override": points,
        "error_explanation_override": error_explanation,
        "reviewer": reviewer.strip(),
        "notes": notes,
        "created_at": created_at,
    }
    try:
        from .retrieval import index_memory_attempt

        result["index_update"] = index_memory_attempt(
            db_path, model=model, attempt_id=attempt_id
        )
    except Exception as index_exc:
        result["index_error"] = str(index_exc)
    return result
