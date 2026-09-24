from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import connect, initialize
from .media import image_bytes_from_source, image_mime_type


ANNOTATION_STATUSES = {"draft", "approved", "rejected"}


def get_attempt_image(db_path: str | Path, *, attempt_id: str) -> tuple[bytes, str]:
    cache_root = Path(db_path).parent / "evaluation-image-cache"
    cache_key = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()
    cache_path = cache_root / f"{cache_key}.image"
    if cache_path.exists():
        content = cache_path.read_bytes()
        return content, image_mime_type(content)
    connection = connect(db_path)
    initialize(connection)
    row = connection.execute(
        "SELECT image_source_path FROM attempts WHERE attempt_id=?", (attempt_id,)
    ).fetchone()
    connection.close()
    if row is None:
        raise KeyError(f"Unknown attempt_id: {attempt_id}")
    content = image_bytes_from_source(row["image_source_path"])
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_root / f".{cache_key}.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(content)
    temporary.replace(cache_path)
    return content, image_mime_type(content)


def _suite_row(connection, suite: str):
    row = connection.execute(
        "SELECT * FROM evaluation_suites WHERE suite_id=? OR name=?", (suite, suite)
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown evaluation suite: {suite}")
    return row


def _bbox(value) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("error_bbox must contain four normalized coordinates")
    result = [float(item) for item in value]
    if any(item < 0 or item > 1 for item in result):
        raise ValueError("error_bbox values must be between 0 and 1")
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("error_bbox must satisfy x1 < x2 and y1 < y2")
    return result


def submit_evaluation_annotation(
    db_path: str | Path,
    *,
    suite: str,
    attempt_id: str,
    annotator: str,
    status: str,
    has_error: bool | None,
    first_error_step: int | None = None,
    error_bbox: list[float] | None = None,
    knowledge_points: list[str] | None = None,
    corrected_final_answer: str | None = None,
    correction_steps: list[dict] | None = None,
    notes: str = "",
    parent_annotation_id: str | None = None,
) -> dict:
    if status not in ANNOTATION_STATUSES:
        raise ValueError(f"Unsupported annotation status: {status}")
    if not annotator.strip():
        raise ValueError("annotator is required")
    if has_error is not None and not isinstance(has_error, bool):
        raise ValueError("has_error must be true, false, or null")
    if first_error_step is not None:
        if isinstance(first_error_step, bool) or not isinstance(first_error_step, int):
            raise ValueError("first_error_step must be an integer")
        if first_error_step < 0:
            raise ValueError("first_error_step must be zero or greater")
    bbox = _bbox(error_bbox)
    if knowledge_points is not None and not isinstance(knowledge_points, list):
        raise ValueError("knowledge_points must be an array")
    points = [str(item).strip() for item in (knowledge_points or []) if str(item).strip()]
    steps = correction_steps or []
    if not isinstance(steps, list) or any(not isinstance(item, dict) for item in steps):
        raise ValueError("correction_steps must be an array of objects")
    if status == "approved" and has_error is None:
        raise ValueError("approved annotation requires has_error")
    if status == "approved" and has_error:
        if first_error_step is None:
            raise ValueError("approved error annotation requires first_error_step")
        if bbox is None:
            raise ValueError("approved error annotation requires error_bbox")
        if not points:
            raise ValueError("approved error annotation requires knowledge_points")
        if not str(corrected_final_answer or "").strip():
            raise ValueError("approved error annotation requires corrected_final_answer")
    if status == "approved" and has_error is False:
        first_error_step = None
        bbox = None

    connection = connect(db_path)
    initialize(connection)
    try:
        suite_row = _suite_row(connection, suite)
        attempt_ids = set(json.loads(suite_row["attempt_ids_json"]))
        if attempt_id not in attempt_ids:
            raise ValueError(f"Attempt {attempt_id} is not part of suite {suite_row['name']}")
        if parent_annotation_id:
            parent = connection.execute(
                """SELECT annotation_id FROM evaluation_annotations
                   WHERE annotation_id=? AND suite_id=? AND attempt_id=?""",
                (parent_annotation_id, suite_row["suite_id"], attempt_id),
            ).fetchone()
            if parent is None:
                raise ValueError("parent_annotation_id does not belong to this suite case")
        else:
            parent = connection.execute(
                """SELECT annotation_id FROM evaluation_annotations
                   WHERE suite_id=? AND attempt_id=?
                   ORDER BY created_at DESC,annotation_id DESC LIMIT 1""",
                (suite_row["suite_id"], attempt_id),
            ).fetchone()
            parent_annotation_id = parent["annotation_id"] if parent else None
        annotation_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with connection:
            connection.execute(
                """INSERT INTO evaluation_annotations(
                     annotation_id,suite_id,attempt_id,parent_annotation_id,
                     annotator,status,has_error,first_error_step,error_bbox_json,
                     knowledge_points_json,corrected_final_answer,
                     correction_steps_json,notes,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    annotation_id, suite_row["suite_id"], attempt_id,
                    parent_annotation_id, annotator.strip(), status,
                    None if has_error is None else int(bool(has_error)),
                    first_error_step, None if bbox is None else json.dumps(bbox),
                    json.dumps(points, ensure_ascii=False), corrected_final_answer,
                    json.dumps(steps, ensure_ascii=False), notes, created_at,
                ),
            )
    finally:
        connection.close()
    return {
        "annotation_id": annotation_id,
        "suite_id": suite_row["suite_id"],
        "suite_name": suite_row["name"],
        "attempt_id": attempt_id,
        "parent_annotation_id": parent_annotation_id,
        "annotator": annotator.strip(),
        "status": status,
        "has_error": has_error,
        "first_error_step": first_error_step,
        "error_bbox": bbox,
        "knowledge_points": points,
        "corrected_final_answer": corrected_final_answer,
        "correction_steps": steps,
        "notes": notes,
        "created_at": created_at,
    }


def list_evaluation_annotations(
    db_path: str | Path, *, suite: str, attempt_id: str
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    suite_row = _suite_row(connection, suite)
    rows = connection.execute(
        """SELECT * FROM evaluation_annotations
           WHERE suite_id=? AND attempt_id=?
           ORDER BY created_at DESC,annotation_id DESC""",
        (suite_row["suite_id"], attempt_id),
    ).fetchall()
    connection.close()
    return {
        "suite_id": suite_row["suite_id"],
        "suite_name": suite_row["name"],
        "attempt_id": attempt_id,
        "versions": [_annotation_dict(row) for row in rows],
    }


def list_evaluation_cases(
    db_path: str | Path,
    *,
    suite: str,
    annotation_status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    if annotation_status not in (None, "unannotated", "draft", "approved", "rejected"):
        raise ValueError(f"Unsupported annotation_status: {annotation_status}")
    connection = connect(db_path)
    initialize(connection)
    suite_row = _suite_row(connection, suite)
    attempt_ids = json.loads(suite_row["attempt_ids_json"])
    placeholders = ",".join("?" for _ in attempt_ids)
    rows = connection.execute(
        f"""WITH ranked AS (
               SELECT e.*,ROW_NUMBER() OVER(
                 PARTITION BY e.attempt_id
                 ORDER BY e.created_at DESC,e.annotation_id DESC
               ) AS rn
               FROM evaluation_annotations e WHERE e.suite_id=?
             )
             SELECT a.attempt_id,a.orig_q,a.image_source_path,a.domain_code,
                    a.subdomain_code,e.annotation_id,e.status,
                    e.has_error,e.first_error_step,e.error_bbox_json,
                    e.knowledge_points_json,e.corrected_final_answer,e.annotator,
                    e.created_at AS annotated_at
             FROM attempts a
             LEFT JOIN ranked e ON e.attempt_id=a.attempt_id AND e.rn=1
             WHERE a.attempt_id IN ({placeholders})
             ORDER BY a.attempt_id""",
        [suite_row["suite_id"], *attempt_ids],
    ).fetchall() if attempt_ids else []
    if annotation_status == "unannotated":
        rows = [row for row in rows if row["annotation_id"] is None]
    elif annotation_status:
        rows = [row for row in rows if row["status"] == annotation_status]
    total = len(rows)
    page = rows[max(0, offset):max(0, offset) + max(1, min(limit, 500))]
    connection.close()
    return {
        "suite_id": suite_row["suite_id"],
        "suite_name": suite_row["name"],
        "annotation_status": annotation_status,
        "total": total,
        "returned": len(page),
        "offset": max(0, offset),
        "cases": [_case_dict(row) for row in page],
    }


def annotation_progress(db_path: str | Path, *, suite: str) -> dict:
    connection = connect(db_path)
    initialize(connection)
    suite_row = _suite_row(connection, suite)
    attempt_count = len(json.loads(suite_row["attempt_ids_json"]))
    rows = connection.execute(
        """WITH ranked AS (
             SELECT status,attempt_id,ROW_NUMBER() OVER(
               PARTITION BY attempt_id
               ORDER BY created_at DESC,annotation_id DESC
             ) AS rn
             FROM evaluation_annotations WHERE suite_id=?
           )
           SELECT status,COUNT(*) AS count FROM ranked
           WHERE rn=1 GROUP BY status""",
        (suite_row["suite_id"],),
    ).fetchall()
    connection.close()
    counts = {"unannotated": 0, "draft": 0, "approved": 0, "rejected": 0}
    for row in rows:
        counts[row["status"]] = row["count"]
    counts["unannotated"] = attempt_count - sum(
        counts[status] for status in ("draft", "approved", "rejected")
    )
    return {
        "suite_id": suite_row["suite_id"],
        "suite_name": suite_row["name"],
        "attempt_count": attempt_count,
        "counts": counts,
        "approved_coverage": counts["approved"] / attempt_count if attempt_count else 0.0,
    }


def _annotation_dict(row) -> dict:
    result = dict(row)
    result["has_error"] = None if result["has_error"] is None else bool(result["has_error"])
    bbox_json = result.pop("error_bbox_json")
    points_json = result.pop("knowledge_points_json")
    steps_json = result.pop("correction_steps_json")
    result["error_bbox"] = json.loads(bbox_json) if bbox_json else None
    result["knowledge_points"] = json.loads(points_json)
    result["correction_steps"] = json.loads(steps_json)
    return result


def _case_dict(row) -> dict:
    result = dict(row)
    result["has_error"] = None if result["has_error"] is None else bool(result["has_error"])
    bbox_json = result.pop("error_bbox_json")
    points_json = result.pop("knowledge_points_json")
    result["error_bbox"] = json.loads(bbox_json) if bbox_json else None
    result["knowledge_points"] = json.loads(points_json) if points_json else []
    result["annotation_status"] = result.pop("status")
    return result
