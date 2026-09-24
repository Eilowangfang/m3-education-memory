from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .database import connect, initialize


@dataclass(frozen=True)
class MemoryQuery:
    domain_code: str | None = None
    subdomain_code: str | None = None
    errors_only: bool = False
    time_from: str | None = None
    time_to: str | None = None
    limit: int = 100
    offset: int = 0


def resolve_recent_vlm_window(
    db_path: str | Path,
    *,
    model: str,
    recent_months: int | None,
    domain_code: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve relative months against the configured simulation anchor."""
    if not recent_months:
        return None, None
    connection = connect(db_path)
    params: list[object] = [model]
    domain_filter = ""
    if domain_code:
        domain_filter = " AND a.domain_code=?"
        params.append(domain_code)
    row = connection.execute(
        """SELECT MAX(a.Time) AS latest_time
           FROM attempts a JOIN analysis_runs r USING(attempt_id)
           WHERE r.model=? AND r.status='completed'""" + domain_filter,
        params,
    ).fetchone()
    latest_value = row["latest_time"] if row else None
    metadata = dict(connection.execute(
        "SELECT key,value FROM metadata WHERE key IN ('simulation_anchor_date','timezone')"
    ).fetchall())
    connection.close()
    anchor_value = metadata.get("simulation_anchor_date")
    if anchor_value:
        anchor = date.fromisoformat(anchor_value)
        latest = datetime(
            anchor.year, anchor.month, anchor.day, 23, 59, 59,
            tzinfo=ZoneInfo(metadata.get("timezone", "Asia/Shanghai")),
        )
    elif latest_value:
        latest = datetime.fromisoformat(latest_value)
    else:
        return None, None
    month_index = latest.year * 12 + latest.month - (recent_months - 1)
    start_year = (month_index - 1) // 12
    start_month = (month_index - 1) % 12 + 1
    start = latest.replace(
        year=start_year, month=start_month, day=1,
        hour=0, minute=0, second=0, microsecond=0,
    )
    return start.isoformat(), latest.isoformat()


def query_attempts(db_path: str | Path, query: MemoryQuery) -> dict:
    connection = connect(db_path)
    where: list[str] = []
    params: list[object] = []
    if query.domain_code:
        where.append("a.domain_code = ?")
        params.append(query.domain_code)
    if query.subdomain_code:
        where.append("a.subdomain_code = ?")
        params.append(query.subdomain_code)
    if query.errors_only:
        where.append("b.has_error = 1")
    if query.time_from:
        where.append("a.Time >= ?")
        params.append(query.time_from)
    if query.time_to:
        where.append("a.Time <= ?")
        params.append(query.time_to)
    clause = " WHERE " + " AND ".join(where) if where else ""

    total = connection.execute(
        "SELECT COUNT(*) FROM attempts a JOIN benchmark_truth b USING(attempt_id)" + clause,
        params,
    ).fetchone()[0]
    rows = connection.execute(
        """SELECT a.*,b.has_error,b.orig_a,b.pert_a,b.pert_reasoning,
                  b.reference_error_type
           FROM attempts a JOIN benchmark_truth b USING(attempt_id)"""
        + clause
        + " ORDER BY a.Time, a.attempt_id LIMIT ? OFFSET ?",
        [*params, query.limit, query.offset],
    ).fetchall()
    summary_rows = connection.execute(
        """SELECT COALESCE(a.subdomain_code,'unknown') AS subdomain_code,
                  b.reference_error_type,COUNT(*) AS count
           FROM attempts a JOIN benchmark_truth b USING(attempt_id)"""
        + clause
        + " GROUP BY a.subdomain_code,b.reference_error_type ORDER BY count DESC",
        params,
    ).fetchall()
    connection.close()
    return {
        "query": {
            "domain_code": query.domain_code,
            "subdomain_code": query.subdomain_code,
            "errors_only": query.errors_only,
            "time_from": query.time_from,
            "time_to": query.time_to,
            "limit": query.limit,
            "offset": query.offset,
        },
        "total": total,
        "returned": len(rows),
        "source": "benchmark_reference",
        "time_notice": "Time is simulated from grade and is not a real attempt timestamp.",
        "summary": [dict(row) for row in summary_rows],
        "attempts": [dict(row) for row in rows],
    }


def query_vlm_memories(
    db_path: str | Path,
    *,
    model: str,
    domain_code: str | None = None,
    subdomain_code: str | None = None,
    attempt_id: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    errors_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Retrieve latest image-only VLM diagnoses without joining benchmark truth."""
    connection = connect(db_path)
    initialize(connection)
    base = f"""WITH ranked AS (
                 SELECT r.*,
                        ROW_NUMBER() OVER (
                          PARTITION BY r.attempt_id
                          ORDER BY r.created_at DESC,r.run_id DESC
                        ) AS row_number
                 FROM analysis_runs r
                 WHERE r.model=? AND r.status='completed'
               ), review_ranked AS (
                 SELECT v.*,
                        ROW_NUMBER() OVER (
                          PARTITION BY v.run_id
                          ORDER BY v.created_at DESC,v.review_id DESC
                        ) AS row_number
                 FROM diagnosis_reviews v
               ), correction_ranked AS (
                 SELECT cv.*,
                        ROW_NUMBER() OVER (
                          PARTITION BY cv.attempt_id,cv.model
                          ORDER BY cv.created_at DESC,cv.version_id DESC
                        ) AS row_number
                 FROM correction_versions cv WHERE cv.status='active'
               )
               SELECT a.attempt_id,a.Time,a.grade,a.domain_code,a.subdomain_code,
                      a.orig_q,a.image_source_path,a.image_sha256,
                      r.run_id,r.provider,r.model,r.prompt_version,r.created_at,
                      r.input_image_bytes,r.input_width,r.input_height,
                      r.input_mime_type,r.preprocessing_json,r.latency_ms,
                      r.provider_usage_json,r.request_attempts,
                      d.question_transcription,d.solution_transcription,
                      d.has_error_pred,d.error_type_pred,d.first_error_step,
                      d.error_explanation,d.knowledge_points_json,d.confidence,
                      d.requires_review,
                      COALESCE(v.has_error_override,d.has_error_pred) AS has_error_effective,
                      COALESCE(v.error_type_override,d.error_type_pred) AS error_type_effective,
                      CASE WHEN COALESCE(v.has_error_override,d.has_error_pred)=0
                           THEN NULL
                           ELSE COALESCE(v.first_error_step_override,d.first_error_step)
                      END AS first_error_step_effective,
                      COALESCE(v.knowledge_points_override_json,d.knowledge_points_json)
                        AS knowledge_points_effective_json,
                      COALESCE(v.error_explanation_override,d.error_explanation)
                        AS error_explanation_effective,
                      v.error_bbox_override_json AS error_bbox_effective_json,
                      v.review_id,v.verdict AS review_verdict,v.reviewer,
                      v.notes AS review_notes,
                      v.created_at AS reviewed_at,
                      COALESCE(cv.version_id,c.correction_id) AS correction_id,
                      cv.version_id AS correction_version_id,
                      COALESCE(cv.source,'vlm') AS correction_source,
                      COALESCE(cv.corrected_solution,c.corrected_solution) AS corrected_solution,
                      COALESCE(cv.corrected_steps_json,c.corrected_steps_json) AS corrected_steps_json,
                      COALESCE(cv.final_answer,c.final_answer) AS final_answer,
                      COALESCE(cv.verification_expression,c.verification_expression) AS verification_expression,
                      COALESCE(cv.verification_status,c.verification_status) AS verification_status,
                      COALESCE(cv.verification_method,c.verification_method) AS verification_method,
                      COALESCE(cv.verification_details_json,c.verification_details_json) AS verification_details_json,
                      COALESCE(cv.confidence,c.confidence) AS correction_confidence,
                      COALESCE(cv.rendered_path,c.rendered_path) AS rendered_path
               FROM ranked r
               JOIN attempts a ON a.attempt_id=r.attempt_id
               JOIN diagnoses d ON d.run_id=r.run_id
               LEFT JOIN corrections c ON c.run_id=r.run_id
               LEFT JOIN review_ranked v ON v.run_id=r.run_id AND v.row_number=1
               LEFT JOIN correction_ranked cv
                 ON cv.attempt_id=a.attempt_id AND cv.model=r.model AND cv.row_number=1
               WHERE r.row_number=1"""
    # The model predicate is already applied inside the CTE.
    filters = []
    filter_params: list[object] = [model]
    if domain_code:
        filters.append("a.domain_code=?")
        filter_params.append(domain_code)
    if subdomain_code:
        filters.append("a.subdomain_code=?")
        filter_params.append(subdomain_code)
    if attempt_id:
        filters.append("a.attempt_id=?")
        filter_params.append(attempt_id)
    if time_from:
        filters.append("a.Time>=?")
        filter_params.append(time_from)
    if time_to:
        filters.append("a.Time<=?")
        filter_params.append(time_to)
    if errors_only:
        filters.append("COALESCE(v.has_error_override,d.has_error_pred)=1")
    if filters:
        base += " AND " + " AND ".join(filters)
    count_sql = "SELECT COUNT(*) FROM (" + base + ")"
    total = connection.execute(count_sql, filter_params).fetchone()[0]
    rows = connection.execute(
        base + " ORDER BY a.Time,a.attempt_id LIMIT ? OFFSET ?",
        [*filter_params, limit, offset],
    ).fetchall()
    result_rows = []
    for row in rows:
        item = dict(row)
        item["analysis_observability"] = {
            "input_image_bytes": item.pop("input_image_bytes"),
            "input_width": item.pop("input_width"),
            "input_height": item.pop("input_height"),
            "input_mime_type": item.pop("input_mime_type"),
            "preprocessing": json.loads(item.pop("preprocessing_json") or "{}"),
            "latency_ms": item.pop("latency_ms"),
            "provider_usage": json.loads(item.pop("provider_usage_json") or "{}"),
            "request_attempts": item.pop("request_attempts"),
        }
        item["knowledge_points_pred"] = json.loads(item.pop("knowledge_points_json"))
        item["knowledge_points_effective"] = json.loads(
            item.pop("knowledge_points_effective_json")
        )
        item["knowledge_points"] = item["knowledge_points_effective"]
        item["error_explanation_pred"] = item["error_explanation"]
        item["error_explanation"] = item.pop("error_explanation_effective")
        bbox_override_json = item.pop("error_bbox_effective_json")
        item["error_bbox_effective"] = (
            json.loads(bbox_override_json) if bbox_override_json else None
        )
        item["review"] = (
            {
                "review_id": item.pop("review_id"),
                "verdict": item.pop("review_verdict"),
                "reviewer": item.pop("reviewer"),
                "notes": item.pop("review_notes"),
                "reviewed_at": item.pop("reviewed_at"),
            }
            if item["review_id"] else None
        )
        if item["review"] is None:
            for key in (
                "review_id", "review_verdict", "reviewer",
                "review_notes", "reviewed_at",
            ):
                item.pop(key, None)
        if item["correction_id"]:
            item["correction"] = {
                "correction_id": item.pop("correction_id"),
                "version_id": item.pop("correction_version_id"),
                "source": item.pop("correction_source"),
                "corrected_solution": item.pop("corrected_solution"),
                "corrected_steps": json.loads(item.pop("corrected_steps_json")),
                "final_answer": item.pop("final_answer"),
                "verification_expression": item.pop("verification_expression"),
                "verification_status": item.pop("verification_status"),
                "verification_method": item.pop("verification_method"),
                "verification_details": json.loads(item.pop("verification_details_json")),
                "confidence": item.pop("correction_confidence"),
                "rendered_path": item.pop("rendered_path"),
            }
        else:
            for key in (
                "correction_id", "corrected_solution", "corrected_steps_json",
                "correction_version_id", "correction_source",
                "final_answer", "verification_expression", "verification_status",
                "verification_method", "verification_details_json",
                "correction_confidence", "rendered_path",
            ):
                item.pop(key)
            item["correction"] = None
        item["steps"] = [
            dict(step)
            for step in connection.execute(
                """SELECT step_index,transcription,normalized_latex,bbox_json,
                          is_error,confidence
                   FROM diagnosis_steps WHERE run_id=? ORDER BY step_index""",
                (item["run_id"],),
            ).fetchall()
        ]
        for step in item["steps"]:
            step["bbox"] = (
                None if step["bbox_json"] is None else json.loads(step["bbox_json"])
            )
            del step["bbox_json"]
        if item["has_error_effective"] and item["error_bbox_effective"] is None:
            preferred = next(
                (
                    step for step in item["steps"]
                    if step["step_index"] == item["first_error_step_effective"]
                    and step["bbox"] is not None
                ),
                None,
            )
            if preferred is None:
                preferred = next(
                    (step for step in item["steps"] if step["is_error"] and step["bbox"]),
                    None,
                )
            item["error_bbox_effective"] = preferred["bbox"] if preferred else None
        item["display_source"] = "teacher_override" if item["review"] else "vlm"
        result_rows.append(item)
    connection.close()
    return {
        "model": model,
        "domain_code": domain_code,
        "subdomain_code": subdomain_code,
        "attempt_id": attempt_id,
        "errors_only": errors_only,
        "total": total,
        "returned": len(result_rows),
        "source": "vlm_diagnosis",
        "time_notice": "Time is simulated from grade.",
        "attempts": result_rows,
    }
