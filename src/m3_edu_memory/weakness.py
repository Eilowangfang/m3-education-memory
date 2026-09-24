from __future__ import annotations

import json
from pathlib import Path

from .database import connect
from .vlm import PROMPT_VERSION


def extract_weaknesses(
    db_path: str | Path, *, domain_code: str | None = None
) -> dict:
    """Build transparent weakness indicators from benchmark-labeled attempts."""
    connection = connect(db_path)
    where = "WHERE a.domain_code = ?" if domain_code else ""
    params = [domain_code] if domain_code else []
    groups = connection.execute(
        f"""SELECT a.domain_code,COALESCE(a.subdomain_code,'unknown') AS subdomain_code,
                   COUNT(*) AS attempt_count,
                   SUM(b.has_error) AS error_count,
                   MAX(a.Time) AS latest_Time,
                   AVG(b.has_error) AS error_rate,
                   (SUM(CAST(substr(a.Time,6,2) AS INTEGER) * b.has_error) * 1.0 /
                    NULLIF(SUM(CAST(substr(a.Time,6,2) AS INTEGER)),0)) AS recent_weighted_error_rate
            FROM attempts a JOIN benchmark_truth b USING(attempt_id)
            {where}
            GROUP BY a.domain_code,a.subdomain_code""",
        params,
    ).fetchall()
    weaknesses = []
    for group in groups:
        error_types = connection.execute(
            """SELECT b.reference_error_type,COUNT(*) AS count
               FROM attempts a JOIN benchmark_truth b USING(attempt_id)
               WHERE a.domain_code=? AND COALESCE(a.subdomain_code,'unknown')=?
                 AND b.has_error=1
               GROUP BY b.reference_error_type ORDER BY count DESC""",
            (group["domain_code"], group["subdomain_code"]),
        ).fetchall()
        evidence = connection.execute(
            """SELECT a.attempt_id FROM attempts a
               JOIN benchmark_truth b USING(attempt_id)
               WHERE a.domain_code=? AND COALESCE(a.subdomain_code,'unknown')=?
                 AND b.has_error=1 ORDER BY a.Time,a.attempt_id""",
            (group["domain_code"], group["subdomain_code"]),
        ).fetchall()
        attempts = group["attempt_count"]
        errors = group["error_count"]
        smoothed = (errors + 1) / (attempts + 2)
        recent = group["recent_weighted_error_rate"] or 0.0
        score = round(100 * (0.6 * smoothed + 0.4 * recent), 2)
        weaknesses.append(
            {
                "domain_code": group["domain_code"],
                "subdomain_code": group["subdomain_code"],
                "attempt_count": attempts,
                "error_count": errors,
                "error_rate": round(group["error_rate"], 4),
                "recent_weighted_error_rate": round(recent, 4),
                "weakness_score": score,
                "latest_Time": group["latest_Time"],
                "error_types": [dict(row) for row in error_types],
                "evidence_attempt_ids": [row["attempt_id"] for row in evidence],
            }
        )
    connection.close()
    weaknesses.sort(
        key=lambda item: (item["weakness_score"], item["error_count"]), reverse=True
    )
    return {
        "domain_code": domain_code,
        "source": "benchmark_reference",
        "time_notice": "Time is simulated from grade; trend weights are demonstrative.",
        "score_formula": (
            "100 * (0.6 * ((errors + 1) / (attempts + 2)) + "
            "0.4 * month-weighted error rate)"
        ),
        "weaknesses": weaknesses,
    }


def extract_vlm_weaknesses(
    db_path: str | Path,
    *,
    model: str,
    domain_code: str | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> dict:
    """Return explainable mastery hypotheses built only from VLM diagnoses."""
    connection = connect(db_path)
    where = ["model=?", "prompt_version=?", "error_count>0"]
    params: list[object] = [model, prompt_version]
    if domain_code:
        where.append("domain_code=?")
        params.append(domain_code)
    rows = connection.execute(
        "SELECT * FROM mastery_states WHERE " + " AND ".join(where)
        + " ORDER BY weakness_score DESC,error_count DESC,knowledge_point",
        params,
    ).fetchall()
    connection.close()
    weaknesses = []
    for row in rows:
        item = dict(row)
        item["score_components"] = json.loads(item.pop("score_components_json"))
        item["error_types"] = json.loads(item.pop("error_types_json"))
        item["evidence_attempt_ids"] = json.loads(
            item.pop("evidence_attempt_ids_json")
        )
        weaknesses.append(item)
    return {
        "model": model,
        "prompt_version": prompt_version,
        "domain_code": domain_code,
        "source": "vlm_diagnosis",
        "status": "hypothesis",
        "time_notice": "Time is simulated from grade; recent weighting is demonstrative.",
        "score_formula": (
            "100 * (0.4*smoothed_error_rate + 0.25*recent_weighted_error_rate "
            "+ 0.2*repeat_factor + 0.15*correction_gap)"
        ),
        "weaknesses": weaknesses,
    }


def mastery_history(
    db_path: str | Path,
    *,
    model: str,
    domain_code: str | None = None,
    knowledge_point: str | None = None,
    prompt_version: str | None = None,
    limit: int = 100,
) -> dict:
    """Return immutable mastery snapshots, newest first, for audit and trends."""
    connection = connect(db_path)
    where = ["model=?"]
    params: list[object] = [model]
    if domain_code:
        where.append("domain_code=?")
        params.append(domain_code)
    if knowledge_point:
        where.append("knowledge_point=?")
        params.append(knowledge_point)
    if prompt_version:
        where.append("prompt_version=?")
        params.append(prompt_version)
    params.append(max(1, min(int(limit), 1000)))
    rows = connection.execute(
        "SELECT * FROM mastery_state_history WHERE " + " AND ".join(where)
        + " ORDER BY created_at DESC,snapshot_id DESC LIMIT ?",
        params,
    ).fetchall()
    connection.close()
    snapshots = []
    for row in rows:
        item = dict(row)
        item["score_components"] = json.loads(item.pop("score_components_json"))
        item["error_types"] = json.loads(item.pop("error_types_json"))
        item["evidence_attempt_ids"] = json.loads(
            item.pop("evidence_attempt_ids_json")
        )
        item.pop("state_hash", None)
        snapshots.append(item)
    return {
        "model": model,
        "domain_code": domain_code,
        "knowledge_point": knowledge_point,
        "prompt_version": prompt_version,
        "total": len(snapshots),
        "snapshots": snapshots,
    }
