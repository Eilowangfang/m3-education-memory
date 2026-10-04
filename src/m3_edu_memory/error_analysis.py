"""Evidence-backed aggregation for the student error-analysis demo."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .control import plan_memory_query
from .control_tools import ERROR_LABELS
from .database import connect
from .query import query_vlm_memories, resolve_recent_vlm_window


DOMAIN_LABELS = {
    "clc": "微积分", "alg": "代数", "art": "算术", "mgm": "几何",
    "pst": "概率统计", "trg": "三角学", "apt": "综合能力",
}


def analyze_errors(db_path: str | Path, *, request: str, model: str) -> dict:
    """Analyze the full requested cohort; every displayed count is traceable to cases."""
    plan = plan_memory_query(request)
    time_from, time_to = resolve_recent_vlm_window(
        db_path, model=model, recent_months=plan.recent_months,
        domain_code=plan.domain_code,
    )
    result = query_vlm_memories(
        db_path, model=model, domain_code=plan.domain_code,
        time_from=time_from, time_to=time_to, limit=100_000,
    )
    attempts = result["attempts"]
    connection = connect(db_path)
    where, params = [], []
    if plan.domain_code:
        where.append("domain_code=?")
        params.append(plan.domain_code)
    if time_from:
        where.append("Time>=?")
        params.append(time_from)
    if time_to:
        where.append("Time<=?")
        params.append(time_to)
    clause = " WHERE " + " AND ".join(where) if where else ""
    cohort_total = connection.execute(
        "SELECT COUNT(*) FROM attempts" + clause, params
    ).fetchone()[0]
    connection.close()

    cases = []
    by_type: dict[str, list[dict]] = defaultdict(list)
    all_point_attempts: Counter[str] = Counter()
    monthly: dict[str, Counter[str]] = defaultdict(Counter)
    for item in attempts:
        points = list(dict.fromkeys(
            str(point).strip() for point in item["knowledge_points"]
            if str(point).strip()
        ))
        all_point_attempts.update(points)
        if not item["has_error_effective"]:
            continue
        error_type = item["error_type_effective"] or "uncertain"
        step_number = item["first_error_step_effective"]
        first_step = next(
            (step for step in item["steps"] if step["step_index"] == step_number),
            None,
        )
        case = {
            "attempt_id": item["attempt_id"],
            "time": item["Time"],
            "month": item["Time"][:7],
            "domain_code": item["domain_code"],
            "domain_label": DOMAIN_LABELS.get(item["domain_code"], item["domain_code"]),
            "error_type": error_type,
            "error_label": ERROR_LABELS.get(error_type, error_type),
            "question": item["question_transcription"] or item["orig_q"],
            "explanation": item["error_explanation"] or "模型尚未给出具体原因。",
            "first_error_step": step_number,
            "trigger": first_step["transcription"] if first_step else None,
            "has_bbox": item["error_bbox_effective"] is not None,
            "knowledge_points": points,
            "confidence": item["confidence"],
            "display_source": item["display_source"],
        }
        cases.append(case)
        by_type[error_type].append(case)
        monthly[case["month"]][error_type] += 1

    categories = []
    for error_type, members in by_type.items():
        point_counts = Counter(
            point for case in members for point in case["knowledge_points"]
        )
        domains = Counter(case["domain_label"] for case in members)
        categories.append({
            "error_type": error_type,
            "label": ERROR_LABELS.get(error_type, error_type),
            "count": len(members),
            "share": round(len(members) / len(cases), 4) if cases else 0,
            "top_knowledge_points": [
                {"name": point, "error_count": count,
                 "attempt_count": all_point_attempts[point]}
                for point, count in point_counts.most_common(5)
            ],
            "domains": [
                {"name": domain, "count": count}
                for domain, count in domains.most_common()
            ],
            "trigger_available": sum(bool(case["trigger"]) for case in members),
            "bbox_available": sum(case["has_bbox"] for case in members),
            "example_attempt_id": members[0]["attempt_id"],
        })
    categories.sort(key=lambda row: (-row["count"], row["error_type"]))
    cases.sort(key=lambda row: (row["time"], row["attempt_id"]), reverse=True)
    point_errors = Counter(
        point for case in cases for point in case["knowledge_points"]
    )
    weaknesses = [
        {"knowledge_point": point, "error_count": count,
         "attempt_count": all_point_attempts[point],
         "error_rate": round(count / all_point_attempts[point], 4),
         "error_types": [
             {"error_type": category["error_type"], "label": category["label"],
              "count": sum(point in case["knowledge_points"] for case in by_type[category["error_type"]])}
             for category in categories
             if any(point in case["knowledge_points"] for case in by_type[category["error_type"]])
         ]}
        for point, count in point_errors.most_common()
    ]
    return {
        "request": request,
        "model": model,
        "domain_code": plan.domain_code,
        "recent_months": plan.recent_months,
        "time_window": {"from": time_from, "to": time_to},
        "coverage": {
            "cohort_attempts": cohort_total,
            "diagnosed_attempts": len(attempts),
            "missing_diagnosis": cohort_total - len(attempts),
            "error_attempts": len(cases),
            "trigger_available": sum(bool(case["trigger"]) for case in cases),
            "bbox_available": sum(case["has_bbox"] for case in cases),
        },
        "categories": categories,
        "monthly": [
            {"month": month, "total": sum(counts.values()),
             "types": dict(counts)}
            for month, counts in sorted(monthly.items())
        ],
        "weaknesses": weaknesses,
        "cases": cases,
        "source": "vlm_diagnosis_with_teacher_overrides",
    }
