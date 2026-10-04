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

TYPE_ADVICE = {
    "algebraic_manipulation": "逐步核对等式变形与符号",
    "arithmetic": "用估算或回代核查关键计算",
    "conceptual": "复述定义和适用条件后再解题",
    "notation": "完成解答后检查符号、单位与记号",
    "transcription": "对照题目和草稿检查誊写内容",
    "omitted_step": "补全推导中跳过的关键步骤",
    "assumption": "先写清已知条件和所用假设",
    "presentation": "按步骤写出依据与最终结论",
}


def _frequency_summary(categories: list[dict], domains: list[dict], total: int) -> str:
    """A readable analysis and study plan derived only from displayed counts."""
    if not total:
        return "分析：当前范围没有模型判错题目。建议：扩大时间范围后再查看错误分布。"
    leading = categories[:3]
    leading_count = sum(item["count"] for item in leading)
    type_text = "、".join(
        f"{item['label'][:20]}{item['count']}道" for item in leading
    )
    domain_text = "、".join(
        f"{item['name'][:20]}{item['count']}道" for item in domains[:3]
    )
    advice = [
        TYPE_ADVICE.get(item["error_type"], "回看对应错题并标出首个出错步骤")
        for item in categories[:2]
    ]
    while len(advice) < 2:
        advice.append("每周按首错步骤复做并对照原图")
    point = next(
        (item["name"][:18] for item in leading[0]["top_knowledge_points"]),
        "高频错题",
    )
    top_domain = domains[0] if domains else None
    domain_detail = (
        f"其中{top_domain['name'][:20]}的{top_domain['attempt_count']}道已诊断作答中，"
        f"模型判错{top_domain['count']}道（{top_domain['model_error_rate']:.1%}）；"
        f"该方向以{top_domain['types'][0]['label'][:20]}为最多，"
        f"有{top_domain['types'][0]['count']}道。"
        if top_domain and top_domain["types"] else ""
    )
    point_detail = (
        f"在{leading[0]['label'][:20]}中，“{point}”标签关联"
        f"{leading[0]['top_knowledge_points'][0]['error_count']}道错题，"
        "可作为优先回访的知识点线索。"
        if leading[0]["top_knowledge_points"] else ""
    )
    introduction = (
        f"分析：本期模型判错{total}道，{type_text}最常见，"
        f"前{len(leading)}类合计{leading_count}道，占本期错题的{leading_count / total:.1%}。"
        f"这表明复盘时可以先处理出现次数最多的错误类型，再检查较少见的错误。"
        f"按数学方向，{domain_text}数量居前。"
    )
    qualification = "各方向作答量不同，不能只凭错题数量判断哪个方向掌握得更差。"
    recommendations = (
        f"建议：①先按高频错误类型挑选代表题，查看手写原图、模型指出的首错步骤与订正，"
        f"确认错误究竟发生在概念选择、列式还是推导过程；"
        f"②围绕“{point}”做一组同知识点的相近题，逐题写下关键依据；"
        f"③针对前两类高频错误，分别练习{advice[0]}、{advice[1]}；"
        f"④隔几天重新独立完成这些题，对照原解答检查同一错误是否再次出现，"
        f"再决定是否扩大到其他数学方向。"
    )
    details = [domain_detail, qualification, point_detail]
    summary = introduction + "".join(details) + recommendations
    while len(summary) > 500 and details:
        details.pop()
        summary = introduction + "".join(details) + recommendations
    if len(summary) > 500:
        return (
            f"分析：本期模型判错{total}道；{type_text}最常见，"
            f"合计{leading_count}道（{leading_count / total:.1%}）。"
            f"按数学方向，{domain_text}数量居前；各方向作答量不同。"
            f"建议：①回访“{point}”相关题；②{advice[0]}；"
            f"③{advice[1]}，再对照首错步骤复做。"
        )
    return summary


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
    diagnosed_by_domain = Counter(item["domain_code"] for item in attempts)
    errors_by_domain: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        errors_by_domain[case["domain_code"]].append(case)
    domains = []
    for code, members in errors_by_domain.items():
        type_counts = Counter(case["error_type"] for case in members)
        domains.append({
            "domain_code": code,
            "name": DOMAIN_LABELS.get(code, code),
            "count": len(members),
            "attempt_count": diagnosed_by_domain[code],
            "model_error_rate": round(len(members) / diagnosed_by_domain[code], 4),
            "types": [
                {"error_type": error_type,
                 "label": ERROR_LABELS.get(error_type, error_type),
                 "count": count}
                for error_type, count in type_counts.most_common()
            ],
        })
    domains.sort(key=lambda row: (-row["count"], row["domain_code"]))
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
        "domains": domains,
        "frequency_summary": _frequency_summary(categories, domains, len(cases)),
        "monthly": [
            {"month": month, "total": sum(counts.values()),
             "types": dict(counts)}
            for month, counts in sorted(monthly.items())
        ],
        "weaknesses": weaknesses,
        "cases": cases,
        "source": "vlm_diagnosis_with_teacher_overrides",
    }
