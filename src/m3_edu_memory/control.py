from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from .audit_render import render_audit_image
from .media import image_bytes_from_source
from .query import query_vlm_memories, resolve_recent_vlm_window
from .weakness import extract_vlm_weaknesses


DOMAIN_ALIASES = {
    "微积分": "clc",
    "calculus": "clc",
    "代数": "alg",
    "algebra": "alg",
    "算术": "art",
    "arithmetic": "art",
    "几何": "mgm",
    "geometry": "mgm",
    "概率": "pst",
    "统计": "pst",
    "probability": "pst",
    "statistics": "pst",
    "三角": "trg",
    "trigonometry": "trg",
    "综合能力": "apt",
    "aptitude": "apt",
}


@dataclass(frozen=True)
class QueryPlan:
    request: str
    domain_code: str | None
    errors_only: bool
    limit: int
    recent_months: int | None = None
    strategy: str = "structured_filter_then_evidence_expansion"


def _chinese_integer(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(value)


def plan_memory_query(request: str, *, default_limit: int = 10) -> QueryPlan:
    normalized = request.casefold().strip()
    domain = next(
        (code for alias, code in DOMAIN_ALIASES.items() if alias in normalized),
        None,
    )
    errors_only = any(token in normalized for token in ("错题", "错误", "薄弱", "订正", "error", "wrong"))
    match = re.search(r"(?:前|最近|抽取|选取)?\s*(\d{1,3})\s*(?:道|个|条)", normalized)
    limit = int(match.group(1)) if match else default_limit
    month_match = re.search(
        r"(?:过去|最近|近)\s*([0-9一二两三四五六七八九十]+)\s*(?:个)?月",
        normalized,
    )
    recent_months = _chinese_integer(month_match.group(1)) if month_match else None
    if recent_months is not None:
        recent_months = max(1, min(recent_months, 120))
    return QueryPlan(
        request=request,
        domain_code=domain,
        errors_only=errors_only,
        limit=max(1, min(limit, 500)),
        recent_months=recent_months,
    )


def _error_summary(attempts: list[dict]) -> list[dict]:
    counts: dict[str, dict] = {}
    for attempt in attempts:
        error_type = attempt.get("error_type_effective") or attempt["error_type_pred"]
        entry = counts.setdefault(
            error_type,
            {"error_type": error_type, "count": 0, "evidence_attempt_ids": []},
        )
        entry["count"] += 1
        entry["evidence_attempt_ids"].append(attempt["attempt_id"])
    return sorted(counts.values(), key=lambda item: (-item["count"], item["error_type"]))


def build_memory_report(
    db_path: str | Path,
    *,
    request: str,
    model: str,
    output_dir: str | Path,
    default_limit: int = 10,
) -> dict:
    """Execute a deterministic M3-style query plan and export an evidence report."""
    plan = plan_memory_query(request, default_limit=default_limit)
    time_from, time_to = resolve_recent_vlm_window(
        db_path,
        model=model,
        recent_months=plan.recent_months,
        domain_code=plan.domain_code,
    )
    memories = query_vlm_memories(
        db_path,
        model=model,
        domain_code=plan.domain_code,
        errors_only=plan.errors_only,
        time_from=time_from,
        time_to=time_to,
        limit=plan.limit,
    )
    weakness_result = extract_vlm_weaknesses(
        db_path, model=model, domain_code=plan.domain_code
    )
    root = Path(output_dir)
    originals = root / "images"
    audits = root / "audits"
    originals.mkdir(parents=True, exist_ok=True)
    audits.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    exported: list[dict] = []
    for attempt in memories["attempts"]:
        attempt_id = attempt["attempt_id"]
        original_path = originals / f"{attempt_id}.png"
        with Image.open(BytesIO(image_bytes_from_source(attempt["image_source_path"]))) as source:
            source.convert("RGB").save(original_path, format="PNG")

        audit_path = audits / f"{attempt_id}_audit.png"
        audit_error = None
        correction = attempt.get("correction")
        if correction:
            stored = correction.get("rendered_path")
            if stored and Path(stored).exists():
                stored_path = Path(stored).resolve()
                if stored_path != audit_path.resolve():
                    shutil.copy2(stored_path, audit_path)
            else:
                try:
                    render_audit_image(
                        db_path,
                        attempt_id=attempt_id,
                        run_id=attempt["run_id"],
                        output_path=audit_path,
                    )
                except Exception as exc:
                    audit_error = str(exc)

        exported.append(
            {
                "attempt_id": attempt_id,
                "original_image": str(original_path.resolve()),
                "audit_image": str(audit_path.resolve()) if audit_path.exists() else None,
                "audit_error": audit_error,
            }
        )
        evidence = html.escape(attempt_id)
        knowledge = ", ".join(attempt["knowledge_points"])
        correction_text = correction["corrected_solution"] if correction else "尚无订正"
        verification = correction["verification_status"] if correction else "not_available"
        audit_figure = (
            f"<figure><img src='audits/{audit_path.name}' alt='可审计订正图'>"
            "<figcaption>可审计标注与订正</figcaption></figure>"
            if audit_path.exists()
            else f"<div class='missing'>订正图未生成：{html.escape(audit_error or '无订正')}</div>"
        )
        cards.append(
            "<article class='card'>"
            f"<h3>{evidence}</h3>"
            f"<p><b>模拟时间：</b>{html.escape(attempt['Time'])}</p>"
            f"<p><b>知识点：</b>{html.escape(knowledge)}</p>"
            f"<p><b>错因：</b>{html.escape(attempt['error_explanation'])}</p>"
            f"<p><b>正确结果：</b>{html.escape(correction_text)}</p>"
            f"<p><b>校验：</b>{html.escape(verification)}</p>"
            "<div class='pair'>"
            f"<figure><img src='images/{original_path.name}' alt='手写错题原图'>"
            "<figcaption>原始手写作答</figcaption></figure>"
            f"{audit_figure}</div>"
            f"<p class='review-action'><a class='review-link' href='http://127.0.0.1:8765/review?attempt_id={html.escape(attempt_id)}&amp;model={html.escape(model)}'>教师修改 VLM 诊断或订正</a></p>"
            "</article>"
        )

    error_summary = _error_summary(memories["attempts"])
    weaknesses = weakness_result["weaknesses"]
    summary_items = "".join(
        f"<li><b>{html.escape(item['error_type'])}</b>：{item['count']} 次；证据 "
        f"{html.escape(', '.join(item['evidence_attempt_ids']))}</li>"
        for item in error_summary
    ) or "<li>当前 VLM 记忆中没有符合条件的错题。</li>"
    weakness_items = "".join(
        f"<li><b>{html.escape(item['knowledge_point'])}</b>：薄弱分 "
        f"{item['weakness_score']:.2f}，错误 {item['error_count']}/{item['attempt_count']}；"
        f"证据 {html.escape(', '.join(item['evidence_attempt_ids']))}</li>"
        for item in weaknesses[:10]
    ) or "<li>尚无足够的 VLM 诊断形成薄弱点假设。</li>"

    report = {
        "request": request,
        "query_plan": asdict(plan),
        "resolved_time_window": {"time_from": time_from, "time_to": time_to},
        "model": model,
        "source": "vlm_diagnosis_only",
        "time_notice": "Time 由 grade 模拟，不是真实作答时间。",
        "total_matched": memories["total"],
        "returned": memories["returned"],
        "error_summary": error_summary,
        "weaknesses": weaknesses,
        "evidence": memories["attempts"],
        "artifacts": exported,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>M3 教育记忆报告</title><style>
body{{margin:0;background:#f4f6f8;color:#1f2933;font-family:'Microsoft YaHei',Segoe UI,sans-serif}}
main{{max-width:1400px;margin:auto;padding:28px}}h1,h2{{color:#17324d}}
.notice{{background:#fff8db;border-left:5px solid #d5a514;padding:14px 18px}}
.summary,.card{{background:white;border:1px solid #d9e0e7;border-radius:12px;padding:20px;margin:20px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0}}
img{{display:block;max-width:100%;height:auto;border:1px solid #cfd8e3;border-radius:6px}}
figcaption{{padding-top:8px;color:#52606d}}.missing{{padding:20px;background:#f1f3f5}}
code{{background:#eef2f5;padding:2px 5px}}@media(max-width:860px){{.pair{{grid-template-columns:1fr}}}}
.review-link{{display:inline-block;padding:8px 12px;background:#155e75;color:white;text-decoration:none;border-radius:5px}}
.review-action{{margin:22px 0 0;padding-top:16px;border-top:1px solid #d9e0e7}}
</style></head><body><main>
<h1>M3-Agent 教育错题记忆报告</h1>
<p><b>用户请求：</b>{html.escape(request)}</p>
<p class='notice'>本报告使用 VLM 诊断及教师运行时修订；时间由 grade 模拟。教师修改不会改变 FERMAT 评测 ground truth，每条结论保留证据 ID。</p>
<section class='summary'><h2>1. 错因与薄弱点总结</h2><h3>错误类型</h3><ul>{summary_items}</ul>
<h3>知识点薄弱状态</h3><ul>{weakness_items}</ul></section>
<section><h2>2. 错题回访与 3. 可审计订正</h2>{''.join(cards)}</section>
</main></body></html>"""
    (root / "gallery.html").write_text(document, encoding="utf-8")
    return {
        "request": request,
        "query_plan": asdict(plan),
        "matched": memories["total"],
        "returned": memories["returned"],
        "report_json": str((root / "report.json").resolve()),
        "gallery_html": str((root / "gallery.html").resolve()),
    }
