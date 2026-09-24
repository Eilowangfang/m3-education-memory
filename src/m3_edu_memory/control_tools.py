from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .audit_render import AUDIT_RENDER_VERSION, render_audit_image
from .control import DOMAIN_ALIASES, plan_memory_query
from .database import connect
from .query import query_vlm_memories, resolve_recent_vlm_window
from .retrieval import EmbeddingClient, hybrid_search, memory_index_stats
from .weakness import extract_vlm_weaknesses


ERROR_LABELS = {
    "conceptual": "概念理解错误",
    "assumption": "错误假设",
    "algebraic_manipulation": "代数变形错误",
    "arithmetic": "计算错误",
    "notation": "符号与记号错误",
    "transcription": "誊写错误",
    "omitted_step": "步骤遗漏",
    "presentation": "表达不规范",
    "no_actual_error": "实际无错",
    "uncertain": "尚不确定",
}


def summarize_weaknesses(weaknesses: list[dict], evidence: list[dict]) -> dict:
    """Create an auditable narrative from VLM-produced diagnosis fields."""
    if not weaknesses:
        return {
            "text": "当前大模型诊断样本不足，尚不能形成稳定的薄弱点结论。",
            "source": "derived_from_vlm_diagnoses",
            "evidence_count": len(evidence),
        }
    top = weaknesses[:3]
    point_names = [str(item["knowledge_point"]) for item in top]
    display_names = [re.split(r"[（(]", name, maxsplit=1)[0].strip() for name in point_names]
    scores = [float(item.get("weakness_score") or 0) for item in top]
    error_counts = Counter(
        item.get("error_type_effective") or item.get("error_type_pred") or "uncertain"
        for item in evidence
    )
    error_phrase = "、".join(
        f"{ERROR_LABELS.get(name, name)}（{count}道）"
        for name, count in error_counts.most_common(2)
    )
    focus = "、".join(display_names)
    lead = display_names[0]
    text = (
        f"基于本次检索到的{len(evidence)}道错题及历史大模型诊断，"
        f"薄弱点主要集中在{focus}。其中“{lead}”的薄弱度最高"
        f"（{scores[0]:.0f}分）"
    )
    if error_phrase:
        text += f"，本次错题较常出现{error_phrase}"
    text += f"。建议优先回访“{lead}”相关题目，并对照订正步骤逐行核对关键条件和计算过程。"
    return {
        "text": text,
        "source": "derived_from_vlm_diagnoses",
        "evidence_count": len(evidence),
        "knowledge_points": point_names,
    }


def expand_concepts(
    db_path: str | Path, *, query: str, model: str
) -> dict:
    """Resolve curriculum aliases and return concepts already present in memory."""
    normalized = query.casefold()
    domain = next(
        (code for alias, code in DOMAIN_ALIASES.items() if alias in normalized), None
    )
    connection = connect(db_path)
    where = ["model=?"]
    params: list[object] = [model]
    if domain:
        where.append("domain_code=?")
        params.append(domain)
    rows = connection.execute(
        """SELECT domain_code,subdomain_code,knowledge_point,weakness_score
           FROM mastery_states WHERE """ + " AND ".join(where)
        + " ORDER BY weakness_score DESC,knowledge_point LIMIT 50",
        params,
    ).fetchall()
    connection.close()
    return {
        "domain_code": domain,
        "concepts": [dict(row) for row in rows],
    }


def filter_attempts(
    db_path: str | Path,
    *,
    model: str,
    domain_code: str | None = None,
    subdomain_code: str | None = None,
    errors_only: bool = False,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 100,
) -> dict:
    """Apply exact filters before any fuzzy ranking."""
    return query_vlm_memories(
        db_path,
        model=model,
        domain_code=domain_code,
        subdomain_code=subdomain_code,
        errors_only=errors_only,
        time_from=time_from,
        time_to=time_to,
        limit=max(1, min(limit, 100_000)),
    )


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[\w\u4e00-\u9fff]+", text.casefold())
        if len(token) > 1
    }


def search_memories(
    db_path: str | Path,
    *,
    model: str,
    query: str,
    candidate_attempt_ids: list[str] | None = None,
    limit: int = 20,
    embedding_client: EmbeddingClient | None = None,
    vector_weight: float = 0.65,
) -> dict:
    """Use the hybrid index when available, otherwise retain the lexical baseline."""
    if memory_index_stats(db_path, model=model)["documents"]:
        result = hybrid_search(
            db_path,
            model=model,
            query=query,
            candidate_attempt_ids=candidate_attempt_ids,
            limit=limit,
            embedding_client=embedding_client,
            vector_weight=vector_weight,
        )
        if result["results"]:
            return result
    memories = query_vlm_memories(db_path, model=model, limit=100_000)
    candidates = set(candidate_attempt_ids or [])
    query_tokens = _tokens(query)
    ranked = []
    for item in memories["attempts"]:
        if candidates and item["attempt_id"] not in candidates:
            continue
        searchable = " ".join(
            [
                item["question_transcription"], item["solution_transcription"],
                item["error_explanation"], *item["knowledge_points"],
            ]
        )
        matched = sorted(query_tokens & _tokens(searchable))
        score = len(matched) / max(1, len(query_tokens))
        ranked.append(
            {
                "attempt_id": item["attempt_id"],
                "score": round(score, 4),
                "matched_terms": matched,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["attempt_id"]))
    return {
        "query": query,
        "backend": "lexical_fallback",
        "results": ranked[: max(1, min(limit, 500))],
    }


def get_evidence(
    db_path: str | Path, *, model: str, attempt_id: str
) -> dict:
    result = query_vlm_memories(
        db_path, model=model, attempt_id=attempt_id, limit=1
    )
    if not result["attempts"]:
        raise KeyError(f"No completed {model} diagnosis for {attempt_id}")
    return result["attempts"][0]


def generate_correction(
    db_path: str | Path,
    *,
    model: str,
    attempt_id: str,
    output_dir: str | Path | None = None,
    force_render: bool = False,
) -> dict:
    """Return the versioned correction and optionally ensure its audit artifact."""
    evidence = get_evidence(db_path, model=model, attempt_id=attempt_id)
    correction = evidence.get("correction")
    if not correction:
        return {"attempt_id": attempt_id, "status": "not_available", "correction": None}
    artifact = correction.get("rendered_path")
    artifact_path = Path(artifact) if artifact else None
    metadata_path = artifact_path.with_suffix(".json") if artifact_path else None
    render_is_current = False
    if metadata_path and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            render_is_current = metadata.get("render_version") == AUDIT_RENDER_VERSION
        except (OSError, ValueError):
            render_is_current = False
    if output_dir and (
        force_render or not artifact_path or not artifact_path.exists() or not render_is_current
    ):
        output = Path(output_dir) / f"{attempt_id}_audit.png"
        artifact = render_audit_image(
            db_path,
            attempt_id=attempt_id,
            run_id=evidence["run_id"],
            output_path=output,
        )["output"]
    return {
        "attempt_id": attempt_id,
        "status": correction["verification_status"],
        "correction": correction,
        "audit_artifact": artifact,
    }


class EducationControlAgent:
    """Deterministic tool controller; an LLM planner can replace only this policy."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        model: str,
        embedding_client: EmbeddingClient | None = None,
        vector_weight: float = 0.65,
    ):
        self.db_path = Path(db_path)
        self.model = model
        self.embedding_client = embedding_client
        self.vector_weight = vector_weight

    def query(self, request: str, *, default_limit: int = 10) -> dict:
        plan = plan_memory_query(request, default_limit=default_limit)
        expansion = expand_concepts(
            self.db_path, query=request, model=self.model
        )
        time_from, time_to = resolve_recent_vlm_window(
            self.db_path,
            model=self.model,
            recent_months=plan.recent_months,
            domain_code=plan.domain_code,
        )
        filtered = filter_attempts(
            self.db_path,
            model=self.model,
            domain_code=plan.domain_code,
            errors_only=plan.errors_only,
            time_from=time_from,
            time_to=time_to,
            limit=100_000,
        )
        evidence_ids = [item["attempt_id"] for item in filtered["attempts"]]
        ranked = search_memories(
            self.db_path,
            model=self.model,
            query=request,
            candidate_attempt_ids=evidence_ids,
            limit=plan.limit,
            embedding_client=self.embedding_client,
            vector_weight=self.vector_weight,
        )
        by_id = {item["attempt_id"]: item for item in filtered["attempts"]}
        evidence = [
            by_id[item["attempt_id"]]
            for item in ranked["results"]
            if item["attempt_id"] in by_id
        ][:plan.limit]
        weaknesses = extract_vlm_weaknesses(
            self.db_path, model=self.model, domain_code=plan.domain_code
        )
        weakness_summary = summarize_weaknesses(
            weaknesses["weaknesses"], evidence
        )
        return {
            "request": request,
            "model": self.model,
            "plan": asdict(plan),
            "resolved_time_window": {
                "time_from": time_from,
                "time_to": time_to,
                "basis": "configured_memory_time_anchor",
                "is_simulated": True,
            },
            "source": "vlm_diagnosis_only",
            "tool_trace": [
                {"tool": "expand_concepts", "result_count": len(expansion["concepts"])},
                {"tool": "filter_attempts", "result_count": filtered["returned"]},
                {"tool": "search_memories", "result_count": len(ranked["results"])},
                {"tool": "get_evidence", "result_count": len(evidence)},
            ],
            "concept_expansion": expansion,
            "search_backend": ranked.get("backend", "lexical_fallback"),
            "weaknesses": weaknesses["weaknesses"],
            "weakness_summary": weakness_summary,
            "evidence": evidence,
            "teacher_review": {
                "enabled": True,
                "purpose": "Override an incorrect VLM diagnosis or correction without changing benchmark ground truth.",
                "diagnosis_endpoint_template": "/v1/errors/{attempt_id}/review",
                "correction_endpoint_template": "/v1/errors/{attempt_id}/correction",
                "review_page_template": "/review?attempt_id={attempt_id}&model=" + self.model,
            },
        }
