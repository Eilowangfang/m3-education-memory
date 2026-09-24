from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import connect, initialize
from .verification import verify_correction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_evaluation_suite(
    db_path: str | Path,
    *,
    name: str,
    limit: int = 100,
    seed: str = "fermat-math-vlm-eval-v1",
    domain_code: str | None = None,
    description: str = "",
) -> dict:
    if not name.strip():
        raise ValueError("Evaluation suite name cannot be empty")
    if limit < 1:
        raise ValueError("Evaluation suite limit must be at least 1")
    connection = connect(db_path)
    initialize(connection)
    where = "WHERE a.domain_code=?" if domain_code else ""
    params = [domain_code] if domain_code else []
    rows = connection.execute(
        f"""SELECT a.attempt_id,a.orig_q,a.domain_code,b.has_error,
                   b.reference_error_type
            FROM attempts a JOIN benchmark_truth b USING(attempt_id)
            {where}""",
        params,
    ).fetchall()

    # One deterministic variant per original problem prevents duplicated
    # FERMAT perturbations from dominating a suite.
    unique: dict[str, tuple[str, object]] = {}
    for row in rows:
        rank = hashlib.sha256(
            f"{seed}\0{row['orig_q']}\0{row['attempt_id']}".encode("utf-8")
        ).hexdigest()
        current = unique.get(row["orig_q"])
        if current is None or rank < current[0]:
            unique[row["orig_q"]] = (rank, row)
    buckets = {0: [], 1: []}
    for rank, row in unique.values():
        buckets[int(row["has_error"])].append((rank, row))
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item[0])
    selected = []
    while len(selected) < limit and (buckets[0] or buckets[1]):
        for label in (1, 0):
            if buckets[label] and len(selected) < limit:
                selected.append(buckets[label].pop(0)[1])
    attempt_ids = [row["attempt_id"] for row in selected]
    suite_id = str(uuid.uuid4())
    created_at = _now()
    try:
        with connection:
            connection.execute(
                """INSERT INTO evaluation_suites(
                     suite_id,name,description,selection_seed,attempt_ids_json,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (suite_id, name.strip(), description, seed,
                 json.dumps(attempt_ids), created_at),
            )
    finally:
        connection.close()
    return {
        "suite_id": suite_id,
        "name": name.strip(),
        "attempt_count": len(attempt_ids),
        "error_count": sum(int(row["has_error"]) for row in selected),
        "no_error_count": sum(not int(row["has_error"]) for row in selected),
        "domain_code": domain_code,
        "selection_seed": seed,
        "attempt_ids": attempt_ids,
        "created_at": created_at,
    }


def get_evaluation_suite(db_path: str | Path, *, suite: str) -> dict:
    connection = connect(db_path)
    initialize(connection)
    row = connection.execute(
        "SELECT * FROM evaluation_suites WHERE suite_id=? OR name=?", (suite, suite)
    ).fetchone()
    connection.close()
    if row is None:
        raise KeyError(f"Unknown evaluation suite: {suite}")
    result = dict(row)
    result["attempt_ids"] = json.loads(result.pop("attempt_ids_json"))
    result["attempt_count"] = len(result["attempt_ids"])
    return result


def _metrics(rows: list, *, total: int) -> dict:
    known = [row for row in rows if row["has_error_pred"] is not None]
    tp = sum(row["has_error_pred"] == 1 and row["has_error"] == 1 for row in known)
    tn = sum(row["has_error_pred"] == 0 and row["has_error"] == 0 for row in known)
    fp = sum(row["has_error_pred"] == 1 and row["has_error"] == 0 for row in known)
    fn = sum(row["has_error_pred"] == 0 and row["has_error"] == 1 for row in known)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    truth_predicted_errors = [
        row for row in known if row["has_error"] == 1 and row["has_error_pred"] == 1
    ]
    predicted_errors = [row for row in known if row["has_error_pred"] == 1]
    answer_truth = [
        row for row in truth_predicted_errors
        if row["reference_answer"] and row["predicted_final_answer"]
    ]
    equivalent_answers = [
        answers_equivalent(
            row["predicted_final_answer"], row["reference_answer"]
        )
        for row in answer_truth
    ]
    result = {
        "suite_attempts": total,
        "prediction_count": len(rows),
        "prediction_coverage": len(rows) / total if total else 0.0,
        "known_error_predictions": len(known),
        "has_error_accuracy": (tp + tn) / len(known) if known else None,
        "error_precision": precision,
        "error_recall": recall,
        "error_f1": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "error_type_exact_match": (
            sum(
                row["error_type_pred"] == row["reference_error_type"]
                for row in truth_predicted_errors
            ) / len(truth_predicted_errors)
            if truth_predicted_errors else None
        ),
        "first_error_step_coverage": (
            sum(row["first_error_step"] is not None for row in predicted_errors)
            / len(predicted_errors) if predicted_errors else None
        ),
        "error_bbox_coverage": (
            sum(bool(row["has_error_bbox"]) for row in predicted_errors)
            / len(predicted_errors) if predicted_errors else None
        ),
        "knowledge_point_coverage": (
            sum(bool(json.loads(row["knowledge_points_json"] or "[]")) for row in rows)
            / len(rows) if rows else None
        ),
        "verified_correction_rate": (
            sum(row["verification_status"] == "verified" for row in predicted_errors)
            / len(predicted_errors) if predicted_errors else None
        ),
        "review_required_rate": (
            sum(bool(row["requires_review"]) for row in rows) / len(rows)
            if rows else None
        ),
        "reference_answer_exact_match": (
            sum(
                _normalized_answer(row["predicted_final_answer"])
                == _normalized_answer(row["reference_answer"])
                for row in answer_truth
            ) / len(answer_truth) if answer_truth else None
        ),
        "reference_answer_equivalent_match": (
            sum(equivalent_answers) / len(equivalent_answers)
            if equivalent_answers else None
        ),
        "reference_answer_comparison_count": len(answer_truth),
        "ground_truth_source": "FERMAT benchmark_truth",
    }
    return result


def _normalized_answer(value: str | None) -> str:
    return "".join(str(value or "").split()).casefold()


_UNIT = re.compile(
    r"(?i)(?<=\d)\s*(?:mm|cm|km|kg|mg|ml|m|g|l|s|sec|seconds?|"
    r"meters?|metres?|degrees?|rad)(?:\^?\{?[23]\}?|[²³])?\b"
)
_LATEX_TEXT = re.compile(r"\\(?:text|mathrm)\{[^{}]*\}(?:\^?\{?[23]\}?)?")
_LATEX_FRAC = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_SQRT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")


def _answer_candidate(value: str | None) -> str:
    """Extract a conservative symbolic candidate from a displayed final answer."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("\\pi", "pi")
    text = re.sub(r"\\[,;! ]", "", text)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "").replace("$", "")
    for _ in range(8):
        updated = _LATEX_FRAC.sub(r"((\1)/(\2))", text)
        if updated == text:
            break
        text = updated
    text = _LATEX_SQRT.sub(r"sqrt(\1)", text)
    text = _LATEX_TEXT.sub("", text)
    text = re.sub(r"(?i)^\s*(?:answer|result|final answer|答案|结果)\s*[:：]?", "", text)
    # Reference answers frequently contain a derivation. The final equality
    # member is the safest available candidate without executing free-form text.
    if "=" in text:
        text = text.rsplit("=", 1)[-1]
    text = text.splitlines()[0] if text.splitlines() else text
    text = _UNIT.sub("", text)
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"(?<=\d)%", "/100", text)
    text = re.sub(r"[。；;，,]+.*$", "", text).strip()
    return "".join(text.split()).strip(".:。")


def answers_equivalent(predicted: str | None, reference: str | None) -> bool:
    """Compare final answers by exact form, then safe symbolic equivalence."""
    if _normalized_answer(predicted) == _normalized_answer(reference):
        return True
    left = _answer_candidate(predicted)
    right = _answer_candidate(reference)
    if not left or not right:
        return False
    if left.casefold() == right.casefold():
        return True
    result = verify_correction({"verification_expression": f"{left}={right}"})
    return result["status"] == "verified"


def evaluate_model(
    db_path: str | Path,
    *,
    model: str,
    provider: str | None = None,
    suite: str | None = None,
    persist: bool = False,
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    attempt_ids: list[str] | None = None
    suite_row = None
    if suite:
        suite_row = connection.execute(
            "SELECT * FROM evaluation_suites WHERE suite_id=? OR name=?", (suite, suite)
        ).fetchone()
        if suite_row is None:
            connection.close()
            raise KeyError(f"Unknown evaluation suite: {suite}")
        attempt_ids = json.loads(suite_row["attempt_ids_json"])

    params: list[object] = [model]
    provider_filter = ""
    if provider:
        provider_filter = "AND r.provider=?"
        params.append(provider)
    suite_filter = ""
    rows = []
    if attempt_ids is None or attempt_ids:
        if attempt_ids is not None:
            placeholders = ",".join("?" for _ in attempt_ids)
            suite_filter = f"AND r.attempt_id IN ({placeholders})"
            params.extend(attempt_ids)
        rows = connection.execute(
            f"""WITH ranked AS (
                   SELECT r.*,ROW_NUMBER() OVER(
                     PARTITION BY r.attempt_id ORDER BY r.created_at DESC,r.run_id DESC
                   ) AS rn
                   FROM analysis_runs r
                   WHERE r.model=? AND r.status='completed'
                     {provider_filter} {suite_filter}
                 )
                 SELECT d.attempt_id,d.has_error_pred,d.error_type_pred,
                        d.first_error_step,d.knowledge_points_json,d.requires_review,
                        b.has_error,b.reference_error_type,b.orig_a AS reference_answer,
                        CASE WHEN EXISTS(
                          SELECT 1 FROM diagnosis_steps s
                          WHERE s.run_id=r.run_id AND s.is_error=1
                            AND s.bbox_json IS NOT NULL
                        ) THEN 1 ELSE 0 END AS has_error_bbox,
                        (SELECT s.bbox_json FROM diagnosis_steps s
                         WHERE s.run_id=r.run_id AND s.bbox_json IS NOT NULL
                         ORDER BY CASE
                           WHEN s.step_index=d.first_error_step THEN 0
                           WHEN s.is_error=1 THEN 1 ELSE 2 END,s.step_index
                         LIMIT 1) AS predicted_error_bbox_json,
                        c.verification_status,c.final_answer AS predicted_final_answer,
                        r.provider,r.prompt_version
                 FROM ranked r
                 JOIN diagnoses d ON d.run_id=r.run_id
                 JOIN benchmark_truth b ON b.attempt_id=d.attempt_id
                 LEFT JOIN corrections c ON c.run_id=r.run_id
                 WHERE r.rn=1""",
            params,
        ).fetchall()
    total = len(attempt_ids) if attempt_ids is not None else len(rows)
    metrics = _metrics(rows, total=total)
    providers = sorted({row["provider"] for row in rows})
    prompt_versions = sorted({row["prompt_version"] for row in rows})
    result = {
        "model": model,
        "provider": (
            provider if provider else
            providers[0] if len(providers) == 1 else
            providers if providers else None
        ),
        "prompt_version": (
            prompt_versions[0] if len(prompt_versions) == 1 else
            prompt_versions if prompt_versions else None
        ),
        "suite_id": suite_row["suite_id"] if suite_row else None,
        "suite_name": suite_row["name"] if suite_row else None,
        "metrics": metrics,
        "metric_notes": {
            "ground_truth": "All evaluation truth comes directly from FERMAT benchmark_truth; teacher review is never joined.",
            "error_type_exact_match": "Uses the taxonomy mapped from FERMAT pert_reasoning.",
            "reference_answer_exact_match": "Strict normalized comparison against FERMAT orig_a; formatting differences can lower this metric.",
            "reference_answer_equivalent_match": "Conservative safe symbolic comparison after normalizing common LaTeX, units, and terminal equality members.",
            "coverage_metrics": "FERMAT has no step, bbox, or knowledge-point labels, so these are coverage metrics rather than accuracy metrics.",
        },
    }
    if persist:
        if suite_row is None:
            connection.close()
            raise ValueError("Persisted evaluation requires a named suite")
        if not rows:
            connection.close()
            raise ValueError("Cannot persist an evaluation without predictions")
        if len(providers) != 1 or len(prompt_versions) != 1:
            connection.close()
            raise ValueError(
                "Persisted evaluation must resolve to one provider and prompt version"
            )
        evaluation_run_id = str(uuid.uuid4())
        created_at = _now()
        with connection:
            connection.execute(
                """INSERT INTO evaluation_runs(
                     evaluation_run_id,suite_id,provider,model,prompt_version,
                     evaluated_attempt_count,metrics_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (evaluation_run_id, suite_row["suite_id"], str(result["provider"]),
                 model, str(result["prompt_version"]), metrics["prediction_count"],
                 json.dumps(metrics, ensure_ascii=False), created_at),
            )
        result["evaluation_run_id"] = evaluation_run_id
        result["created_at"] = created_at
    connection.close()
    return result


def compare_evaluation_runs(db_path: str | Path, *, suite: str) -> dict:
    connection = connect(db_path)
    initialize(connection)
    suite_row = connection.execute(
        "SELECT * FROM evaluation_suites WHERE suite_id=? OR name=?", (suite, suite)
    ).fetchone()
    if suite_row is None:
        connection.close()
        raise KeyError(f"Unknown evaluation suite: {suite}")
    rows = connection.execute(
        """WITH ranked AS (
             SELECT e.*,ROW_NUMBER() OVER(
               PARTITION BY e.provider,e.model,e.prompt_version
               ORDER BY e.created_at DESC,e.evaluation_run_id DESC
             ) AS rn
             FROM evaluation_runs e WHERE e.suite_id=?
           )
           SELECT * FROM ranked WHERE rn=1 ORDER BY model,provider""",
        (suite_row["suite_id"],),
    ).fetchall()
    connection.close()
    return {
        "suite_id": suite_row["suite_id"],
        "suite_name": suite_row["name"],
        "suite_attempt_count": len(json.loads(suite_row["attempt_ids_json"])),
        "models": [
            {
                "evaluation_run_id": row["evaluation_run_id"],
                "provider": row["provider"],
                "model": row["model"],
                "prompt_version": row["prompt_version"],
                "created_at": row["created_at"],
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in rows
        ],
    }
