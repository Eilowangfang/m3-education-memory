from __future__ import annotations


ERROR_TYPES = (
    "conceptual",
    "assumption",
    "algebraic_manipulation",
    "arithmetic",
    "notation",
    "transcription",
    "omitted_step",
    "presentation",
    "no_actual_error",
    "uncertain",
)


def classify_reference_reason(reason: str | None, has_error: bool) -> str:
    """Map FERMAT reference prose into the first education taxonomy version."""
    if not has_error:
        return "no_actual_error"
    text = (reason or "").lower()
    rules = (
        ("transcription", ("transcri", "copied", "copying", "misread")),
        ("notation", ("notation", "symbol", "sign error", "unit")),
        ("omitted_step", ("omit", "missing", "skip", "forgot")),
        ("assumption", ("assum", "condition", "domain", "constraint")),
        ("arithmetic", ("arithmetic", "calculation", "comput", "numeric")),
        ("algebraic_manipulation", ("algebra", "factor", "expand", "simplif")),
        ("presentation", ("present", "unclear", "incomplete explanation")),
        ("conceptual", ("concept", "misunderstand", "incorrect formula", "wrong formula")),
    )
    for error_type, needles in rules:
        if any(needle in text for needle in needles):
            return error_type
    return "uncertain"
