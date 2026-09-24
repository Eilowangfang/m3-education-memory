from __future__ import annotations

import re
import unicodedata


# A deliberately small, auditable curriculum alias layer. Raw VLM labels remain
# stored in diagnoses; this map is used only for aggregation and retrieval.
_ALIASES: dict[str, tuple[str, ...]] = {
    "链式法则": ("链式法则", "复合函数求导", "chain rule", "derivative of composite"),
    "乘积求导法则": ("乘积求导", "乘法法则", "product rule"),
    "商的求导法则": ("商的求导", "商法则", "quotient rule"),
    "幂函数求导": ("幂函数求导", "power rule"),
    "分部积分": ("分部积分", "integration by parts"),
    "换元积分": ("换元积分", "代换积分", "u substitution", "substitution rule"),
    "指数运算法则": ("指数运算", "指数定律", "负整数指数", "laws of exponents"),
    "多项式展开": ("多项式展开", "展开多项式", "polynomial expansion"),
    "因式分解": ("因式分解", "factorisation", "factorization"),
    "一元方程求解": ("一元方程", "解方程", "solve equation"),
    "矩阵乘法": ("矩阵乘法", "matrix multiplication", "product of matrices"),
    "勾股定理": ("勾股定理", "pythagorean theorem"),
    "三角形相似": ("三角形相似", "相似三角形", "similar triangles"),
    "圆的面积": ("圆的面积", "area of circle", "area of a circle"),
    "三角恒等式": ("三角恒等式", "trigonometric identity", "trigonometric identities"),
    "正弦定理": ("正弦定理", "sine rule", "law of sines"),
    "余弦定理": ("余弦定理", "cosine rule", "law of cosines"),
    "基础概率": ("基础概率", "概率计算", "basic probability"),
    "排列组合": ("排列组合", "permutation and combination", "combinatorics"),
    "均值与中位数": ("均值与中位数", "mean and median", "平均数与中位数"),
}


def _normalized(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = re.sub(r"^(?:知识点|knowledge\s*points?)\s*[:：-]?\s*", "", text)
    text = re.sub(r"[\s_\-—–/、，,;；:：()（）]+", " ", text)
    return text.strip(" .。")


def canonicalize_knowledge_point(
    value: str, *, domain_code: str | None = None, subdomain_code: str | None = None
) -> str:
    """Return a stable curriculum label while preserving unknown useful labels."""
    original = unicodedata.normalize("NFKC", str(value)).strip()
    normalized = _normalized(original)
    if not normalized:
        return subdomain_code or "unknown"
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            token = _normalized(alias)
            if normalized == token or (
                len(token) >= 4 and token in normalized
            ):
                return canonical
    return re.sub(r"\s+", " ", original).strip(" .。")


def canonicalize_knowledge_points(
    values: list[object], *, domain_code: str | None = None,
    subdomain_code: str | None = None,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        point = canonicalize_knowledge_point(
            str(value), domain_code=domain_code, subdomain_code=subdomain_code
        )
        key = point.casefold()
        if point and key not in seen:
            seen.add(key)
            result.append(point)
    return result or [subdomain_code or "unknown"]
