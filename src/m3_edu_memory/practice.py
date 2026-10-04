"""Generate reproducible practice variants from diagnosed error memories."""

from __future__ import annotations

import hashlib
import random
import re


_CHINESE_COUNTS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                   "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_SUPPORTED_POINT_TERMS = (
    "链式法则", "对数求导", "分部积分", "不定积分", "二阶导数",
    "条件概率", "古典概型", "矩阵乘法", "矩阵加法", "特殊角",
    "三角函数值", "正切和角", "和角公式", "子集", "乘法原理",
    "不等式", "并集", "距离", "斜率", "一元一次方程", "解方程",
)


def requested_count(request: str) -> int:
    match = re.search(r"(?:再出|生成|出|练习|做)\s*(\d+|[一二三四五六七八九十])\s*(?:道|题)", request)
    if not match:
        match = re.search(r"(\d+|[一二三四五六七八九十])\s*(?:道|题)", request)
    count = int(match.group(1)) if match and match.group(1).isdigit() else (
        _CHINESE_COUNTS[match.group(1)] if match else 5
    )
    if not 1 <= count <= 10:
        raise ValueError("一次可生成 1–10 道变式练习题")
    return count


def _template(point: str, domain: str) -> str:
    if "链式法则" in point:
        return "chain_rule"
    if "对数求导" in point:
        return "log_derivative"
    if "分部积分" in point:
        return "integration_by_parts"
    if "不定积分" in point:
        return "integral"
    if "二阶导数" in point:
        return "second_derivative"
    if "条件概率" in point:
        return "conditional_probability"
    if "古典概型" in point:
        return "dice_probability"
    if "矩阵乘法" in point:
        return "matrix_product"
    if "矩阵加法" in point:
        return "matrix_sum"
    if "特殊角" in point or "三角函数值" in point:
        return "special_angle"
    if "正切和角" in point or "和角公式" in point:
        return "tangent_sum"
    if "子集" in point:
        return "subsets"
    if "乘法原理" in point:
        return "counting"
    if "不等式" in point:
        return "inequality"
    if "并集" in point:
        return "set_union"
    if "点到直线" in point and "距离" in point:
        return "point_line_distance"
    if "空间两点" in point and "距离" in point:
        return "distance_3d"
    if "距离" in point:
        return "distance"
    if "斜率" in point:
        return "slope"
    if "一元一次方程" in point or "解方程" in point:
        return "linear_equation"
    return {
        "clc": "chain_rule", "alg": "linear_equation", "art": "arithmetic",
        "mgm": "distance", "pst": "conditional_probability",
        "trg": "special_angle", "apt": "counting",
    }.get(domain, "linear_equation")


def _make_question(template: str, rng: random.Random, index: int) -> str:
    a, b, c = rng.randint(2, 7), rng.randint(1, 6), rng.randint(1, 8)
    if template == "chain_rule":
        n = rng.randint(3, 6)
        linear_term = "x" if b == 1 else f"{b}x"
        return rf"求函数 \(y=({a}x^2+{linear_term}+{c})^{n}\) 的导数，并写出内层函数的导数。"
    if template == "log_derivative":
        return rf"求函数 \(y=\ln({a}x^2+{c})\) 的导数，并注明定义域。"
    if template == "integration_by_parts":
        return rf"计算不定积分 \(\int x e^{{{a}x}}\,dx\)，写出分部积分的选择。"
    if template == "integral":
        return rf"计算不定积分 \(\int ({a}x+{b})^{c}\,dx\)，写出换元与积分常数。"
    if template == "second_derivative":
        return rf"已知 \(f(x)={a}x^3+{b}\sin x\)，求 \(f''(x)\)。"
    if template == "conditional_probability":
        red, blue, taken = a + 2, b + 2, 2
        return f"袋中有{red}个红球和{blue}个蓝球，不放回抽取{taken}个。已知第一个是红球，求第二个也是红球的概率。"
    if template == "dice_probability":
        threshold = rng.randint(7, 10)
        return f"同时掷两枚公平的六面骰子，求点数之和不小于{threshold}的概率；写出样本空间大小。"
    if template == "matrix_product":
        return rf"设 \(A=\begin{{pmatrix}}{a}&{b}\\1&{c}\end{{pmatrix}}\)，\(B=\begin{{pmatrix}}2&1\\{b}&3\end{{pmatrix}}\)，求 \(AB\)。"
    if template == "matrix_sum":
        return rf"设 \(A=\begin{{pmatrix}}{a}&-{b}\\1&{c}\end{{pmatrix}}\)，\(B=\begin{{pmatrix}}2&3\\-{c}&4\end{{pmatrix}}\)，求 \(A+B\)。"
    if template == "special_angle":
        angle = rng.choice([30, 45, 60])
        return rf"计算 \(2\sin {angle}^\circ\cos {angle}^\circ+\cos^2 {angle}^\circ\) 的精确值。"
    if template == "tangent_sum":
        return rf"已知锐角 \(\alpha,\beta\) 满足 \(\tan\alpha=1/{a}\)、\(\tan\beta=1/{b}\)，求 \(\tan(\alpha+\beta)\)。"
    if template == "subsets":
        n = rng.randint(4, 7)
        return f"集合 A 有{n}个互不相同的元素。求 A 的子集个数和非空真子集个数，并说明两者的区别。"
    if template == "counting":
        return f"一家店有{a}种主食、{b}种饮品和{c}种甜点。各选一种组成套餐，共有多少种不同选择？写出所用计数原理。"
    if template == "inequality":
        lower, upper = -a, b + c + a
        return rf"解复合不等式 \({lower}<{a}x+{b}\leq {upper}\)，用区间表示解集。"
    if template == "set_union":
        return rf"设 \(A=\{{1,2,{a + 3},{b + 9}\}}\)，\(B=\{{2,3,{a + 3},{c + 15}\}}\)，求 \(A\cup B\) 与 \(A\cap B\)。"
    if template == "distance":
        return rf"平面上两点 \(P({a},-{b})\)、\(Q(-{c},{a + b})\)，求线段 \(PQ\) 的长度。"
    if template == "distance_3d":
        return rf"空间中两点 \(P({a},-{b},{c})\)、\(Q(-{c},{a + b},-{b})\)，求 \(PQ\) 的长度。"
    if template == "point_line_distance":
        return rf"求点 \(P({a},-{b})\) 到直线 \({a}x+{b}y-{c}=0\) 的距离，并写出所用公式。"
    if template == "slope":
        return rf"经过 \(P({a},{b})\)、\(Q({a + c},{b + a})\) 的直线斜率是多少？写出计算过程。"
    if template == "arithmetic":
        return rf"计算 \(({a}+{b})\times {c}-{a}({b}-{c})\)，逐步写出运算顺序。"
    if c == a:
        c = a + 1
    solution = rng.randint(-4, 7)
    d = a * (solution + b) - c * solution
    constant = f"+{d}" if d >= 0 else str(d)
    return rf"解方程 \({a}(x+{b})={c}x{constant}\)，并将结果代回原方程检验。"


def generate_practice(report: dict, *, request: str) -> dict:
    """Use diagnosed error types and source attempts to select fresh variants."""
    count = requested_count(request)
    cases = report.get("cases") or []
    categories = report.get("categories") or []
    if not cases or not categories:
        raise ValueError("当前分析范围没有可用的已诊断错题，无法生成针对性练习")
    by_type: dict[str, list[dict]] = {}
    for case in cases:
        by_type.setdefault(case["error_type"], []).append(case)
    candidates = []
    for category in categories:
        members = by_type.get(category["error_type"], [])
        if not members:
            continue
        chosen = None
        for point in category.get("top_knowledge_points", []):
            if not any(term in point["name"] for term in _SUPPORTED_POINT_TERMS):
                continue
            for case in members:
                if point["name"] in case["knowledge_points"]:
                    chosen = (category, case, point["name"], point["error_count"])
                    break
            if chosen:
                break
        if not chosen:
            first = members[0]
            domain_count = sum(case["domain_code"] == first["domain_code"] for case in members)
            chosen = (category, first, "", domain_count)
        candidates.append(chosen)
    if not candidates:
        raise ValueError("当前错题缺少可用的错误类型证据")
    items = []
    for index in range(count):
        category, case, point, point_count = candidates[index % len(candidates)]
        seed = hashlib.sha256(
            f"{report['request']}|{request}|{case['attempt_id']}|{index}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(seed[:8], "big"))
        template = _template(point, case["domain_code"])
        question = _make_question(template, rng, index)
        items.append({
            "number": index + 1,
            "question": question,
            "domain": case["domain_label"],
            "error_type": category["label"],
            "knowledge_point": point or case["domain_label"],
            "related_error_count": point_count,
            "source_attempt_id": case["attempt_id"],
        })
    return {
        "analysis_request": report["request"],
        "request": request,
        "count": count,
        "generation_method": "evidence_selected_parameterized_variants",
        "questions": items,
    }
