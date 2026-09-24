from __future__ import annotations

import ast
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any


_ARITHMETIC = re.compile(r"^[0-9+\-*/^().=\s]+$")
_SYMBOLIC = re.compile(r"^[A-Za-z0-9_+\-*/^().,=\[\]\s]+$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
_MAX_EXPRESSION_LENGTH = 500
_MAX_AST_NODES = 200


def _fraction_eval(expression: str) -> Fraction:
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right.numerator) <= 20:
                return left ** right.numerator
        raise ValueError("unsupported arithmetic expression")

    return visit(tree)


def _matrix_eval(expression: str):
    """Evaluate a small exact matrix grammar using Fraction arithmetic."""
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def is_matrix(value) -> bool:
        return (
            isinstance(value, tuple) and value
            and all(isinstance(row, tuple) and row for row in value)
        )

    def matrix_binary(left, right, operation):
        if not is_matrix(left) or not is_matrix(right):
            raise ValueError("matrix operation requires two matrices")
        if operation in (ast.Add, ast.Sub):
            if len(left) != len(right) or len(left[0]) != len(right[0]):
                raise ValueError("matrix dimensions do not match")
            return tuple(
                tuple(
                    (a + b if operation is ast.Add else a - b)
                    for a, b in zip(left_row, right_row)
                )
                for left_row, right_row in zip(left, right)
            )
        if operation is ast.Mult:
            if len(left[0]) != len(right):
                raise ValueError("matrix multiplication dimensions do not match")
            return tuple(
                tuple(
                    sum((left[i][k] * right[k][j] for k in range(len(right))), Fraction(0))
                    for j in range(len(right[0]))
                )
                for i in range(len(left))
            )
        raise ValueError("unsupported matrix operation")

    def visit(node: ast.AST):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if is_matrix(value):
                return (
                    value if isinstance(node.op, ast.UAdd)
                    else tuple(tuple(-item for item in row) for row in value)
                )
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, (ast.List, ast.Tuple)):
            return tuple(visit(item) for item in node.elts)
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "matrix" and len(node.args) == 1 and not node.keywords
        ):
            rows = visit(node.args[0])
            if (
                not isinstance(rows, tuple) or not rows
                or any(not isinstance(row, tuple) or not row for row in rows)
                or len({len(row) for row in rows}) != 1
                or len(rows) > 20 or len(rows[0]) > 20
                or any(not isinstance(item, Fraction) for row in rows for item in row)
            ):
                raise ValueError("matrix requires a rectangular numeric list up to 20x20")
            return rows
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if is_matrix(left) or is_matrix(right):
                return matrix_binary(left, right, type(node.op))
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported exact matrix expression")

    return visit(tree)


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = depth = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parentheses")
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ValueError("unbalanced parentheses")
    parts.append(value[start:].strip())
    return parts


def _sympy_expression(expression: str):
    import sympy

    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("symbolic expression is too long")
    if not _SYMBOLIC.fullmatch(expression):
        raise ValueError("symbolic expression contains unsupported characters")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("symbolic expression is too complex")
    functions = {
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt,
        "abs": sympy.Abs,
    }

    def visit(node: ast.AST):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return sympy.Rational(str(node.value))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(item) for item in node.elts]
        if isinstance(node, ast.Name):
            if not _NAME.fullmatch(node.id) or node.id in functions:
                raise ValueError(f"unsupported symbol: {node.id}")
            return sympy.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.is_Integer and abs(int(right)) > 20:
                    raise ValueError("symbolic exponent exceeds safety limit")
                return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "matrix":
                if len(node.args) != 1 or node.keywords:
                    raise ValueError("matrix requires one nested-list argument")
                values = visit(node.args[0])
                if (
                    not isinstance(values, list) or not values
                    or any(not isinstance(row, list) or not row for row in values)
                    or len({len(row) for row in values}) != 1
                    or len(values) > 20 or len(values[0]) > 20
                ):
                    raise ValueError("matrix requires a rectangular list up to 20x20")
                return sympy.Matrix(values)
            function = functions.get(node.func.id)
            if function is None or len(node.args) != 1 or node.keywords:
                raise ValueError(f"unsupported function call: {node.func.id}")
            return function(visit(node.args[0]))
        raise ValueError("unsupported symbolic syntax")

    return visit(tree)


def _rational_probe(expression: str, values: dict[str, Fraction], variable: str | None = None):
    """Evaluate an expression and its exact first derivative using dual numbers."""
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def visit(node: ast.AST) -> tuple[Fraction, Fraction]:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value)), Fraction(0)
        if isinstance(node, ast.Name) and _NAME.fullmatch(node.id):
            return values[node.id], Fraction(int(node.id == variable))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value, derivative = visit(node.operand)
            return (value, derivative) if isinstance(node.op, ast.UAdd) else (-value, -derivative)
        if isinstance(node, ast.BinOp):
            left, left_d = visit(node.left)
            right, right_d = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right, left_d + right_d
            if isinstance(node.op, ast.Sub):
                return left - right, left_d - right_d
            if isinstance(node.op, ast.Mult):
                return left * right, left_d * right + left * right_d
            if isinstance(node.op, ast.Div):
                return left / right, (left_d * right - left * right_d) / (right * right)
            if isinstance(node.op, ast.Pow) and right_d == 0 and right.denominator == 1 and abs(right.numerator) <= 20:
                exponent = right.numerator
                return left ** exponent, exponent * (left ** (exponent - 1)) * left_d
        raise ValueError("unsupported expression for built-in symbolic verifier")

    return visit(tree)


def _probe_environments(expressions: list[str]) -> list[dict[str, Fraction]]:
    names: set[str] = set()
    for expression in expressions:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
        names.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
    if not names or any(not _NAME.fullmatch(name) for name in names):
        raise ValueError("unsupported symbols for built-in verifier")
    ordered = sorted(names)
    return [
        {name: Fraction(seed + index) for index, name in enumerate(ordered)}
        for seed in (2, 5, 11, 17, 23)
    ]


def _verify_without_sympy(expression: str) -> dict[str, Any]:
    if "matrix(" in expression:
        parts = [part.strip() for part in expression.split("=") if part.strip()]
        if len(parts) < 2:
            raise ValueError("matrix verification requires an equality")
        values = [_matrix_eval(part) for part in parts]
        verified = all(value == values[0] for value in values[1:])
        return {
            "status": "verified" if verified else "inconsistent",
            "method": "fraction_matrix_exact",
            "details": {
                "shape": [len(values[0]), len(values[0][0])],
                "members": len(values),
            },
        }
    for operation, method in (("diff", "rational_probe_derivative"), ("integrate", "rational_probe_antiderivative")):
        prefix = operation + "("
        if not expression.startswith(prefix):
            continue
        depth = 0
        close_index = None
        for index, character in enumerate(expression[len(operation):], start=len(operation)):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        suffix = expression[close_index + 1:].strip() if close_index is not None else ""
        if close_index is None or not suffix.startswith("="):
            raise ValueError(f"{operation} verification requires '= result'")
        arguments = _split_top_level(expression[len(prefix):close_index])
        if len(arguments) != 2 or not _NAME.fullmatch(arguments[1]):
            raise ValueError(f"{operation} requires expression and variable")
        source, variable, claimed = arguments[0], arguments[1], suffix[1:].strip()
        matches = 0
        for environment in _probe_environments([source, claimed]):
            try:
                source_value, source_d = _rational_probe(source, environment, variable)
                claimed_value, claimed_d = _rational_probe(claimed, environment, variable)
            except ZeroDivisionError:
                continue
            valid = source_d == claimed_value if operation == "diff" else claimed_d == source_value
            if not valid:
                return {"status": "inconsistent", "method": method, "details": {"probes": matches + 1}}
            matches += 1
        if matches < 3:
            raise ValueError("not enough valid exact probes")
        return {"status": "verified", "method": method, "details": {"probes": matches}}

    parts = [part.strip() for part in expression.split("=") if part.strip()]
    if len(parts) < 2:
        raise ValueError("an equality with at least two sides is required")
    matches = 0
    for environment in _probe_environments(parts):
        try:
            values = [_rational_probe(part, environment)[0] for part in parts]
        except ZeroDivisionError:
            continue
        if any(value != values[0] for value in values[1:]):
            return {"status": "inconsistent", "method": "rational_probe_identity", "details": {"probes": matches + 1}}
        matches += 1
    if matches < 3:
        raise ValueError("not enough valid exact probes")
    return {"status": "verified", "method": "rational_probe_identity", "details": {"probes": matches}}


def _verify_calculus(expression: str) -> dict[str, Any] | None:
    import sympy

    if expression.startswith("limit("):
        operation = "limit"
        depth = 0
        close_index = None
        for index, character in enumerate(expression[len(operation):], start=len(operation)):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        suffix = expression[close_index + 1:].strip() if close_index is not None else ""
        if close_index is None or not suffix.startswith("="):
            raise ValueError("limit verification requires '= result'")
        arguments = _split_top_level(expression[len("limit("):close_index])
        if len(arguments) != 3 or not _NAME.fullmatch(arguments[1]):
            raise ValueError("limit requires expression, variable and approach point")
        source = _sympy_expression(arguments[0])
        variable = sympy.Symbol(arguments[1])
        point = _sympy_expression(arguments[2])
        claimed = _sympy_expression(suffix[1:].strip())
        if point.free_symbols or claimed.free_symbols:
            raise ValueError("limit point and result must be constant")
        if source.free_symbols - {variable}:
            raise ValueError("limit expression contains unrelated symbols")
        actual = sympy.limit(source, variable, point)
        residual = sympy.simplify(actual - claimed)
        return {
            "status": "verified" if residual == 0 else "inconsistent",
            "method": "sympy_limit",
            "details": {
                "actual": str(actual),
                "claimed": str(claimed),
                "residual": str(residual),
                "variable": str(variable),
                "point": str(point),
            },
        }

    for operation, method in (("diff", "sympy_derivative"), ("integrate", "sympy_antiderivative")):
        prefix = operation + "("
        if not expression.startswith(prefix):
            continue
        depth = 0
        close_index = None
        for index, character in enumerate(expression[len(operation):], start=len(operation)):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        suffix = expression[close_index + 1:].strip() if close_index is not None else ""
        if close_index is None or not suffix.startswith("="):
            raise ValueError(f"{operation} verification requires '= result'")
        arguments = _split_top_level(expression[len(prefix):close_index])
        if len(arguments) != 2 or not _NAME.fullmatch(arguments[1]):
            raise ValueError(f"{operation} requires expression and variable")
        source = _sympy_expression(arguments[0])
        variable = sympy.Symbol(arguments[1])
        claimed = _sympy_expression(suffix[1:].strip())
        residual = (
            sympy.simplify(sympy.diff(source, variable) - claimed)
            if operation == "diff"
            else sympy.simplify(sympy.diff(claimed, variable) - source)
        )
        return {
            "status": "verified" if residual == 0 else "inconsistent",
            "method": method,
            "details": {"residual": str(residual), "variable": str(variable)},
        }
    return None


def verify_correction(correction: dict[str, Any] | None) -> dict[str, Any]:
    """Verify numeric, symbolic, derivative, or antiderivative claims without benchmark truth."""
    if not correction:
        return {"status": "not_applicable", "method": "none", "details": {"reason": "no correction supplied"}}
    expression = (correction.get("verification_expression") or "").strip()
    if not expression:
        return {"status": "unverifiable", "method": "none", "details": {"reason": "verification_expression is empty"}}
    if len(expression) > _MAX_EXPRESSION_LENGTH or not _SYMBOLIC.fullmatch(expression):
        return {"status": "unverifiable", "method": "symbolic_allowlist", "details": {"reason": "expression is outside the safe verifier grammar"}}
    try:
        claims = _split_top_level(expression)
        if len(claims) > 1:
            results = [
                verify_correction({"verification_expression": claim})
                for claim in claims
            ]
            statuses = [result["status"] for result in results]
            status = (
                "inconsistent" if "inconsistent" in statuses
                else "unverifiable" if "unverifiable" in statuses
                else "verified"
            )
            return {
                "status": status,
                "method": "compound_claims",
                "details": {"claims": results},
            }
        try:
            import sympy

            if not hasattr(sympy, "simplify"):
                raise ImportError("SymPy runtime is not readable in this process")
            calculus = _verify_calculus(expression)
            if calculus is not None:
                return calculus
            parts = [part.strip() for part in expression.split("=") if part.strip()]
            if len(parts) < 2:
                raise ValueError("an equality with at least two sides is required")
            values = [_sympy_expression(part) for part in parts]
            residuals = [sympy.simplify(value - values[0]) for value in values[1:]]
            verified = all(
                residual == 0 or bool(getattr(residual, "is_zero_matrix", False))
                for residual in residuals
            )
            symbolic = any(value.free_symbols for value in values)
            return {
                "status": "verified" if verified else "inconsistent",
                "method": "sympy_symbolic_identity" if symbolic else "sympy_exact",
                "details": {
                    "normalized_values": [str(value) for value in values],
                    "residuals": [str(value) for value in residuals],
                },
            }
        except ImportError:
            if not _ARITHMETIC.fullmatch(expression):
                return _verify_without_sympy(expression)
            parts = [part.strip() for part in expression.split("=") if part.strip()]
            if len(parts) < 2:
                raise ValueError("an equality with at least two sides is required")
            values = [_fraction_eval(part) for part in parts]
            verified = all(value == values[0] for value in values[1:])
            return {
                "status": "verified" if verified else "inconsistent",
                "method": "fraction_ast_exact",
                "details": {"normalized_values": [str(value) for value in values]},
            }
    except Exception as exc:
        return {"status": "unverifiable", "method": "safe_symbolic", "details": {"reason": str(exc)}}


def reverify_corrections(
    db_path: str | Path, *, model: str | None = None
) -> dict[str, Any]:
    """Recompute stored verification results without making model API calls."""
    from .database import connect, initialize

    connection = connect(db_path)
    initialize(connection)
    targets = []
    correction_sql = """SELECT 'corrections' AS source_table,
                       c.correction_id AS row_id,c.verification_expression
                       FROM corrections c
                       JOIN analysis_runs r ON r.run_id=c.run_id
                       WHERE COALESCE(c.verification_expression,'')<>''"""
    correction_params: tuple[str, ...] = ()
    if model:
        correction_sql += " AND r.model=?"
        correction_params = (model,)
    targets.extend(connection.execute(correction_sql, correction_params).fetchall())
    version_sql = """SELECT 'correction_versions' AS source_table,
                    version_id AS row_id,verification_expression
                    FROM correction_versions
                    WHERE COALESCE(verification_expression,'')<>''"""
    version_params: tuple[str, ...] = ()
    if model:
        version_sql += " AND model=?"
        version_params = (model,)
    targets.extend(connection.execute(version_sql, version_params).fetchall())
    counts = {"verified": 0, "inconsistent": 0, "unverifiable": 0}
    with connection:
        for target in targets:
            result = verify_correction({
                "verification_expression": target["verification_expression"]
            })
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            id_column = (
                "correction_id"
                if target["source_table"] == "corrections"
                else "version_id"
            )
            connection.execute(
                f"""UPDATE {target['source_table']}
                    SET verification_status=?,verification_method=?,
                        verification_details_json=?
                    WHERE {id_column}=?""",
                (
                    result["status"], result["method"],
                    json.dumps(result["details"], ensure_ascii=False),
                    target["row_id"],
                ),
            )
    connection.close()
    return {"model": model, "updated_rows": len(targets), **counts}
