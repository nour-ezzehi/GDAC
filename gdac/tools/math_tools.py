"""Safe Mathematical Expression Evaluator for GDAC Agents."""

from __future__ import annotations

from typing import Any, Dict


def calculate_expression(expression: str) -> Dict[str, Any]:
    """Safely evaluates an arithmetic or Python mathematical expression.
    
    Permits basic math operations and safe builtins: round, abs, min, max, sum, pow, float, int, len.
    Prevents execution of arbitrary code or unauthorized identifiers via AST symbol validation.
    """
    clean_expr = expression.strip().strip("`").strip()
    import math

    safe_math_attrs = {
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_")
    }
    safe_env = {
        "__builtins__": None,
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "float": float,
        "int": int,
        "len": len,
        "math": math,
        **safe_math_attrs,
    }
    try:
        code_obj = compile(clean_expr, "<calc>", "eval")
        for name in code_obj.co_names:
            if name not in safe_env:
                return {
                    "tool": "calculate_expression",
                    "expression": clean_expr,
                    "error": f"Unauthorized identifier '{name}'. Permitted: arithmetic operators (+, -, *, /, //, %, **), builtins (round, abs, min, max, sum, pow, float, int, len), and standard math library functions (sqrt, ceil, floor, log, exp, sin, cos, pi, etc.).",
                }
        result = eval(code_obj, safe_env)
        return {
            "tool": "calculate_expression",
            "expression": clean_expr,
            "result": result,
        }
    except Exception as exc:
        return {
            "tool": "calculate_expression",
            "expression": clean_expr,
            "error": f"Evaluation error: {str(exc)}",
        }
