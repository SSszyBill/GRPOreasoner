"""Countdown scoring: expression uses each given number exactly once and equals target.
Accepts LaTeX-form answers (\times, \cdot, \div, \left, \right, ...)
and equation-form answers like \boxed{24 = 3 \times 8}.
"""
from __future__ import annotations
import ast
import re
from grpo_reasoner.scoring._boxed import last_boxed
_BS = chr(92)
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)
def extract_answer(completion: str) -> str | None:
    """Return the last \boxed{} content, or None. (Raw expression, not evaluated.)"""
    return last_boxed(completion)
def score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None,
    format_score: float,
    score: float,
    ) -> float:
    """Format = \boxed{} present. Correctness = expression parses,
    uses each number in extra_info['numbers'] exactly once, and equals target.
    """
    boxed = last_boxed(solution_str)
    format_reward = format_score if boxed is not None else 0.0
    if boxed is None or not isinstance(extra_info, dict):
        return 0.0 + format_reward
    numbers = extra_info.get("numbers")
    if not numbers:
        return 0.0 + format_reward
    try:
        target = float(int(ground_truth))
    except Exception:
        return 0.0 + format_reward
    required_numbers_sorted = sorted(int(x) for x in numbers)
    cleaned = _clean_latex(boxed)
    
    for segment in cleaned.split("="):
        expr = segment.strip()
        if not expr or not re.fullmatch(r"[0-9+*/() -]+", expr):
            continue
        used = sorted(int(token) for token in re.findall(r"[0-9]+", expr))
        if used != required_numbers_sorted:
            continue
        try:
            value = _safe_eval(expr)
        except Exception:
            continue
        if abs(float(value) - target) < 1e-6:
            return score + format_reward
    return 0.0 + format_reward

def _clean_latex(text: str) -> str:
    for junk in (_BS + "left", _BS + "right", _BS + "!", _BS + ",", _BS + ";"):
        text = text.replace(junk, "")
    text = text.replace(_BS + "times", "*").replace(_BS + "cdot", "*").replace(_BS + "div", "/")
    text = text.replace("{", "(").replace("}", ")")
    return text
def _safe_eval(expr: str):
    node = ast.parse(expr, mode="eval")
    for child in ast.walk(node):
            if not isinstance(child, _ALLOWED_NODES):
                raise ValueError("illegal node: " + type(child).name)
    return eval(compile(node, "<expr>", "eval"))
