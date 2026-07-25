# Multi-source reward for verl GRPO. Routes by data_source.
# Sources: openai/gsm8k (numeric ####), numina_math and math500 (boxed + MATH grader), countdown (expression).
# Total = correctness*score + format*format_score  (decoupled sum). Defaults: format_score=0.1, score=1.0.

import re
import ast

BS = chr(92)
BOXED = BS + "boxed"


def _last_boxed(text):
    """Return the content inside the LAST boxed{...}, or None."""
    idx = text.rfind(BOXED)
    if idx < 0:
        return None
    i = idx + len(BOXED)
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    i += 1
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def _has_boxed(text):
    return _last_boxed(text) is not None


# ---------------- GSM8K (preserves user's original logic) ----------------
def gsm8k_extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        solutions = re.findall("#### (-?[0-9.,]+)", solution_str)
        final_answer = None if len(solutions) == 0 else solutions[-1].replace(",", "").replace("$", "")
    else:
        answer = re.findall("(-?[0-9.,]+)", solution_str)
        final_answer = None
        if len(answer) > 0:
            for final_answer in reversed(answer):
                if final_answer not in ["", "."]:
                    break
    return final_answer


def _gsm8k_normalize(answer):
    if answer is None:
        return None
    return answer.replace(",", "").replace("$", "").strip().rstrip(".")


def gsm8k_score(solution_str, ground_truth, format_score, score):
    strict_answer = gsm8k_extract_solution(solution_str, method="strict")
    format_reward = format_score if strict_answer is not None else 0.0
    answer = strict_answer if strict_answer is not None else gsm8k_extract_solution(solution_str, method="flexible")
    answer = _gsm8k_normalize(answer)
    gt = _gsm8k_normalize(ground_truth)
    correctness_reward = score if (answer is not None and answer == gt) else 0.0
    return correctness_reward + format_reward


# ---------------- MATH-style: numina_math, math500 (verl builtin grader) ----------------
def math_score(solution_str, ground_truth, format_score, score):
    format_reward = format_score if _has_boxed(solution_str) else 0.0
    correctness = 0.0
    try:
        from verl.utils.reward_score import math as verl_math
        val = verl_math.compute_score(solution_str, ground_truth)
        if isinstance(val, dict):
            val = val.get("score", 0.0)
        correctness = score if float(val) >= 1.0 else 0.0
    except Exception:
        correctness = 0.0
    return correctness + format_reward


# ---------------- Countdown ----------------
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)


def _safe_eval(expr):
    node = ast.parse(expr, mode="eval")
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED):
            raise ValueError("illegal node: " + type(n).__name__)
    return eval(compile(node, "<expr>", "eval"))


def countdown_score(solution_str, ground_truth, extra_info, format_score, score):
    boxed = _last_boxed(solution_str)
    format_reward = format_score if boxed is not None else 0.0
    if boxed is None or not isinstance(extra_info, dict):
        return 0.0 + format_reward
    numbers = extra_info.get("numbers")
    try:
        target = float(int(ground_truth))
    except Exception:
        return 0.0 + format_reward
    if not numbers:
        return 0.0 + format_reward
    want = sorted(int(x) for x in numbers)

    raw = boxed
    for junk in [BS + "left", BS + "right", BS + "!", BS + ",", BS + ";"]:
        raw = raw.replace(junk, "")
    raw = raw.replace(BS + "times", "*").replace(BS + "cdot", "*").replace(BS + "div", "/")
    raw = raw.replace("{", "(").replace("}", ")")

    for seg in raw.split("="):
        e = seg.strip()
        if not e or not re.fullmatch("[0-9+*/() -]+", e):
            continue
        used = sorted(int(t) for t in re.findall("[0-9]+", e))
        if used != want:
            continue
        try:
            val = _safe_eval(e)
        except Exception:
            continue
        if abs(float(val) - target) < 1e-6:
            return score + format_reward
    return 0.0 + format_reward


# ---------------- Router ----------------
_GSM8K = {"openai/gsm8k", "gsm8k"}
_MATH = {"numina_math", "math500", "math"}
_COUNTDOWN = {"countdown"}


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  format_score=0.1, score=1.0, **kwargs):
    if data_source in _GSM8K:
        return gsm8k_score(solution_str, ground_truth, format_score, score)
    if data_source in _MATH:
        return math_score(solution_str, ground_truth, format_score, score)
    if data_source in _COUNTDOWN:
        return countdown_score(solution_str, ground_truth, extra_info, format_score, score)
    return 0.0
