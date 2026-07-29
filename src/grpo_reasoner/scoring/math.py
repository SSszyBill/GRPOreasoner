"""MATH-style scoring (numina_math, math500): \boxed{answer} + verl grader.
Correctness delegates to verl.utils.reward_score.math, which handles MATH
normalization (frac, sqrt, boxed unwrapping, etc.). If verl is not
importable, correctness silently falls back to 0.
"""
from future import annotations
from grpo_reasoner.scoring._boxed import has_boxed, last_boxed

def extract_answer(completion: str) -> str | None:
    """Return the last \boxed{} content, or None."""
    return last_boxed(completion)

def score(
    solution_str: str,
    ground_truth: str,
    format_score: float,
    score: float,
) -> float:
    """Format = \boxed{} present. Correctness = verl grader verdict."""
    format_reward = format_score if has_boxed(solution_str) else 0.0
    correctness_reward = 0.0
    try:
        from verl.utils.reward_score import math as verl_math
        raw = verl_math.compute_score(solution_str, ground_truth)
        if isinstance(raw, dict):
            raw = raw.get("score", 0.0)
        correctness_reward = score if float(raw) >= 1.0 else 0.0
    except Exception:
        correctness_reward = 0.0
    return correctness_reward + format_reward