"""Dispatch by data_source to per-source scorers.
This is the ONLY module external callers should use (via
from grpo_reasoner.scoring import compute_score, extract_answer).
"""
from __future__ import annotations
from grpo_reasoner.scoring import countdown, gsm8k, math
_GSM8K_SOURCES = frozenset({"openai/gsm8k", "gsm8k"})
_MATH_SOURCES = frozenset({"numina_math", "math500", "math"})
_COUNTDOWN_SOURCES = frozenset({"countdown"})
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_score: float = 0.1,
    score: float = 1.0,
    _unused_kwargs=None,
) -> float:
    """Return correctnessscore + formatformat_score.
    Signature stays fixed because verl calls this via file path + name.
    Extra kwargs are swallowed to tolerate verl passing new fields in the future.
    """
    if data_source in _GSM8K_SOURCES:
        return gsm8k.score(solution_str, ground_truth, format_score, score)
    if data_source in _MATH_SOURCES:
        return math.score(solution_str, ground_truth, format_score, score)
    if data_source in _COUNTDOWN_SOURCES:
        return countdown.score(solution_str, ground_truth, extra_info, format_score, score)
    return 0.0
def extract_answer(data_source: str, completion: str) -> str | None:
    """Return the model's proposed answer string, or None.
    Used by evaluation.aggregator for maj@k (needs to compare extracted
    answers across samples) and for per-sample parquet output.
    """
    if data_source in _GSM8K_SOURCES:
        return gsm8k.extract_answer(completion)
    if data_source in _MATH_SOURCES:
        return math.extract_answer(completion)
    if data_source in _COUNTDOWN_SOURCES:
        return countdown.extract_answer(completion)
    return None
