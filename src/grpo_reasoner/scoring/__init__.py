"""Scoring / reward logic (shared by training and evaluation).
Public API:
compute_score(data_source, solution_str, ground_truth,
extra_info=None, format_score=0.1, score=1.0) -> float
extract_answer(data_source, completion) -> str | None
Everything else in this package (gsm8k, math, countdown, _boxed) is
implementation detail and may change.
"""
from grpo_reasoner.scoring.router import compute_score, extract_answer
__all__ = ["compute_score", "extract_answer"]

