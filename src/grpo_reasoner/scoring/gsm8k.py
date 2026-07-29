"""GSM8K scoring: '#### number' format + numeric equality.
Format and correctness are decoupled:
correct answer without '####' still earns correctness*score
wrong answer with    '####' still earns format*format_score
"""
from __future__ import annotations
import re

def extract_answer(completion: str) -> str | None:
    """Return the normalized answer string, or None.
    Two-pass extraction:
        strict:   captures '#### number'
        flexible: falls back to the last standalone number in the text
    """
    strict = _extract(completion, method="strict")
    if strict is not None:
        return _normalize(strict)
    flexible = _extract(completion, method="flexible")
    return _normalize(flexible) if flexible else None

def score(
    solution_str: str,
    ground_truth: str,
    format_score: float,
    score: float,
) -> float:
    """Return correctnessscore + formatformat_score."""
    strict = _extract(solution_str, method="strict")
    format_reward = format_score if strict is not None else 0.0
    answer_raw = strict if strict is not None else _extract(solution_str, method="flexible")
    answer_norm = _normalize(answer_raw)
    ground_truth_norm = _normalize(ground_truth)
    correct = answer_norm is not None and answer_norm == ground_truth_norm
    correctness_reward = score if correct else 0.0
    return correctness_reward + format_reward

def _extract(text: str, method: str = "strict") -> str | None:
    assert method in ("strict", "flexible")
    if method == "strict":
        matches = re.findall(r"#### (-?[0-9.,]+)", text)
        return matches[-1].replace(",", "").replace("$", "") if matches else None
    matches = re.findall(r"(-?[0-9.,]+)", text)
    if not matches:
        return None
    for candidate in reversed(matches):
        if candidate not in ("", "."):
            return candidate
    return None
def _normalize(answer: str | None) -> str | None:
    if answer is None:
        return None
    return answer.replace(",", "").replace("$", "").strip().rstrip(".")