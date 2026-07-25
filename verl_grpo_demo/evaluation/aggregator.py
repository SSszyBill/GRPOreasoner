"""Aggregate ScoredGeneration lists into per-eval-set metrics.

Metrics:
    pass_at_1                  : mean(greedy.correctness)
    maj_at_8                   : true majority vote (extract -> Counter -> re-judge)
    format_rate                : mean(greedy.format_ok)
    total_reward_greedy        : mean(greedy.total_reward)
    total_reward_sampled_mean  : mean(sampled.total_reward) over all sampled completions

Answer extraction (per data_source) delegates to reward_multi's helpers so the
same source of truth is used for judging in both training and evaluation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from evaluation.scorer import ScoredGeneration
from utils import reward_multi


@dataclass(frozen=True)
class EvalSetMetrics:
    """All four eval metrics for one eval_set, plus provenance fields."""
    eval_set_name: str
    data_source: str
    num_samples: int
    pass_at_1: float
    maj_at_8: float
    format_rate: float
    total_reward_greedy: float
    total_reward_sampled_mean: float


def extract_answer(data_source: str, completion: str) -> str | None:
    """Extract the model's proposed answer from a completion string.

    Uses reward_multi's existing extractors so extraction logic stays in one
    place. Returns None if nothing extractable found.
    """
    if data_source == "openai/gsm8k":
        # Try strict (#### number) first, then flexible (last number in text).
        strict_answer = reward_multi.gsm8k_extract_solution(completion, method="strict")
        if strict_answer is not None:
            return reward_multi._gsm8k_normalize(strict_answer)
        flexible_answer = reward_multi.gsm8k_extract_solution(completion, method="flexible")
        return reward_multi._gsm8k_normalize(flexible_answer) if flexible_answer else None

    if data_source in ("math500", "numina_math", "countdown"):
        # All three use \boxed{...} extraction; math500 / numina content is a
        # LaTeX expression, countdown content is an arithmetic expression.
        return reward_multi._last_boxed(completion)

    return None


def _synthesize_completion(data_source: str, extracted_answer: str) -> str:
    """Wrap an extracted answer back into the format reward_multi expects.

    This lets us re-judge a bare answer string (from majority vote) through the
    same compute_score path as any regular completion.
    """
    if data_source == "openai/gsm8k":
        return f"#### {extracted_answer}"
    if data_source in ("math500", "numina_math", "countdown"):
        return "\\boxed{" + extracted_answer + "}"
    return extracted_answer


def compute_pass_at_1(scored_generations: Sequence[ScoredGeneration]) -> float:
    """pass@1 = mean of greedy correctness across all samples."""
    if not scored_generations:
        return 0.0
    correct_total = sum(
        scored.greedy_score.correctness for scored in scored_generations
    )
    return correct_total / len(scored_generations)


def compute_maj_at_k(scored_generations: Sequence[ScoredGeneration]) -> float:
    """True majority vote across sampled completions, then re-judged.

    For each sample:
      1. Extract answer from each of k sampled completions.
      2. Skip completions that produced no extractable answer.
      3. Majority winner: Counter.most_common(1) (ties broken by first-inserted).
      4. If no completion produced an answer, count sample as wrong.
      5. Otherwise re-judge synthesized "answer completion" via reward_multi.

    Returns mean over all samples.
    """
    if not scored_generations:
        return 0.0

    correct_count = 0
    for scored in scored_generations:
        extracted_answers = [
            extract_answer(scored.data_source, completion)
            for completion in scored.sampled_completions
        ]
        valid_answers = [answer for answer in extracted_answers if answer]
        if not valid_answers:
            continue  # no valid answer => sample counted as wrong

        majority_answer = Counter(valid_answers).most_common(1)[0][0]
        synthesized_completion = _synthesize_completion(
            scored.data_source, majority_answer,
        )

        # Re-judge via reward_multi with pure correctness weight.
        judged_score = reward_multi.compute_score(
            scored.data_source,
            synthesized_completion,
            scored.ground_truth,
            extra_info=None,
            format_score=0.0,
            score=1.0,
        )
        if judged_score > 0.5:
            correct_count += 1

    return correct_count / len(scored_generations)


def compute_format_rate(scored_generations: Sequence[ScoredGeneration]) -> float:
    """format_rate = mean of greedy format_ok across all samples."""
    if not scored_generations:
        return 0.0
    format_total = sum(
        scored.greedy_score.format_ok for scored in scored_generations
    )
    return format_total / len(scored_generations)


def compute_total_reward_greedy(scored_generations: Sequence[ScoredGeneration]) -> float:
    """Mean of greedy total_reward across all samples."""
    if not scored_generations:
        return 0.0
    total = sum(
        scored.greedy_score.total_reward for scored in scored_generations
    )
    return total / len(scored_generations)


def compute_total_reward_sampled_mean(
    scored_generations: Sequence[ScoredGeneration],
) -> float:
    """Mean of sampled total_reward across ALL sampled completions (flat)."""
    all_sampled_totals = [
        sampled.total_reward
        for scored in scored_generations
        for sampled in scored.sampled_scores
    ]
    if not all_sampled_totals:
        return 0.0
    return sum(all_sampled_totals) / len(all_sampled_totals)


def aggregate_eval_set(
    eval_set_name: str,
    data_source: str,
    scored_generations: Sequence[ScoredGeneration],
) -> EvalSetMetrics:
    """Compute all six metrics for one eval set."""
    return EvalSetMetrics(
        eval_set_name=eval_set_name,
        data_source=data_source,
        num_samples=len(scored_generations),
        pass_at_1=compute_pass_at_1(scored_generations),
        maj_at_8=compute_maj_at_k(scored_generations),
        format_rate=compute_format_rate(scored_generations),
        total_reward_greedy=compute_total_reward_greedy(scored_generations),
        total_reward_sampled_mean=compute_total_reward_sampled_mean(scored_generations),
    )