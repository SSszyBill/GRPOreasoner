"""Aggregate ScoredGeneration lists into per-eval-set metrics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from grpo_reasoner.evaluation.scorer import ScoredGeneration
from grpo_reasoner.scoring import compute_score, extract_answer


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


def _synthesize_completion(data_source: str, extracted_answer: str) -> str:
    if data_source == "openai/gsm8k":
        return f"#### {extracted_answer}"
    if data_source in ("math500", "numina_math", "countdown"):
        return "\\boxed{" + extracted_answer + "}"
    return extracted_answer


def compute_pass_at_1(scored_generations: Sequence[ScoredGeneration]) -> float:
    if not scored_generations:
        return 0.0
    correct_total = sum(
        scored.greedy_score.correctness for scored in scored_generations
    )
    return correct_total / len(scored_generations)


def compute_maj_at_k(scored_generations: Sequence[ScoredGeneration]) -> float:
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
            continue

        majority_answer = Counter(valid_answers).most_common(1)[0][0]
        synthesized_completion = _synthesize_completion(
            scored.data_source, majority_answer,
        )

        judged_score = compute_score(
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
    if not scored_generations:
        return 0.0
    format_total = sum(
        scored.greedy_score.format_ok for scored in scored_generations
    )
    return format_total / len(scored_generations)


def compute_total_reward_greedy(scored_generations: Sequence[ScoredGeneration]) -> float:
    if not scored_generations:
        return 0.0
    total = sum(
        scored.greedy_score.total_reward for scored in scored_generations
    )
    return total / len(scored_generations)


def compute_total_reward_sampled_mean(
    scored_generations: Sequence[ScoredGeneration],
) -> float:
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

