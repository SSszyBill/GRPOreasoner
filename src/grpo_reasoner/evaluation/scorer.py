"""Route each SampleGeneration through reward_multi and decompose the
scalar reward into (correctness, format_ok, total_reward).

Why two calls to compute_score:
    compute_score returns a scalar = correctness*score + format*format_score.
    Calling twice with different (score, format_score) weights lets us extract
    correctness and format_ok separately, without reimplementing format
    detection ourselves (which would drift from the training reward source).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from grpo_reasoner.evaluation.generator import SampleGeneration
from grpo_reasoner.rewards import reward_multi


@dataclass(frozen=True)
class ScoredCompletion:
    """Decomposition of one completion's reward."""
    correctness: float
    format_ok: float
    total_reward: float


@dataclass(frozen=True)
class ScoredGeneration:
    """All scored completions for one sample."""
    index: int
    data_source: str
    ground_truth: str
    greedy_completion: str
    sampled_completions: tuple[str, ...]
    greedy_score: ScoredCompletion
    sampled_scores: tuple[ScoredCompletion, ...]


def score_completion(
    data_source: str,
    completion: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> ScoredCompletion:
    """Compute (correctness, format_ok, total_reward) for one completion."""
    correctness = reward_multi.compute_score(
        data_source, completion, ground_truth,
        extra_info=extra_info,
        format_score=0.0,
        score=1.0,
    )
    format_ok = reward_multi.compute_score(
        data_source, completion, ground_truth,
        extra_info=extra_info,
        format_score=1.0,
        score=0.0,
    )
    total_reward = correctness + 0.1 * format_ok
    return ScoredCompletion(
        correctness=correctness,
        format_ok=format_ok,
        total_reward=total_reward,
    )


def score_generations(
    generations: Sequence[SampleGeneration],
    extra_info_by_index: dict[int, dict] | None = None,
) -> list[ScoredGeneration]:
    """Score every generation (greedy + all sampled) for every sample."""
    scored: list[ScoredGeneration] = []
    for generation in generations:
        extra_info = None
        if extra_info_by_index is not None:
            extra_info = extra_info_by_index.get(generation.index)

        greedy_score = score_completion(
            generation.data_source,
            generation.greedy_completion,
            generation.ground_truth,
            extra_info,
        )
        sampled_scores = tuple(
            score_completion(
                generation.data_source,
                completion,
                generation.ground_truth,
                extra_info,
            )
            for completion in generation.sampled_completions
        )

        scored.append(ScoredGeneration(
            index=generation.index,
            data_source=generation.data_source,
            ground_truth=generation.ground_truth,
            greedy_completion=generation.greedy_completion,
            sampled_completions=generation.sampled_completions,
            greedy_score=greedy_score,
            sampled_scores=sampled_scores,
        ))
    return scored

