"""pytest tests for evaluation.aggregator."""
from __future__ import annotations

import pytest

from evaluation.aggregator import (
    aggregate_eval_set,
    compute_format_rate,
    compute_maj_at_k,
    compute_pass_at_1,
    compute_total_reward_greedy,
    compute_total_reward_sampled_mean,
    extract_answer,
)
from evaluation.scorer import ScoredCompletion, ScoredGeneration


BS = chr(92)


def boxed(inner: str) -> str:
    return BS + "boxed{" + inner + "}"


# ---------- extract_answer ----------

@pytest.mark.parametrize(
    "data_source, completion, expected",
    [
        ("openai/gsm8k", "reasoning\n#### 42", "42"),
        ("openai/gsm8k", "the answer is 42", "42"),  # flexible path
        ("openai/gsm8k", "no digits here", None),
        ("math500", "answer is " + boxed("3.14"), "3.14"),
        ("math500", "no boxed", None),
        ("numina_math", "final " + boxed("x + 1"), "x + 1"),
        ("countdown", "so " + boxed("3 * 8"), "3 * 8"),
        ("unknown_source", "whatever", None),
    ],
    ids=[
        "gsm8k_strict",
        "gsm8k_flexible",
        "gsm8k_none",
        "math500_boxed",
        "math500_none",
        "numina_boxed",
        "countdown_boxed",
        "unknown_source_returns_none",
    ],
)
def test_extract_answer(data_source, completion, expected):
    assert extract_answer(data_source, completion) == expected


# ---------- Fixtures ----------

def _make_scored(
    data_source: str,
    ground_truth: str,
    greedy_completion: str,
    greedy_correct: float,
    greedy_format: float,
    sampled_specs: list[tuple[str, float, float]],
    index: int = 0,
) -> ScoredGeneration:
    """Build a ScoredGeneration from (completion, correct, format) triples."""
    sampled_completions = tuple(spec[0] for spec in sampled_specs)
    sampled_scores = tuple(
        ScoredCompletion(
            correctness=correct,
            format_ok=fmt,
            total_reward=correct + 0.1 * fmt,
        )
        for _, correct, fmt in sampled_specs
    )
    greedy_score = ScoredCompletion(
        correctness=greedy_correct,
        format_ok=greedy_format,
        total_reward=greedy_correct + 0.1 * greedy_format,
    )
    return ScoredGeneration(
        index=index,
        data_source=data_source,
        ground_truth=ground_truth,
        greedy_completion=greedy_completion,
        sampled_completions=sampled_completions,
        greedy_score=greedy_score,
        sampled_scores=sampled_scores,
    )


@pytest.fixture
def three_gsm8k_samples():
    """Three GSM8K samples designed so metrics are hand-verifiable.

    Sample 0: greedy correct+fmt; sampled 8 all '#### 42' (correct)
    Sample 1: greedy wrong+fmt;    sampled 5x '#### 41' (wrong majority)
    Sample 2: greedy correct+nofmt; sampled 6x '#### 42' vs 2x '#### 41'
    """
    samples = [
        _make_scored(
            data_source="openai/gsm8k",
            ground_truth="42",
            greedy_completion="reason\n#### 42",
            greedy_correct=1.0, greedy_format=1.0,
            sampled_specs=[(f"a\n#### 42", 1.0, 1.0)] * 8,
            index=0,
        ),
        _make_scored(
            data_source="openai/gsm8k",
            ground_truth="42",
            greedy_completion="reason\n#### 41",
            greedy_correct=0.0, greedy_format=1.0,
            sampled_specs=(
                [(f"a\n#### 41", 0.0, 1.0)] * 5
                + [(f"b\n#### 42", 1.0, 1.0)] * 3
            ),
            index=1,
        ),
        _make_scored(
            data_source="openai/gsm8k",
            ground_truth="42",
            greedy_completion="the answer is 42",
            greedy_correct=1.0, greedy_format=0.0,
            sampled_specs=(
                [(f"a\n#### 42", 1.0, 1.0)] * 6
                + [(f"b\n#### 41", 0.0, 1.0)] * 2
            ),
            index=2,
        ),
    ]
    return samples


# ---------- Individual metric fns ----------

def test_compute_pass_at_1(three_gsm8k_samples):
    # Greedy correctness: 1, 0, 1 -> mean 2/3
    assert compute_pass_at_1(three_gsm8k_samples) == pytest.approx(2 / 3)


def test_compute_format_rate(three_gsm8k_samples):
    # Greedy format: 1, 1, 0 -> mean 2/3
    assert compute_format_rate(three_gsm8k_samples) == pytest.approx(2 / 3)


def test_compute_maj_at_k(three_gsm8k_samples):
    # Sample 0: 8x '42' majority -> correct  (1.0)
    # Sample 1: 5x '41' vs 3x '42', majority '41' -> wrong (0.0)
    # Sample 2: 6x '42' vs 2x '41', majority '42' -> correct (1.0)
    # Mean = 2/3
    assert compute_maj_at_k(three_gsm8k_samples) == pytest.approx(2 / 3)


def test_compute_total_reward_greedy(three_gsm8k_samples):
    # Totals: 1.1, 0.1, 1.0 -> mean 2.2/3
    assert compute_total_reward_greedy(three_gsm8k_samples) == pytest.approx(2.2 / 3)


def test_compute_total_reward_sampled_mean(three_gsm8k_samples):
    # Sample 0: 8x (1.0+0.1) = 8x 1.1
    # Sample 1: 5x (0.0+0.1) + 3x (1.0+0.1) = 5x 0.1 + 3x 1.1
    # Sample 2: 6x 1.1 + 2x 0.1
    # Total sum = 8*1.1 + (5*0.1 + 3*1.1) + (6*1.1 + 2*0.1)
    #           = 8.8 + 0.5 + 3.3 + 6.6 + 0.2 = 19.4
    # Count = 24
    expected = (8 * 1.1 + 5 * 0.1 + 3 * 1.1 + 6 * 1.1 + 2 * 0.1) / 24
    assert compute_total_reward_sampled_mean(three_gsm8k_samples) == pytest.approx(expected)


# ---------- aggregate_eval_set ----------

def test_aggregate_eval_set_shape(three_gsm8k_samples):
    metrics = aggregate_eval_set(
        "gsm8k_test", "openai/gsm8k", three_gsm8k_samples,
    )
    assert metrics.eval_set_name == "gsm8k_test"
    assert metrics.data_source == "openai/gsm8k"
    assert metrics.num_samples == 3
    assert metrics.pass_at_1 == pytest.approx(2 / 3)
    assert metrics.maj_at_8 == pytest.approx(2 / 3)
    assert metrics.format_rate == pytest.approx(2 / 3)


def test_aggregate_eval_set_empty():
    metrics = aggregate_eval_set("empty_set", "openai/gsm8k", [])
    assert metrics.num_samples == 0
    assert metrics.pass_at_1 == 0.0
    assert metrics.maj_at_8 == 0.0
    assert metrics.format_rate == 0.0