"""pytest tests for evaluation.scorer, CPU-only, no vLLM needed.

Covers:
    - Four-quadrant decomposition on GSM8K (correctness x format)
    - MATH500 boxed extraction via verl builtin grader
    - Full score_generations pipeline on a hand-crafted SampleGeneration
"""
from __future__ import annotations

import pytest

from grpo_reasoner.evaluation.generator import SampleGeneration
from grpo_reasoner.evaluation.scorer import score_completion, score_generations


BS = chr(92)
BOXED = BS + "boxed"


def boxed(inner: str) -> str:
    return BOXED + "{" + inner + "}"


# ---------- score_completion: gsm8k four quadrants ----------

@pytest.mark.parametrize(
    "completion, ground_truth, expected_correctness, expected_format_ok",
    [
        ("reasoning\n#### 42", "42", 1.0, 1.0),   # fmt + correct
        ("reasoning\n#### 41", "42", 0.0, 1.0),   # fmt + wrong
        ("the answer is 42",  "42", 1.0, 0.0),    # nofmt + correct (flexible)
        ("the answer is 41",  "42", 0.0, 0.0),    # nofmt + wrong
    ],
    ids=["fmt_correct", "fmt_wrong", "nofmt_correct_flexible", "nofmt_wrong"],
)
def test_score_completion_gsm8k_four_quadrants(
    completion, ground_truth, expected_correctness, expected_format_ok,
):
    result = score_completion("openai/gsm8k", completion, ground_truth)
    assert result.correctness == pytest.approx(expected_correctness)
    assert result.format_ok == pytest.approx(expected_format_ok)
    expected_total = expected_correctness + 0.1 * expected_format_ok
    assert result.total_reward == pytest.approx(expected_total)


# ---------- score_completion: math500 (verl-dependent) ----------

def test_score_completion_math500_boxed():
    """Skips if verl is not importable, otherwise checks boxed extraction."""
    try:
        from verl.utils.reward_score import math as _  # noqa: F401
    except ImportError:
        pytest.skip("verl not importable")

    result = score_completion("math500", "answer is " + boxed("42"), "42")
    assert result.format_ok == pytest.approx(1.0)
    assert result.correctness == pytest.approx(1.0)


# ---------- score_generations pipeline ----------

@pytest.fixture
def gsm8k_sample_generation() -> SampleGeneration:
    """One SampleGeneration with a mix of correct/wrong and formatted/unformatted
    completions to exercise all four quadrants in one pipeline call.
    """
    return SampleGeneration(
        index=0,
        data_source="openai/gsm8k",
        prompt_text="fake prompt",
        ground_truth="42",
        raw_question="fake question",
        greedy_completion="reasoning\n#### 42",
        sampled_completions=(
            "a\n#### 42",           # correct + fmt
            "b\n#### 42",           # correct + fmt
            "c\n#### 41",           # wrong + fmt
            "the answer is 41",    # wrong + nofmt
        ),
    )


def test_score_generations_greedy(gsm8k_sample_generation):
    scored = score_generations([gsm8k_sample_generation])
    assert len(scored) == 1
    assert scored[0].greedy_score.correctness == pytest.approx(1.0)
    assert scored[0].greedy_score.format_ok == pytest.approx(1.0)
    assert scored[0].greedy_score.total_reward == pytest.approx(1.1)


@pytest.mark.parametrize(
    "sampled_index, expected_correctness, expected_format_ok",
    [
        (0, 1.0, 1.0),
        (1, 1.0, 1.0),
        (2, 0.0, 1.0),
        (3, 0.0, 0.0),
    ],
)
def test_score_generations_sampled(
    gsm8k_sample_generation, sampled_index, expected_correctness, expected_format_ok,
):
    scored = score_generations([gsm8k_sample_generation])
    sampled = scored[0].sampled_scores[sampled_index]
    assert sampled.correctness == pytest.approx(expected_correctness)
    assert sampled.format_ok == pytest.approx(expected_format_ok)