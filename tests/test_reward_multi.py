"""Four-quadrant unit tests for grpo_reasoner.scoring per data_source.
Each source is tested in four quadrants:
format correct + answer correct
format correct + answer wrong
format wrong   + answer correct (gsm8k flexible extraction only)
format wrong   + answer wrong
MATH-style tests skip gracefully when verl is not importable.
"""
import pytest
from grpo_reasoner.scoring import compute_score
BS = chr(92)
def boxed(inner: str) -> str:
    return BS + "boxed{" + inner + "}"
TIMES = " " + BS + "times "

# GSM8K 
@pytest.mark.parametrize(
    "solution, expected",
    [
    ("reasonn#### 42", 1.1),
    ("reasonn#### 41", 0.1),
    ("the answer is 42", 1.0),
    ("the answer is 41", 0.0),
    ],
    ids=["fmt_correct", "fmt_wrong", "nofmt_correct_flexible", "nofmt_wrong"],
)
def test_gsm8k_four_quadrants(solution, expected):
    assert compute_score("openai/gsm8k", solution, "42") == pytest.approx(expected)
# Countdown
COUNTDOWN_EXTRA = {"numbers": [3, 8], "target": 24}
@pytest.mark.parametrize(
    "solution, expected",
    [
    ("so " + boxed("3" + TIMES + "8"), 1.1),
    ("so " + boxed("3 + 8"), 0.1),
    ("so " + boxed("8" + TIMES + "8"), 0.1),
    ("thus " + boxed("24 = 3" + TIMES + "8"), 1.1),
    ("3*8", 0.0),
    ],
    ids=["fmt_correct", "fmt_wrong", "reuses_number", "equation_form", "nofmt"],
)
def test_countdown_four_quadrants(solution, expected):
    result = compute_score("countdown", solution, "24", COUNTDOWN_EXTRA)
    assert result == pytest.approx(expected)
# MATH (numina_math / math500)
def _verl_math_available() -> bool:
    try:
        from verl.utils.reward_score import math as _  # noqa: F401
        return True
    except Exception:
        return False
    
needs_verl = pytest.mark.skipif(
    not _verl_math_available(), reason="verl not importable in this env"
    )
@needs_verl
@pytest.mark.parametrize(
    "solution, ground_truth, expected",
    [
    ("hence " + boxed("42"), "42", 1.1),
    ("hence " + boxed("41"), "42", 0.1),
    (boxed(BS + "frac{1}{2}"), BS + "frac{1}{2}", 1.1),
    ("the answer is 41", "42", 0.0),
    ],
    ids=["fmt_correct", "fmt_wrong", "frac_normalize", "nofmt_wrong"],
)
def test_math_four_quadrants(solution, ground_truth, expected):
    assert compute_score("math500", solution, ground_truth) == pytest.approx(expected)