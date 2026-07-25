# Four-quadrant unit tests for utils/reward_multi.py (per data_source).
# Layout assumed:
#   verl_grpo_demo/utils/reward_multi.py   (+ utils/__init__.py)
#   verl_grpo_demo/tests/test_reward_multi.py   (this file)
# Run from anywhere, e.g.:  python tests/test_reward_multi.py
import os
import sys

# Put the project root (verl_grpo_demo) on sys.path so `import utils` works
# no matter the current working directory or how the script is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import reward_multi as R

BS = chr(92)


def boxed(s):
    return BS + "boxed{" + s + "}"


TIMES = " " + BS + "times "
results = []


def check(name, got, exp, tol=1e-6):
    ok = abs(got - exp) < tol
    print(("PASS" if ok else "FAIL"), name, "got=%.3f exp=%.3f" % (got, exp))
    results.append(ok)


print("==== GSM8K ====")
ds = "openai/gsm8k"
check("gsm8k fmt+correct", R.compute_score(ds, "reason\n#### 42", "42"), 1.1)
check("gsm8k fmt+wrong", R.compute_score(ds, "reason\n#### 41", "42"), 0.1)
check("gsm8k nofmt+correct(flexible)", R.compute_score(ds, "the answer is 42", "42"), 1.0)
check("gsm8k nofmt+wrong", R.compute_score(ds, "the answer is 41", "42"), 0.0)

print("==== Countdown ====")
ds = "countdown"
ei = {"numbers": [3, 8], "target": 24}
check("cd fmt+correct", R.compute_score(ds, "so " + boxed("3" + TIMES + "8"), "24", ei), 1.1)
check("cd fmt+wrong", R.compute_score(ds, "so " + boxed("3 + 8"), "24", ei), 0.1)
check("cd reuse-number(fmt ok, ans wrong)", R.compute_score(ds, "so " + boxed("8" + TIMES + "8"), "24", ei), 0.1)
check("cd equation-form", R.compute_score(ds, "thus " + boxed("24 = 3" + TIMES + "8"), "24", ei), 1.1)
check("cd nofmt", R.compute_score(ds, "3*8", "24", ei), 0.0)

print("==== MATH (numina_math / math500) ====")
try:
    from verl.utils.reward_score import math as verl_math
    HAS_VERL = True
except Exception as e:
    HAS_VERL = False
    print("SKIP math tests: verl not importable ->", type(e).__name__)

if HAS_VERL:
    print("verl_math.compute_score(boxed 42, '42') raw =", verl_math.compute_score(boxed("42"), "42"))
    ds = "math500"
    check("math fmt+correct", R.compute_score(ds, "hence " + boxed("42"), "42"), 1.1)
    check("math fmt+wrong", R.compute_score(ds, "hence " + boxed("41"), "42"), 0.1)
    check("math frac correct", R.compute_score(ds, boxed(BS + "frac{1}{2}"), BS + "frac{1}{2}"), 1.1)
    check("math nofmt+wrong", R.compute_score(ds, "the answer is 41", "42"), 0.0)

print("-" * 40)
print("TOTAL: %d/%d passed" % (sum(1 for r in results if r), len(results)))