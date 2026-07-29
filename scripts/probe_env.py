import importlib

MODS = [
    "math_verify",
    "sympy",
    "latex2sympy2",
    "antlr4",
    "verl",
    "verl.utils.reward_score",
    "verl.utils.reward_score.math",
    "verl.utils.reward_score.gsm8k",
]

for m in MODS:
    try:
        importlib.import_module(m)
        print("OK  ", m)
    except Exception as e:
        print("NO  ", m, "->", type(e).__name__, str(e)[:80])

print("-" * 40)
try:
    import os
    import verl.utils.reward_score as r
    d = os.path.dirname(r.__file__)
    print("reward_score 目录:", d)
    print("现成判分模块:", sorted(f for f in os.listdir(d) if not f.startswith("__")))
    print("reward_score 导出:", [x for x in dir(r) if not x.startswith("_")])
except Exception as e:
    print("无法列出 reward_score 目录:", type(e).__name__, str(e)[:120])