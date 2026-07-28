"""MATH-500 conversion helpers for verl-format eval parquet."""
from __future__ import annotations


INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DATA_SOURCE = "math500"
ABILITY = "math"


def process_example(example: dict, idx: int, split: str) -> dict:
    problem = example.get("problem", "") or ""
    answer = (example.get("answer", "") or "").strip()
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": problem + " " + INSTRUCTION}],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "problem": problem,
            "subject": example.get("subject", ""),
            "level": example.get("level", ""),
            "unique_id": example.get("unique_id", ""),
        },
    }

