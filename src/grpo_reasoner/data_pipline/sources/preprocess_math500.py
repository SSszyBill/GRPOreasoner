"""MATH-500 conversion helpers for verl-format eval parquet."""
from __future__ import annotations
from grpo_reasoner.data_pipline.schema import VerlSample

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DATA_SOURCE = "math500"
ABILITY = "math"


def process_example(example: dict, idx: int, split: str) -> dict:
    problem = example.get("problem", "") or ""
    answer = (example.get("answer", "") or "").strip()
    return VerlSample.build(
            data_source=DATA_SOURCE,
            user_content=content,
            ground_truth=str(target),
            ability=ABILITY,
            extra_info={"split": split, "index": idx, "numbers": nums, "target": target, "example_solution": expr},
            ).to_dict()

