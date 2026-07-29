"""Countdown puzzle generation helpers."""
from __future__ import annotations

import random
from typing import Any
from grpo_reasoner.data_pipline.schema import VerlSample

INSTRUCTION = (
    "Using each of the given numbers exactly once and the operations +, -, *, /, "
    "write an arithmetic expression that equals the target. "
    "Please reason step by step, and put your final expression within \\boxed{}."
)
DATA_SOURCE = "countdown"
ABILITY = "math"


def combine(a: tuple[int, str], b: tuple[int, str], max_inter: int) -> tuple[int, str] | None:
    av, ae = a
    bv, be = b
    ops = ["+", "-", "*", "/"]
    random.shuffle(ops)
    for op in ops:
        if op == "+":
            value = av + bv
        elif op == "-":
            if av <= bv:
                continue
            value = av - bv
        elif op == "*":
            value = av * bv
        else:
            if bv == 0 or av % bv != 0:
                continue
            value = av // bv
        if value <= 0 or value > max_inter:
            continue
        return value, f"({ae} {op} {be})"
    return None


def make_puzzle(k: int, lo: int, hi: int, max_inter: int, max_tries: int = 50) -> tuple[list[int], int, str] | None:
    for _ in range(max_tries):
        nums = [random.randint(lo, hi) for _ in range(k)]
        nodes = [(n, str(n)) for n in nums]
        random.shuffle(nodes)
        ok = True
        while len(nodes) > 1:
            a = nodes.pop()
            b = nodes.pop()
            result = combine(a, b, max_inter)
            if result is None:
                ok = False
                break
            nodes.append(result)
        if ok:
            target, expr = nodes[0]
            return nums, target, expr
    return None


def to_record(nums: list[int], target: int, expr: str, idx: int, split: str) -> dict[str, Any]:
    numbers_str = ", ".join(str(n) for n in nums)
    content = (
        f"Numbers: {numbers_str}\n"
        f"Target: {target}\n"
        f"{INSTRUCTION}"
    )
    return VerlSample.build(
        data_source=DATA_SOURCE,
        user_content=content,
        ground_truth=str(target),
        ability=ABILITY,
        extra_info={"split": split, "index": idx, "numbers": nums, "target": target, "example_solution": expr},
        ).to_dict()

def gen_split(
    n: int,
    k_min: int,
    k_max: int,
    lo: int,
    hi: int,
    max_inter: int,
    seen: set[tuple[tuple[int, ...], int]],
    split: str,
    logger=None,
) -> list[dict[str, Any]]:
    records = []
    tries = 0
    while len(records) < n:
        tries += 1
        k = random.randint(k_min, k_max)
        puzzle = make_puzzle(k, lo, hi, max_inter)
        if puzzle is None:
            continue
        nums, target, expr = puzzle
        key = (tuple(sorted(nums)), target)
        if key in seen:
            continue
        seen.add(key)
        records.append(to_record(nums, target, expr, len(records), split))
        if logger is not None and len(records) % 20000 == 0:
            logger.info(f"  [{split}] generated {len(records)}/{n}")
    if logger is not None:
        logger.info(f"[{split}] done {len(records)} records ({tries} attempts, deduped)")
    return records


def count_bad_reference_solutions(records: list[dict[str, Any]], k: int = 5) -> int:
    sample = random.sample(records, min(k, len(records)))
    bad = 0
    for record in sample:
        expr = record["extra_info"]["example_solution"]
        target = record["extra_info"]["target"]
        try:
            if abs(eval(expr) - target) > 1e-6:
                bad += 1
        except Exception:
            bad += 1
    return bad

