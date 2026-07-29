# -*- coding: utf-8 -*-
"""Countdown 数据生成 -> verl 五字段 parquet。
保证有解：先对随机整数套一棵随机合法表达式树算出 target，操作数即 numbers。
约束：+,-,*,/；减法结果为正；除法必须整除；中间结果为正整数且不超上限。
输出 schema: data_source / prompt / ability / reward_model / extra_info
"""
import os
import random
import logging
import argparse
from datetime import datetime

INSTRUCTION = (
    "Using each of the given numbers exactly once and the operations +, -, *, /, "
    "write an arithmetic expression that equals the target. "
    "Please reason step by step, and put your final expression within \\boxed{}."
)
DATA_SOURCE = "countdown"
ABILITY = "math"


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("generate_countdown")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def combine(a, b, max_inter):
    av, ae = a
    bv, be = b
    ops = ["+", "-", "*", "/"]
    random.shuffle(ops)
    for op in ops:
        if op == "+":
            v = av + bv
        elif op == "-":
            if av <= bv:
                continue
            v = av - bv
        elif op == "*":
            v = av * bv
        else:
            if bv == 0 or av % bv != 0:
                continue
            v = av // bv
        if v <= 0 or v > max_inter:
            continue
        return (v, f"({ae} {op} {be})")
    return None


def make_puzzle(k, lo, hi, max_inter, max_tries=50):
    for _ in range(max_tries):
        nums = [random.randint(lo, hi) for _ in range(k)]
        nodes = [(n, str(n)) for n in nums]
        random.shuffle(nodes)
        ok = True
        while len(nodes) > 1:
            a = nodes.pop()
            b = nodes.pop()
            res = combine(a, b, max_inter)
            if res is None:
                ok = False
                break
            nodes.append(res)
        if ok:
            target, expr = nodes[0]
            return nums, target, expr
    return None


def to_record(nums, target, expr, idx, split):
    numbers_str = ", ".join(str(n) for n in nums)
    content = (
        f"Numbers: {numbers_str}\n"
        f"Target: {target}\n"
        f"{INSTRUCTION}"
    )
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": content}],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": str(target)},
        "extra_info": {
            "split": split,
            "index": idx,
            "numbers": nums,
            "target": target,
            "example_solution": expr,
        },
    }


def gen_split(n, k_min, k_max, lo, hi, max_inter, seen, split, logger):
    records = []
    tries = 0
    while len(records) < n:
        tries += 1
        k = random.randint(k_min, k_max)
        p = make_puzzle(k, lo, hi, max_inter)
        if p is None:
            continue
        nums, target, expr = p
        key = (tuple(sorted(nums)), target)
        if key in seen:
            continue
        seen.add(key)
        records.append(to_record(nums, target, expr, len(records), split))
        if len(records) % 20000 == 0:
            logger.info(f"  [{split}] 生成 {len(records)}/{n}")
    logger.info(f"[{split}] 完成 {len(records)} 条 (尝试 {tries} 次, 已去重)")
    return records


def sanity_check(records, logger, k=5):
    sample = random.sample(records, min(k, len(records)))
    bad = 0
    for r in sample:
        expr = r["extra_info"]["example_solution"]
        target = r["extra_info"]["target"]
        try:
            if abs(eval(expr) - target) > 1e-6:
                bad += 1
        except Exception:
            bad += 1
    logger.info(f"  自检: 抽样 {len(sample)} 条, 参考解表达式核对不符 {bad} 条")


def main():
    import datasets
    home = os.path.expanduser("~/autodl-tmp/rl_playground")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=f"{home}/data/countdown")
    parser.add_argument("--log_dir", default=f"{home}/verl_grpo_demo/logs")
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--k_min", type=int, default=3)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--lo", type=int, default=1)
    parser.add_argument("--hi", type=int, default=99)
    parser.add_argument("--max_inter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"generate_countdown_{ts}.log")
    logger = setup_logger(log_path)
    logger.info(f"===== Countdown 生成开始 seed={args.seed} k={args.k_min}~{args.k_max} range={args.lo}~{args.hi} max_inter={args.max_inter} =====")

    seen = set()
    logger.info("生成 train ...")
    train = gen_split(args.train, args.k_min, args.k_max, args.lo, args.hi, args.max_inter, seen, "train", logger)
    sanity_check(train, logger)
    logger.info("生成 test (与 train 去重) ...")
    test = gen_split(args.test, args.k_min, args.k_max, args.lo, args.hi, args.max_inter, seen, "test", logger)
    sanity_check(test, logger)

    os.makedirs(args.local_dir, exist_ok=True)
    datasets.Dataset.from_list(train).to_parquet(os.path.join(args.local_dir, "train.parquet"))
    datasets.Dataset.from_list(test).to_parquet(os.path.join(args.local_dir, "test.parquet"))
    logger.info(f"已写入 {args.local_dir}/train.parquet ({len(train)}) 和 test.parquet ({len(test)})")
    logger.info("===== Countdown 生成完成 =====")


if __name__ == "__main__":
    main()
