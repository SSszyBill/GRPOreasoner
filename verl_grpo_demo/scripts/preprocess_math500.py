# -*- coding: utf-8 -*-
"""MATH-500 -> verl 评测 parquet（与 gsm8k.py / preprocess_numina.py 对齐）。
评测集：保留全部 500 条，不过滤；answer 字段直接作为 ground_truth。
输出 schema: data_source / prompt / ability / reward_model / extra_info
"""
import os
import logging
import argparse
from datetime import datetime

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DATA_SOURCE = "math500"
ABILITY = "math"


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("preprocess_math500")
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


def process_example(example, idx, split):
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


def main():
    import datasets
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/autodl-tmp/rl_playground/hf_cache"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.expanduser("~/autodl-tmp/rl_playground/data/math500"))
    parser.add_argument("--log_dir", default=os.path.expanduser("~/autodl-tmp/rl_playground/verl_grpo_demo/logs"))
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"preprocess_math500_{ts}.log")
    logger = setup_logger(log_path)
    logger.info("===== MATH500 转换开始 =====")
    logger.info(f"数据源: HuggingFaceH4/MATH-500 | 输出目录: {args.local_dir} | 日志: {log_path}")

    ds = datasets.load_dataset("HuggingFaceH4/MATH-500")
    os.makedirs(args.local_dir, exist_ok=True)

    split = "test"
    d = ds[split]
    mapped = d.map(lambda ex, idx: process_example(ex, idx, split),
                   with_indices=True, remove_columns=d.column_names)

    gts = mapped["reward_model"]
    n_empty = sum(1 for g in gts if not g.get("ground_truth"))
    logger.info(f"[{split}] 原始 {len(d)} | 转换 {len(mapped)} | 空答案 {n_empty}")
    if n_empty:
        logger.warning(f"[{split}] 有 {n_empty} 条空答案，需人工核验")

    out = os.path.join(args.local_dir, f"{split}.parquet")
    mapped.to_parquet(out)
    logger.info(f"[{split}] 已写入 {out} ({len(mapped)} 条)")
    logger.info("===== MATH500 转换完成 =====")


if __name__ == "__main__":
    main()
