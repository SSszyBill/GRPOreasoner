# -*- coding: utf-8 -*-
"""NuminaMath-CoT -> verl RL parquet (aligned with official gsm8k.py).
Keeps only samples with a single verifiable boxed answer (parseability filter).
Writes a log file + clean_stats.json in addition to console output.
Output schema: data_source / prompt / ability / reward_model / extra_info
"""
import os
import json
import logging
import argparse
from datetime import datetime
from collections import Counter

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DATA_SOURCE = "numina_math"
ABILITY = "math"


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("preprocess_numina")
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


def extract_all_boxed(text):
    key = r"\boxed"
    out, i = [], 0
    while True:
        idx = text.find(key, i)
        if idx == -1:
            break
        j = idx + len(key)
        while j < len(text) and text[j] == " ":
            j += 1
        if j >= len(text) or text[j] != "{":
            i = idx + len(key)
            continue
        depth, start = 0, j
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(text[start + 1:j])
        i = j + 1
    return out


def normalize_answer(s):
    s = s.strip()
    for tok in [r"\left", r"\right", r"\!", r"\,", r"\;", r"\ ", "$", "\n"]:
        s = s.replace(tok, "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return s.strip()


def extract_ground_truth(solution):
    boxes = extract_all_boxed(solution or "")
    norm = [normalize_answer(b) for b in boxes]
    uniq = set(norm)
    if len(boxes) == 0:
        return None, "no_boxed"
    if len(uniq) > 1:
        return None, "multi_boxed"
    gt = norm[0]
    if gt == "":
        return None, "empty"
    if len(gt) > 100:
        return None, "too_long"
    return gt, "ok"


def process_example(example, idx, split):
    problem = example.get("problem", "") or ""
    solution = example.get("solution", "") or ""
    subsource = example.get("source", "") or ""
    gt, reason = extract_ground_truth(solution)
    keep = gt is not None
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": (problem + " " + INSTRUCTION) if keep else ""}],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": gt or ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": problem,
            "answer": solution,
            "subsource": subsource,
        },
        "_keep": keep,
        "_reason": reason,
        "_subsource": subsource,
    }


def compute_stats(split, n_total, keeps, subs, reasons):
    """汇总一个 split 的清洗统计，返回可序列化 dict。"""
    n_keep = sum(1 for k in keeps if k)
    total_by_src = Counter(subs)
    kept_by_src = Counter(s for s, k in zip(subs, keeps) if k)
    drop_reasons = Counter(r for r in reasons if r != "ok")
    per_sub = {}
    for src in sorted(total_by_src, key=lambda x: -total_by_src[x]):
        t, kk = total_by_src[src], kept_by_src[src]
        per_sub[src] = {"total": t, "kept": kk, "rate": round(kk / t, 4) if t else 0.0}
    return {
        "split": split,
        "original": n_total,
        "kept": n_keep,
        "dropped": n_total - n_keep,
        "retention": round(n_keep / n_total, 4) if n_total else 0.0,
        "drop_reasons": dict(drop_reasons),
        "per_subsource": per_sub,
    }


def log_stats(logger, stats):
    logger.info(f"[{stats['split']}] 原始 {stats['original']} | 保留 {stats['kept']} | 丢弃 {stats['dropped']} | 留存率 {stats['retention']:.1%}")
    logger.info(f"[{stats['split']}] drop 明细: {stats['drop_reasons']}")
    logger.info(f"[{stats['split']}] 各 subsource 留存率:")
    for src, v in stats["per_subsource"].items():
        logger.info(f"    {src:16s} {v['kept']:>7d}/{v['total']:<7d} ({v['rate']:.1%})")


def main():
    import datasets
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/autodl-tmp/rl_playground/hf_cache"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.expanduser("~/autodl-tmp/rl_playground/data/numinamath"))
    parser.add_argument("--log_dir", default=os.path.expanduser("~/autodl-tmp/rl_playground/verl_grpo_demo/logs"))
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"preprocess_numina_{ts}.log")
    logger = setup_logger(log_path)
    logger.info("===== NuminaMath 清洗开始 =====")
    logger.info(f"数据源: AI-MO/NuminaMath-CoT | 输出目录: {args.local_dir} | 日志: {log_path}")

    ds = datasets.load_dataset("AI-MO/NuminaMath-CoT")
    os.makedirs(args.local_dir, exist_ok=True)
    all_stats = {}

    for split in ds.keys():
        d = ds[split]
        mapped = d.map(lambda ex, idx: process_example(ex, idx, split),
                       with_indices=True, remove_columns=d.column_names,
                       num_proc=args.num_proc, desc=f"processing {split}")
        stats = compute_stats(split, len(d), mapped["_keep"], mapped["_subsource"], mapped["_reason"])
        log_stats(logger, stats)
        all_stats[split] = stats

        kept = mapped.filter(lambda x: x["_keep"]).remove_columns(["_keep", "_reason", "_subsource"])
        out = os.path.join(args.local_dir, f"{split}.parquet")
        kept.to_parquet(out)
        logger.info(f"[{split}] 已写入 {out} ({len(kept)} 条)")

    stats_path = os.path.join(args.local_dir, "clean_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计已写入 {stats_path}")
    logger.info("===== NuminaMath 清洗完成 =====")


if __name__ == "__main__":
    main()
