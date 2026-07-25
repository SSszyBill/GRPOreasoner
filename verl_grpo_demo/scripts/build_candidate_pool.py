# -*- coding: utf-8 -*-
"""构建候选池（粗下采样）：为难度分桶做准备。
- GSM8K：全留（最小、锚数据）
- NuminaMath：按 extra_info.subsource 分层下采样到 --numina 条
- Countdown：均匀随机下采样到 --countdown 条
全程 pyarrow 分批读写，低内存，保留 verl 五字段 arrow schema。
输出到 {out_dir}/{gsm8k,numinamath,countdown}_train.parquet
"""
import os
import random
import shutil
import logging
import argparse
from collections import defaultdict
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("build_candidate_pool")
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


def _write_selected(src_path, out_path, selected, batch_size, logger):
    """第二遍：按 selected（全局行号集合）流式过滤写出。"""
    pf = pq.ParquetFile(src_path)
    writer = pq.ParquetWriter(out_path, pf.schema_arrow)
    gidx = 0
    kept = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        n_rows = batch.num_rows
        mask = [(gidx + i) in selected for i in range(n_rows)]
        gidx += n_rows
        if any(mask):
            tbl = pa.Table.from_batches([batch])
            filtered = tbl.filter(pa.array(mask, type=pa.bool_()))
            if filtered.num_rows > 0:
                writer.write_table(filtered)
                kept += filtered.num_rows
    writer.close()
    logger.info(f"  写出 {kept} 条 -> {out_path}")
    return kept


def subsample_uniform(src_path, out_path, n, seed, batch_size, logger):
    pf = pq.ParquetFile(src_path)
    total = pf.metadata.num_rows
    n = min(n, total)
    rng = random.Random(seed)
    selected = set(rng.sample(range(total), n))
    logger.info(f"[uniform] {src_path}: 总 {total} -> 抽 {n}")
    return _write_selected(src_path, out_path, selected, batch_size, logger)


def subsample_stratified(src_path, out_path, n, seed, batch_size, logger):
    pf = pq.ParquetFile(src_path)
    total = pf.metadata.num_rows
    # 第一遍：只读 extra_info，取每行 subsource
    keys = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=["extra_info"]):
        for ei in batch.column("extra_info").to_pylist():
            sub = ei.get("subsource", "unknown") if isinstance(ei, dict) else "unknown"
            keys.append(sub)
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    rng = random.Random(seed)
    selected = set()
    dist = {}
    for k, idxs in sorted(groups.items()):
        t = min(len(idxs), round(n * len(idxs) / total))
        picked = rng.sample(idxs, t)
        selected.update(picked)
        dist[k] = t
    logger.info(f"[stratified] {src_path}: 总 {total} -> 抽 {len(selected)}")
    logger.info(f"  分层结果(subsource -> 抽取数): {dist}")
    return _write_selected(src_path, out_path, selected, batch_size, logger)


def main():
    home = os.path.expanduser("~/autodl-tmp/rl_playground")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=f"{home}/data")
    parser.add_argument("--out_dir", default=f"{home}/data/candidate")
    parser.add_argument("--log_dir", default=f"{home}/verl_grpo_demo/logs")
    parser.add_argument("--numina", type=int, default=30000)
    parser.add_argument("--countdown", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(os.path.join(args.log_dir, f"build_candidate_pool_{ts}.log"))
    logger.info(f"===== 构建候选池开始 seed={args.seed} numina={args.numina} countdown={args.countdown} =====")

    # GSM8K：全留，直接复制
    gsm8k_src = os.path.join(args.data_root, "gsm8k", "train.parquet")
    gsm8k_out = os.path.join(args.out_dir, "gsm8k_train.parquet")
    shutil.copy(gsm8k_src, gsm8k_out)
    gsm8k_n = pq.ParquetFile(gsm8k_out).metadata.num_rows
    logger.info(f"[copy] GSM8K 全留 {gsm8k_n} 条 -> {gsm8k_out}")

    # NuminaMath：分层下采样
    numina_src = os.path.join(args.data_root, "numinamath", "train_decontam.parquet")
    numina_out = os.path.join(args.out_dir, "numinamath_train.parquet")
    numina_n = subsample_stratified(numina_src, numina_out, args.numina, args.seed, args.batch_size, logger)

    # Countdown：均匀下采样
    cd_src = os.path.join(args.data_root, "countdown", "train.parquet")
    cd_out = os.path.join(args.out_dir, "countdown_train.parquet")
    cd_n = subsample_uniform(cd_src, cd_out, args.countdown, args.seed, args.batch_size, logger)

    logger.info(f"候选池合计: GSM8K {gsm8k_n} + NuminaMath {numina_n} + Countdown {cd_n} = {gsm8k_n + numina_n + cd_n} 条")
    logger.info(f"输出目录: {args.out_dir}")
    logger.info("===== 构建候选池完成 =====")


if __name__ == "__main__":
    main()
