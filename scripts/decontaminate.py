# -*- coding: utf-8 -*-
"""NuminaMath 训练集去污染：剔除与评测集（GSM8K-test / MATH500）重叠的题。
口径：exact 规范化匹配 + 13-gram 命中率 > 阈值。
内存友好版：用 pyarrow 分批（streaming）读写，不一次性加载全部。
默认 dry-run（只报告+抽样，不写清洗结果）；--apply 才真正写出。
"""
import os
import re
import json
import logging
import argparse
from datetime import datetime
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("decontaminate")
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


def norm_exact(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def ngrams(toks, n):
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def get_field(extra_info, key):
    if isinstance(extra_info, dict):
        return extra_info.get(key, "") or ""
    return ""


def build_reference(ref_specs, n, logger):
    """ref_specs: [(path, problem_key, tag)]. 读小评测集，返回 exact_ref, gram_to_eval."""
    exact_ref, gram_to_eval = {}, {}
    for path, key, tag in ref_specs:
        tbl = pq.read_table(path, columns=["extra_info"])
        cnt = 0
        for i, row in enumerate(tbl.to_pylist()):
            prob = get_field(row.get("extra_info"), key)
            if not prob:
                continue
            eid = f"{tag}:{i}"
            exact_ref.setdefault(norm_exact(prob), eid)
            for g in ngrams(tokens(prob), n):
                gram_to_eval.setdefault(g, eid)
            cnt += 1
        logger.info(f"  参照集 {tag}: {cnt} 题 (来自 {path})")
    logger.info(f"参照集合计: exact {len(exact_ref)} 条, {n}-gram {len(gram_to_eval)} 个")
    return exact_ref, gram_to_eval


def classify(prob, exact_ref, gram_to_eval, gram_set, n, threshold):
    ne = norm_exact(prob)
    if ne and ne in exact_ref:
        return True, "exact", 1.0, exact_ref[ne]
    grams = ngrams(tokens(prob), n)
    if not grams:
        return False, "", 0.0, None
    matched = [g for g in grams if g in gram_set]
    hit = len(matched) / len(grams)
    if hit > threshold:
        eid = Counter(gram_to_eval[g] for g in matched).most_common(1)[0][0]
        return True, "ngram", hit, eid
    return False, "", hit, None


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    home = os.path.expanduser("~/autodl-tmp/rl_playground")
    parser = argparse.ArgumentParser()
    parser.add_argument("--numina", default=f"{home}/data/numinamath/train.parquet")
    parser.add_argument("--gsm8k_test", default=f"{home}/data/gsm8k/test.parquet")
    parser.add_argument("--math500_test", default=f"{home}/data/math500/test.parquet")
    parser.add_argument("--out", default=f"{home}/data/numinamath/train_decontam.parquet")
    parser.add_argument("--review", default=f"{home}/data/numinamath/decontam_review.jsonl")
    parser.add_argument("--stats", default=f"{home}/data/numinamath/decontam_stats.json")
    parser.add_argument("--log_dir", default=f"{home}/verl_grpo_demo/logs")
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--apply", action="store_true", help="真正写出清洗结果；不加则 dry-run")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"decontaminate_{ts}.log")
    logger = setup_logger(log_path)
    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info(f"===== 去污染开始 [{mode}] n={args.n} threshold={args.threshold} batch={args.batch_size} =====")

    logger.info("构建参照集（评测集题面）...")
    ref_specs = [
        (args.gsm8k_test, "question", "gsm8k_test"),
        (args.math500_test, "problem", "math500"),
    ]
    exact_ref, gram_to_eval = build_reference(ref_specs, args.n, logger)
    gram_set = set(gram_to_eval)
    if not exact_ref:
        logger.warning("参照集为空！检查评测集路径/字段名。")
        return

    pf = pq.ParquetFile(args.numina)
    n_total = pf.metadata.num_rows
    schema = pf.schema_arrow
    logger.info(f"扫描 NuminaMath: {args.numina} (共 {n_total} 条, 分批处理 batch={args.batch_size})")

    writer = None
    read_cols = None
    if args.apply:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        writer = pq.ParquetWriter(args.out, schema)
    else:
        read_cols = ["extra_info"]

    by_type, by_sub = Counter(), Counter()
    flagged_sample = []
    seen, n_keep = 0, 0
    next_log = 100000
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=read_cols):
        ei_col = batch.column("extra_info").to_pylist()
        keep_flags = []
        for ei in ei_col:
            prob = get_field(ei, "question")
            sub = get_field(ei, "subsource")
            is_hit, mtype, hit, eid = classify(prob, exact_ref, gram_to_eval, gram_set, args.n, args.threshold)
            keep_flags.append(not is_hit)
            if is_hit:
                by_type[mtype] += 1
                by_sub[sub] += 1
                if len(flagged_sample) < args.sample:
                    flagged_sample.append({
                        "match_type": mtype, "hit_rate": round(hit, 3),
                        "matched_eval": eid, "subsource": sub,
                        "numina_problem": (prob or "")[:300],
                    })
        n_keep += sum(keep_flags)
        seen += len(keep_flags)
        if args.apply:
            mask = pa.array(keep_flags, type=pa.bool_())
            filtered = batch.filter(mask)
            if filtered.num_rows:
                writer.write_table(pa.Table.from_batches([filtered]))
        if seen >= next_log:
            logger.info(f"  进度 {seen}/{n_total} 已标记 {sum(by_type.values())}")
            next_log += 100000

    if writer is not None:
        writer.close()

    n_flagged = sum(by_type.values())
    logger.info(f"扫描完成：总 {seen} | 命中(污染) {n_flagged} | 保留 {n_keep} ({n_keep/seen:.2%})")
    logger.info(f"命中类型分布: {dict(by_type)}")
    logger.info(f"命中 subsource 分布: {dict(by_sub)}")

    os.makedirs(os.path.dirname(args.review), exist_ok=True)
    with open(args.review, "w", encoding="utf-8") as f:
        for item in flagged_sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"已写入 {len(flagged_sample)} 条命中抽样到 {args.review}（请人工核验有无误杀）")

    stats = {"mode": mode, "n": args.n, "threshold": args.threshold,
             "total": seen, "flagged": n_flagged, "kept": n_keep,
             "by_type": dict(by_type), "by_subsource": dict(by_sub)}
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计已写入 {args.stats}")

    if args.apply:
        logger.info(f"[APPLY] 已写出去污染结果 {args.out} ({n_keep} 条)")
    else:
        logger.info("[DRY-RUN] 未写出清洗结果。核验抽样无误后，加 --apply 再跑一次即可落地。")
    logger.info("===== 去污染完成 =====")


if __name__ == "__main__":
    main()
