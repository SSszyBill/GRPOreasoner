# Inspect the candidate pool produced by build_candidate_pool.py.
# Reads data/candidate/*.parquet and reports counts + NuminaMath subsource distribution.
# Run: python scripts/inspect_candidate.py
import os
import argparse
from collections import Counter

import pyarrow.parquet as pq


def count_rows(path):
    return pq.ParquetFile(path).metadata.num_rows


def subsource_dist(path, batch_size=2000):
    c = Counter()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["extra_info"]):
        for ei in batch.column("extra_info").to_pylist():
            sub = ei.get("subsource", "unknown") if isinstance(ei, dict) else "unknown"
            c[sub] += 1
    return c


def main():
    home = os.path.expanduser("~/autodl-tmp/rl_playground")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{home}/data/candidate")
    args = ap.parse_args()

    files = {
        "gsm8k": os.path.join(args.dir, "gsm8k_train.parquet"),
        "numinamath": os.path.join(args.dir, "numinamath_train.parquet"),
        "countdown": os.path.join(args.dir, "countdown_train.parquet"),
    }
    total = 0
    for name, p in files.items():
        if not os.path.exists(p):
            print("MISSING     ", name, p)
            continue
        n = count_rows(p)
        total += n
        print("%-12s %8d  %s" % (name, n, p))
    print("-" * 50)
    print("TOTAL %d" % total)

    np_path = files["numinamath"]
    if os.path.exists(np_path):
        print("---- NuminaMath subsource distribution ----")
        dist = subsource_dist(np_path)
        sub_total = sum(dist.values())
        for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * v / sub_total if sub_total else 0.0
            print("  %-18s %6d  (%.1f%%)" % (k, v, pct))


if __name__ == "__main__":
    main()