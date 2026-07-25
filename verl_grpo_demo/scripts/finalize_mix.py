#!/usr/bin/env python
"""Finalize the training mix from bucketed candidate pools.

Reads data/bucketed/*_pass_rate.parquet + configs/data.yaml, filters by
pass_rate_range, samples per-source `take`, shuffles with seed, writes:
    <out-dir>/train.parquet
    <out-dir>/stats.json      per-source counts + kept-histogram
    <out-dir>/manifest.json   yaml sha256 + seed + timestamp + counts

Design notes:
    - Config lives in YAML (versioned with the experiment), not CLI args.
    - Default filter is strict 0<p<1 (both extremes give zero GRPO gradient).
    - `take: N` errors if not enough post-filter samples, unless --allow-shortage.
    - Output preserves verl 5 fields + pass_rate / correct_count / k_samples
      so post-training analysis can slice by difficulty without re-joining.
    - All sampling / shuffling is deterministic under (config.seed, source).

Usage (dry run: load, filter, count, print plan; no writes):
    python -m scripts.finalize_mix \\
        --config       experiments/4090_grpo_multisource_v1/configs/data.yaml \\
        --bucketed-dir data/bucketed \\
        --out-dir      data/train_final \\
        --working-dir  /root/autodl-tmp/rl_playground \\
        --dry-run

Usage (real run):
    python -m scripts.finalize_mix \\
        --config       experiments/4090_grpo_multisource_v1/configs/data.yaml \\
        --bucketed-dir data/bucketed \\
        --out-dir      data/train_final \\
        --working-dir  /root/autodl-tmp/rl_playground
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# ---------------------------------------------------------------------------
# Source registry (mirrors bucket_pass_rate.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BucketedSource:
    """Where each source's pass_rate parquet lives."""
    name: str
    data_source: str
    input_filename: str


BUCKETED_SOURCES: tuple[BucketedSource, ...] = (
    BucketedSource("gsm8k",      "openai/gsm8k", "gsm8k_train_pass_rate.parquet"),
    BucketedSource("numinamath", "numina_math",  "numinamath_train_pass_rate.parquet"),
    BucketedSource("countdown",  "countdown",    "countdown_train_pass_rate.parquet"),
)

SOURCE_NAMES: frozenset[str] = frozenset(s.name for s in BUCKETED_SOURCES)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PassRateRange:
    """Half-open filter (min_exclusive, max_exclusive)."""
    min_exclusive: float
    max_exclusive: float

    def matches(self, pass_rate: float) -> bool:
        return self.min_exclusive < pass_rate < self.max_exclusive


@dataclass(frozen=True)
class SourceTakeSpec:
    """How many rows to take from one source after filtering."""
    name: str
    take: int | None  # None means "all after filter"


@dataclass(frozen=True)
class MixConfig:
    """Full training-mix config, loaded from YAML."""
    version: str
    seed: int
    pass_rate_range: PassRateRange
    source_takes: tuple[SourceTakeSpec, ...]
    yaml_path: Path
    yaml_sha256: str


def load_mix_config(config_path: Path) -> MixConfig:
    """Load and validate configs/data.yaml.

    Fails fast on unknown source names, negative takes, or bad pass_rate range.
    Better a loud error here than silent nonsense in the training data.
    """
    raw_bytes = config_path.read_bytes()
    yaml_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = yaml.safe_load(raw_bytes)

    version = str(raw.get("version", "v1"))
    seed = int(raw.get("seed", 42))

    # pass_rate_range
    range_raw = raw.get("pass_rate_range", {})
    pass_rate_range = PassRateRange(
        min_exclusive=float(range_raw.get("min_exclusive", 0.0)),
        max_exclusive=float(range_raw.get("max_exclusive", 1.0)),
    )
    if not (0.0 <= pass_rate_range.min_exclusive
            < pass_rate_range.max_exclusive <= 1.0):
        raise ValueError(
            f"pass_rate_range invalid: {pass_rate_range} "
            f"(need 0 <= min_exclusive < max_exclusive <= 1)"
        )

    # sources
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, dict) or not sources_raw:
        raise ValueError("config.sources must be a non-empty dict")

    unknown = set(sources_raw.keys()) - SOURCE_NAMES
    if unknown:
        raise ValueError(
            f"config.sources has unknown source names {sorted(unknown)}; "
            f"known: {sorted(SOURCE_NAMES)}"
        )

    source_takes: list[SourceTakeSpec] = []
    for source_name, spec in sources_raw.items():
        take_raw = spec.get("take") if isinstance(spec, dict) else spec
        if take_raw == "all" or take_raw is None:
            take: int | None = None
        else:
            take = int(take_raw)
            if take <= 0:
                raise ValueError(
                    f"config.sources.{source_name}.take must be 'all' or "
                    f"a positive int, got {take_raw!r}"
                )
        source_takes.append(SourceTakeSpec(name=source_name, take=take))

    return MixConfig(
        version=version,
        seed=seed,
        pass_rate_range=pass_rate_range,
        source_takes=tuple(source_takes),
        yaml_path=config_path,
        yaml_sha256=yaml_sha256,
    )


# ---------------------------------------------------------------------------
# Streaming parquet load
# ---------------------------------------------------------------------------

def iter_rows(parquet_path: Path, batch_size: int = 512) -> Iterator[dict]:
    parquet_file = pq.ParquetFile(str(parquet_path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


# ---------------------------------------------------------------------------
# Per-source finalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceResult:
    name: str
    data_source: str
    input_path: Path
    n_pool: int                       # rows in bucketed parquet
    n_after_filter: int               # after pass_rate range filter
    n_taken: int                      # after take-N
    take_requested: int | None        # None if "all"
    was_shortage: bool                # take_requested > n_after_filter
    pass_rate_hist_taken: dict[str, int]   # histogram over correct_count in taken
    rows: list[dict]                  # actual taken rows (pre-shuffle)


def finalize_one_source(
    source: BucketedSource,
    input_path: Path,
    take: int | None,
    pass_rate_range: PassRateRange,
    seed: int,
    allow_shortage: bool,
) -> SourceResult:
    """Load, filter by pass_rate_range, sample `take` rows deterministically."""
    print(f"\n[finalize] === {source.name} ({source.data_source}) ===")
    print(f"[finalize] reading: {input_path}")

    pool_rows: list[dict] = list(iter_rows(input_path))
    n_pool = len(pool_rows)

    filtered_rows = [
        row for row in pool_rows if pass_rate_range.matches(row["pass_rate"])
    ]
    n_after_filter = len(filtered_rows)

    print(f"[finalize] {source.name}: pool={n_pool}  "
          f"after_filter({pass_rate_range.min_exclusive}<p<"
          f"{pass_rate_range.max_exclusive})={n_after_filter}")

    was_shortage = False
    if take is None:
        taken_rows = filtered_rows
    else:
        if take > n_after_filter:
            was_shortage = True
            msg = (
                f"[finalize] {source.name}: requested take={take} but only "
                f"{n_after_filter} rows available after filter"
            )
            if not allow_shortage:
                print(msg + "  (pass --allow-shortage to take all available)")
                sys.exit(3)
            print(msg + "  --allow-shortage set, taking all available")
            taken_rows = filtered_rows
        else:
            # Per-source rng: derived from (seed, name) so changing seed
            # reshuffles all sources; string seed is stable across Python runs.
            per_source_rng = random.Random(f"{seed}:{source.name}")
            taken_rows = per_source_rng.sample(filtered_rows, take)

    n_taken = len(taken_rows)
    print(f"[finalize] {source.name}: taken={n_taken}"
          + ("  [SHORTAGE]" if was_shortage else ""))

    # Histogram of correct_count in taken rows.
    # k should be uniform per source (all bucketing used same k for this source).
    k = taken_rows[0]["k_samples"] if taken_rows else 0
    pass_rate_hist_taken = {
        str(c): sum(1 for r in taken_rows if r["correct_count"] == c)
        for c in range(k + 1)
    }

    return SourceResult(
        name=source.name,
        data_source=source.data_source,
        input_path=input_path,
        n_pool=n_pool,
        n_after_filter=n_after_filter,
        n_taken=n_taken,
        take_requested=take,
        was_shortage=was_shortage,
        pass_rate_hist_taken=pass_rate_hist_taken,
        rows=taken_rows,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.finalize_mix",
        description="Finalize training mix from bucketed candidate pools.",
    )
    parser.add_argument("--config", required=True,
                        help="Path to configs/data.yaml (resolved from CWD if relative)")
    parser.add_argument("--bucketed-dir", default="data/bucketed",
                        help="Dir containing *_pass_rate.parquet")
    parser.add_argument("--out-dir", default="data/train_final",
                        help="Output dir for train.parquet + stats.json + manifest.json")
    parser.add_argument("--working-dir", default=".",
                        help="Base dir that relative --bucketed-dir / --out-dir are joined onto.")
    parser.add_argument("--allow-shortage", action="store_true",
                        help="If a source's take > available, take all available instead of erroring.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load, filter, count — but do not write any output.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    # Config path resolves from CWD (config typically lives in verl_grpo_demo/,
    # while data lives under rl_playground/ — they have different roots).
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    working_dir = Path(args.working_dir).resolve()
    bucketed_dir = (working_dir / args.bucketed_dir).resolve()
    out_dir = (working_dir / args.out_dir).resolve()
    return config_path, bucketed_dir, out_dir


def main() -> None:
    args = parse_args()
    config_path, bucketed_dir, out_dir = resolve_paths(args)

    print(f"[finalize] config       = {config_path}")
    print(f"[finalize] bucketed_dir = {bucketed_dir}")
    print(f"[finalize] out_dir      = {out_dir}")

    if not config_path.exists():
        print(f"[finalize] ERROR: config not found at {config_path}")
        sys.exit(2)

    config = load_mix_config(config_path)
    print(f"[finalize] config version={config.version}  seed={config.seed}  "
          f"sha256={config.yaml_sha256[:16]}...")
    print(f"[finalize] pass_rate_range: "
          f"{config.pass_rate_range.min_exclusive} < p < "
          f"{config.pass_rate_range.max_exclusive}")

    # Validate all requested source input files exist before doing any work.
    source_by_name = {s.name: s for s in BUCKETED_SOURCES}
    missing: list[str] = []
    for spec in config.source_takes:
        source = source_by_name[spec.name]
        input_path = bucketed_dir / source.input_filename
        if not input_path.exists():
            missing.append(str(input_path))
    if missing:
        print("[finalize] ERROR: missing bucketed parquets:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(2)
    print(f"[finalize] all {len(config.source_takes)} bucketed parquets found.")

    # Finalize each source.
    results: list[SourceResult] = []
    for spec in config.source_takes:
        source = source_by_name[spec.name]
        input_path = bucketed_dir / source.input_filename
        result = finalize_one_source(
            source=source,
            input_path=input_path,
            take=spec.take,
            pass_rate_range=config.pass_rate_range,
            seed=config.seed,
            allow_shortage=args.allow_shortage,
        )
        results.append(result)

    # Concatenate and shuffle.
    all_rows: list[dict] = [row for result in results for row in result.rows]
    rng = random.Random(config.seed)
    rng.shuffle(all_rows)
    total = len(all_rows)

    # Rollup print.
    print("\n[finalize] === rollup ===")
    for result in results:
        take_str = "all" if result.take_requested is None else str(result.take_requested)
        shortage_mark = "  [SHORTAGE]" if result.was_shortage else ""
        print(
            f"  {result.name:12s}  pool={result.n_pool:6d}  "
            f"filter={result.n_after_filter:6d}  "
            f"take({take_str})={result.n_taken:6d}  "
            f"({result.n_taken / max(1, total):.1%}){shortage_mark}"
        )
    print(f"  TOTAL         final={total}")

    if args.dry_run:
        print("\n[finalize] --dry-run set, no output written.")
        return

    # Write outputs.
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    table = pa.Table.from_pylist(all_rows)
    pq.write_table(table, str(train_path))
    print(f"\n[finalize] wrote: {train_path}")

    stats = {
        "total_rows": total,
        "sources": [
            {
                "name": r.name,
                "data_source": r.data_source,
                "n_pool": r.n_pool,
                "n_after_filter": r.n_after_filter,
                "n_taken": r.n_taken,
                "take_requested": r.take_requested,
                "was_shortage": r.was_shortage,
                "pass_rate_hist_taken": r.pass_rate_hist_taken,
                "input_path": str(r.input_path),
            }
            for r in results
        ],
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[finalize] wrote: {out_dir / 'stats.json'}")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_yaml_sha256": config.yaml_sha256,
        "config_version": config.version,
        "seed": config.seed,
        "pass_rate_range": {
            "min_exclusive": config.pass_rate_range.min_exclusive,
            "max_exclusive": config.pass_rate_range.max_exclusive,
        },
        "bucketed_dir": str(bucketed_dir),
        "out_dir": str(out_dir),
        "train_parquet": str(train_path),
        "total_rows": total,
        "allow_shortage_flag": args.allow_shortage,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[finalize] wrote: {out_dir / 'manifest.json'}")

    print("\n[finalize] done. next: 用 train.parquet 起 GRPO Session B。")


if __name__ == "__main__":
    main()