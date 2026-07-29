#!/usr/bin/env python
"""Difficulty bucketing: estimate per-sample pass_rate on the candidate pool.

For each sample in data/candidate/*.parquet, sample k completions with the base
model, judge correctness via utils.reward_multi (correctness weight only), and
write per-source parquet files with three added columns:
    correct_count : int in [0, k]
    k_samples     : int (== k, stored for future auditability)
    pass_rate     : float in [0, 1]

Downstream: filter p==0 (too hard) and p==1 (too easy), keep 0<p<1 for GRPO
training (which needs within-group reward variance to have gradient).

Design notes:
    - Sampling params are DIFFERENT from eval_frozen.yaml on purpose:
      k=8 / T=1.0 / top_p=1.0 / max_new_tokens=1024, aligned with GRPO rollout,
      not baseline evaluation (baseline uses 2048).
    - vLLM engine is built directly here, not via evaluation.build_vllm_engine,
      so bucketing does not depend on EngineConfig internals.
    - Scoring goes through reward_multi.compute_score with score=1.0,
      format_score=0.0. Format has no signal in bucketing.
    - Sources are processed one at a time; each source's parquet and the
      running stats.json are written after that source finishes, so a mid-run
      crash keeps all completed sources.

Usage (no-GPU dry run: validate paths, print plan, no vLLM):
    python -m scripts.bucket_pass_rate \\
        --candidate-dir data/candidate \\
        --out-dir       data/bucketed \\
        --working-dir   /root/autodl-tmp/rl_playground \\
        --dry-run

Usage (GPU smoke, first 100 per source):
    python -m scripts.bucket_pass_rate \\
        --model /root/autodl-tmp/rl_playground/hf_cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323 \\
        --candidate-dir data/candidate \\
        --out-dir       data/bucketed_smoke \\
        --working-dir   /root/autodl-tmp/rl_playground \\
        --limit-per-source 100

Usage (GPU full run):
    python -m scripts.bucket_pass_rate \\
        --model /root/autodl-tmp/rl_playground/hf_cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323 \\
        --candidate-dir data/candidate \\
        --out-dir       data/bucketed \\
        --working-dir   /root/autodl-tmp/rl_playground
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from utils import reward_multi


# ---------------------------------------------------------------------------
# Candidate source registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateSource:
    """One candidate parquet + its reward-router data_source key."""
    name: str             # gsm8k / numinamath / countdown
    data_source: str      # reward router key: openai/gsm8k / numina_math / countdown
    input_filename: str   # under --candidate-dir
    output_filename: str  # under --out-dir


CANDIDATE_SOURCES: tuple[CandidateSource, ...] = (
    CandidateSource(
        name="gsm8k",
        data_source="openai/gsm8k",
        input_filename="gsm8k_train.parquet",
        output_filename="gsm8k_train_pass_rate.parquet",
    ),
    CandidateSource(
        name="numinamath",
        data_source="numina_math",
        input_filename="numinamath_train.parquet",
        output_filename="numinamath_train_pass_rate.parquet",
    ),
    CandidateSource(
        name="countdown",
        data_source="countdown",
        input_filename="countdown_train.parquet",
        output_filename="countdown_train_pass_rate.parquet",
    ),
)


# ---------------------------------------------------------------------------
# Streaming parquet load
# ---------------------------------------------------------------------------

def iter_candidate_rows(
    parquet_path: Path,
    batch_size: int = 512,
) -> Iterator[dict]:
    """Yield rows one at a time from a verl 5-field parquet.

    Streams via ParquetFile.iter_batches to keep RAM bounded (learned the hard
    way with 763k NuminaMath OOM Killed).
    """
    parquet_file = pq.ParquetFile(str(parquet_path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt_text(tokenizer, prompt_messages: Sequence[dict]) -> str:
    """Apply chat template to verl-format prompt messages.

    prompt_messages is the `prompt` column from a verl 5-field parquet:
        [{'role': 'user', 'content': '...'}]
    """
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_correctness(
    data_source: str,
    completion: str,
    ground_truth: str,
    extra_info: dict | None,
) -> float:
    """Pure correctness via reward_multi (format weight = 0)."""
    return reward_multi.compute_score(
        data_source,
        completion,
        ground_truth,
        extra_info=extra_info,
        format_score=0.0,
        score=1.0,
    )


# ---------------------------------------------------------------------------
# Per-source bucketing
# ---------------------------------------------------------------------------

def bucket_one_source(
    source: CandidateSource,
    input_path: Path,
    output_path: Path,
    tokenizer,
    vllm_engine,
    sampling_params,
    k: int,
    limit: int | None,
) -> dict:
    """Bucket one source. Returns a stats dict."""
    print(f"\n[bucket] === {source.name} ({source.data_source}) ===")
    print(f"[bucket] reading: {input_path}")

    # Per-source parquets are small enough to hold prompts in memory (~30k *
    # a few hundred bytes = a few MB). NuminaMath train.parquet is what OOMed
    # before, at 763k; the candidate pool is already down to 30k.
    rows: list[dict] = list(iter_candidate_rows(input_path))
    if limit is not None:
        rows = rows[:limit]
    print(f"[bucket] {source.name}: {len(rows)} samples"
          + (f" (limited from full via --limit-per-source={limit})" if limit else ""))

    prompts: list[str] = [
        build_prompt_text(tokenizer, row["prompt"]) for row in rows
    ]

    # vLLM offline batch: single generate call, n=k per prompt. LLM.generate
    # preserves input order in its returned RequestOutput list, so we can zip
    # rows and outputs directly.
    print(f"[bucket] {source.name}: generating k={k} per sample "
          f"(total completions = {len(prompts) * k}) ...")
    request_outputs = vllm_engine.generate(prompts, sampling_params)
    assert len(request_outputs) == len(prompts), (
        f"vLLM returned {len(request_outputs)} outputs for {len(prompts)} prompts"
    )

    # Score and record pass_rate.
    correct_counts: list[int] = []
    pass_rates: list[float] = []
    for row, request_output in zip(rows, request_outputs):
        ground_truth = row["reward_model"]["ground_truth"]
        extra_info = row.get("extra_info")

        # request_output.outputs has length k (one per sample).
        correct_count = 0
        for completion_output in request_output.outputs:
            correctness = compute_correctness(
                source.data_source,
                completion_output.text,
                ground_truth,
                extra_info,
            )
            if correctness > 0.5:
                correct_count += 1

        correct_counts.append(correct_count)
        pass_rates.append(correct_count / k)

    # Build output table: original 5 fields + 3 new columns.
    output_rows: list[dict] = []
    for row, correct_count, pass_rate in zip(rows, correct_counts, pass_rates):
        output_row = dict(row)
        output_row["correct_count"] = correct_count
        output_row["k_samples"] = k
        output_row["pass_rate"] = pass_rate
        output_rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(output_rows)
    pq.write_table(table, str(output_path))
    print(f"[bucket] {source.name}: wrote {output_path}")

    # Per-source stats.
    bucket_too_easy = sum(1 for p in pass_rates if p == 1.0)
    bucket_too_hard = sum(1 for p in pass_rates if p == 0.0)
    bucket_keep = sum(1 for p in pass_rates if 0.0 < p < 1.0)
    histogram_by_correct_count = {
        str(c): sum(1 for cc in correct_counts if cc == c)
        for c in range(k + 1)
    }
    stats = {
        "name": source.name,
        "data_source": source.data_source,
        "num_samples": len(rows),
        "k": k,
        "bucket_too_easy_p_eq_1": bucket_too_easy,
        "bucket_too_hard_p_eq_0": bucket_too_hard,
        "bucket_keep_0_lt_p_lt_1": bucket_keep,
        "mean_pass_rate": sum(pass_rates) / max(1, len(pass_rates)),
        "histogram_by_correct_count": histogram_by_correct_count,
        "output_path": str(output_path),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.bucket_pass_rate",
        description="Estimate per-sample pass_rate on candidate pool for GRPO difficulty bucketing.",
    )
    parser.add_argument("--model", required=False,
                        help="Base model path or HF repo id; required unless --dry-run.")
    parser.add_argument("--candidate-dir", default="data/candidate",
                        help="Dir containing {gsm8k,numinamath,countdown}_train.parquet")
    parser.add_argument("--out-dir", default="data/bucketed",
                        help="Output dir for *_pass_rate.parquet + stats.json + manifest.json")
    parser.add_argument("--working-dir", default=".",
                        help="Base dir that relative --candidate-dir / --out-dir are joined onto.")
    parser.add_argument("--k", type=int, default=8,
                        help="Samples per prompt (aligned with GRPO rollout group size).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Bucketing uses 1024 (coarse & cheap); baseline uses 2048.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--only",
                        choices=[s.name for s in CANDIDATE_SOURCES],
                        default=None,
                        help="Only process one source (debugging).")
    parser.add_argument("--limit-per-source", type=int, default=None,
                        help="Only take first N samples per source (GPU smoke test).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check inputs exist and print plan; no vLLM, no writes.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    working_dir = Path(args.working_dir).resolve()
    candidate_dir = (working_dir / args.candidate_dir).resolve()
    out_dir = (working_dir / args.out_dir).resolve()
    return candidate_dir, out_dir


def main() -> None:
    args = parse_args()
    candidate_dir, out_dir = resolve_paths(args)

    sources = [
        s for s in CANDIDATE_SOURCES
        if args.only is None or s.name == args.only
    ]

    # Cheap path validation up front — catches typos before any GPU cost.
    print(f"[bucket] candidate_dir = {candidate_dir}")
    print(f"[bucket] out_dir       = {out_dir}")
    missing: list[str] = []
    for source in sources:
        input_path = candidate_dir / source.input_filename
        if not input_path.exists():
            missing.append(str(input_path))
    if missing:
        print("[bucket] ERROR: missing candidate parquets:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(2)
    print(f"[bucket] all {len(sources)} candidate parquets found.")
    print(f"[bucket] plan: k={args.k} T={args.temperature} top_p={args.top_p} "
          f"max_new_tokens={args.max_new_tokens} seed={args.seed}"
          + (f" [limit-per-source={args.limit_per_source}]" if args.limit_per_source else ""))

    if args.dry_run:
        print("[bucket] --dry-run set, exiting before vLLM load.")
        return

    if not args.model:
        print("[bucket] ERROR: --model is required unless --dry-run.")
        sys.exit(2)

    # Deferred import: keep --dry-run cheap and independent of GPU env.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[bucket] tokenizer loaded from {args.model}")

    vllm_engine = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("[bucket] vLLM engine built.")

    sampling_params = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_by_source: dict[str, dict] = {}

    for source in sources:
        input_path = candidate_dir / source.input_filename
        output_path = out_dir / source.output_filename
        source_stats = bucket_one_source(
            source=source,
            input_path=input_path,
            output_path=output_path,
            tokenizer=tokenizer,
            vllm_engine=vllm_engine,
            sampling_params=sampling_params,
            k=args.k,
            limit=args.limit_per_source,
        )
        stats_by_source[source.name] = source_stats

        # Persist stats after each source so partial runs still leave a record.
        (out_dir / "stats.json").write_text(
            json.dumps(stats_by_source, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Manifest: prove which model + sampling params produced this bucketing.
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "candidate_dir": str(candidate_dir),
        "out_dir": str(out_dir),
        "sources": [s.name for s in sources],
        "limit_per_source": args.limit_per_source,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Final rollup.
    print("\n[bucket] === rollup ===")
    total_samples = sum(s["num_samples"] for s in stats_by_source.values())
    total_keep = sum(s["bucket_keep_0_lt_p_lt_1"] for s in stats_by_source.values())
    for name, s in stats_by_source.items():
        print(
            f"  {name:12s}  n={s['num_samples']:6d}  "
            f"too_hard(p=0)={s['bucket_too_hard_p_eq_0']:5d}  "
            f"too_easy(p=1)={s['bucket_too_easy_p_eq_1']:5d}  "
            f"keep(0<p<1)={s['bucket_keep_0_lt_p_lt_1']:5d}  "
            f"mean_p={s['mean_pass_rate']:.3f}"
        )
    print(f"  TOTAL         n={total_samples:6d}  "
          f"keep={total_keep:5d}  ({total_keep / max(1, total_samples):.1%})")

    print("\n[bucket] done. next: 无卡定稿三源配比 → 写 configs/data.yaml")


if __name__ == "__main__":
    main()