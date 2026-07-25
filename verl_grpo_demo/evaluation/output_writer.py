"""Persist eval outputs to disk.

Writes three artifacts per run into `<output_dir>/<timestamp>_<tag>/`:
    metrics.json        : per-eval-set aggregated metrics (small JSON, human-readable)
    per_sample.parquet  : one row per (sample, completion) with correctness /
                          format_ok / total_reward / extracted_answer (for debugging)
    run_meta.json       : yaml sha256, model path, timestamp, tag (for provenance)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.aggregator import EvalSetMetrics, extract_answer
from evaluation.config import EvalConfig
from evaluation.scorer import ScoredGeneration


def write_metrics_json(
    metrics_by_eval_set: Mapping[str, EvalSetMetrics],
    output_dir: Path,
) -> Path:
    """Write per-eval-set metrics as human-readable JSON."""
    output_path = output_dir / "metrics.json"
    serializable_payload = {
        eval_set_name: asdict(metrics)
        for eval_set_name, metrics in metrics_by_eval_set.items()
    }
    output_path.write_text(
        json.dumps(serializable_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def write_run_meta_json(
    config: EvalConfig,
    model_path: str,
    tag: str,
    output_dir: Path,
) -> Path:
    """Provenance record so we can prove which frozen config a run used."""
    output_path = output_dir / "run_meta.json"
    payload = {
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "config_version": config.version,
        "config_frozen_at": config.frozen_at,
        "config_yaml_path": str(config.yaml_path),
        "config_yaml_sha256": config.yaml_sha256,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def write_per_sample_parquet(
    scored_by_eval_set: Mapping[str, list[ScoredGeneration]],
    output_dir: Path,
) -> Path:
    """One row per (sample, completion) pair, greedy + all k sampled.

    Columns:
        eval_set, sample_index, data_source, ground_truth,
        completion_kind ('greedy' | 'sampled'), completion_index,
        completion_text, extracted_answer,
        correctness, format_ok, total_reward
    """
    output_path = output_dir / "per_sample.parquet"
    rows: list[dict] = []

    for eval_set_name, scored_list in scored_by_eval_set.items():
        for scored in scored_list:
            # Greedy row (completion_index = -1 to distinguish from sampled 0..k-1).
            rows.append({
                "eval_set": eval_set_name,
                "sample_index": scored.index,
                "data_source": scored.data_source,
                "ground_truth": scored.ground_truth,
                "completion_kind": "greedy",
                "completion_index": -1,
                "completion_text": scored.greedy_completion,
                "extracted_answer": extract_answer(
                    scored.data_source, scored.greedy_completion
                ) or "",
                "correctness": scored.greedy_score.correctness,
                "format_ok": scored.greedy_score.format_ok,
                "total_reward": scored.greedy_score.total_reward,
            })
            # Sampled rows.
            for completion_index, (completion_text, completion_score) in enumerate(
                zip(scored.sampled_completions, scored.sampled_scores)
            ):
                rows.append({
                    "eval_set": eval_set_name,
                    "sample_index": scored.index,
                    "data_source": scored.data_source,
                    "ground_truth": scored.ground_truth,
                    "completion_kind": "sampled",
                    "completion_index": completion_index,
                    "completion_text": completion_text,
                    "extracted_answer": extract_answer(
                        scored.data_source, completion_text
                    ) or "",
                    "correctness": completion_score.correctness,
                    "format_ok": completion_score.format_ok,
                    "total_reward": completion_score.total_reward,
                })

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output_path))
    return output_path


def write_all_outputs(
    config: EvalConfig,
    model_path: str,
    tag: str,
    metrics_by_eval_set: Mapping[str, EvalSetMetrics],
    scored_by_eval_set: Mapping[str, list[ScoredGeneration]],
    output_dir: Path,
) -> dict[str, Path]:
    """Create <output_dir>/<timestamp>_<tag>/ and write all three artifacts."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "metrics_json": write_metrics_json(metrics_by_eval_set, run_dir),
        "per_sample_parquet": write_per_sample_parquet(scored_by_eval_set, run_dir),
        "run_meta_json": write_run_meta_json(config, model_path, tag, run_dir),
    }