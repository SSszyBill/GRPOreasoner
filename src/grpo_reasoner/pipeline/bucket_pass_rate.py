"""Difficulty bucketing helpers for candidate pools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from grpo_reasoner.rewards import reward_multi


@dataclass(frozen=True)
class CandidateSource:
    """One candidate parquet and its reward-router data_source key."""
    name: str
    data_source: str
    input_filename: str
    output_filename: str


CANDIDATE_SOURCES: tuple[CandidateSource, ...] = (
    CandidateSource("gsm8k", "openai/gsm8k", "gsm8k_train.parquet", "gsm8k_train_pass_rate.parquet"),
    CandidateSource("numinamath", "numina_math", "numinamath_train.parquet", "numinamath_train_pass_rate.parquet"),
    CandidateSource("countdown", "countdown", "countdown_train.parquet", "countdown_train_pass_rate.parquet"),
)


def iter_candidate_rows(parquet_path: Path, batch_size: int = 512) -> Iterator[dict]:
    """Yield rows one at a time from a verl 5-field parquet."""
    parquet_file = pq.ParquetFile(str(parquet_path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def build_prompt_text(tokenizer, prompt_messages: Sequence[dict]) -> str:
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def compute_correctness(
    data_source: str,
    completion: str,
    ground_truth: str,
    extra_info: dict | None,
) -> float:
    return reward_multi.compute_score(
        data_source,
        completion,
        ground_truth,
        extra_info=extra_info,
        format_score=0.0,
        score=1.0,
    )


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
    """Bucket one source and write the pass-rate parquet."""
    rows: list[dict] = list(iter_candidate_rows(input_path))
    if limit is not None:
        rows = rows[:limit]

    prompts: list[str] = [build_prompt_text(tokenizer, row["prompt"]) for row in rows]
    request_outputs = vllm_engine.generate(prompts, sampling_params)
    assert len(request_outputs) == len(prompts)

    correct_counts: list[int] = []
    pass_rates: list[float] = []
    for row, request_output in zip(rows, request_outputs):
        ground_truth = row["reward_model"]["ground_truth"]
        extra_info = row.get("extra_info")
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

    bucket_too_easy = sum(1 for p in pass_rates if p == 1.0)
    bucket_too_hard = sum(1 for p in pass_rates if p == 0.0)
    bucket_keep = sum(1 for p in pass_rates if 0.0 < p < 1.0)
    histogram_by_correct_count = {
        str(c): sum(1 for cc in correct_counts if cc == c)
        for c in range(k + 1)
    }
    return {
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

