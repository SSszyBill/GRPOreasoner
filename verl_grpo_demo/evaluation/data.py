"""Load parquet eval sets and apply chat template.

Uses pyarrow streaming batches so this stays OOM-safe even on the largest
training parquet (763k rows on NuminaMath).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evaluation.config import EvalConfig


@dataclass(frozen=True)
class EvalSample:
    """One evaluation sample, chat template already applied to prompt_text."""
    index: int
    data_source: str
    prompt_text: str
    ground_truth: str
    raw_question: str


def load_eval_data(
    config: EvalConfig,
    tokenizer,
    working_dir: str | Path,
) -> dict[str, list[EvalSample]]:
    """Load all eval sets defined in config, apply chat template to prompts.

    Returns {eval_set_name: [EvalSample, ...]}.
    """
    import pyarrow.parquet as pq

    resolved_working_dir = Path(working_dir).resolve()
    eval_data_by_set: dict[str, list[EvalSample]] = {}

    for eval_set in config.eval_sets:
        parquet_path = resolved_working_dir / eval_set["path"]
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"eval parquet not found: {parquet_path}\n"
                f"  (working_dir={resolved_working_dir}, yaml path={eval_set['path']})\n"
                f"  Hint: pass --working-dir /root/autodl-tmp/rl_playground"
            )

        parquet_file = pq.ParquetFile(str(parquet_path))
        samples: list[EvalSample] = []
        for row_batch in parquet_file.iter_batches(batch_size=256):
            for row_index in range(row_batch.num_rows):
                data_source = row_batch.column("data_source")[row_index].as_py()
                messages = row_batch.column("prompt")[row_index].as_py()
                reward_model_entry = row_batch.column("reward_model")[row_index].as_py()
                extra_info = row_batch.column("extra_info")[row_index].as_py()

                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                extra_info_dict = extra_info if isinstance(extra_info, dict) else {}
                samples.append(EvalSample(
                    index=extra_info_dict.get("index", len(samples)),
                    data_source=data_source,
                    prompt_text=prompt_text,
                    ground_truth=str(reward_model_entry["ground_truth"]),
                    raw_question=(
                        extra_info_dict.get("question")
                        or extra_info_dict.get("problem")
                        or ""
                    ),
                ))

        # Row count check catches truncated / partially-written parquet.
        expected_size = eval_set["size"]
        if len(samples) != expected_size:
            print(f"[loader][WARN] {eval_set['name']}: expected {expected_size}, got {len(samples)}")
        else:
            print(f"[loader] {eval_set['name']}: {len(samples)} samples OK")

        # Data source check catches wrong parquet loaded under this yaml entry.
        unique_data_sources = set(sample.data_source for sample in samples)
        expected_data_source = eval_set["data_source"]
        if unique_data_sources != {expected_data_source}:
            print(
                f"[loader][WARN] {eval_set['name']}: data_source={unique_data_sources}, "
                f"expected={expected_data_source!r}"
            )

        eval_data_by_set[eval_set["name"]] = samples

    return eval_data_by_set