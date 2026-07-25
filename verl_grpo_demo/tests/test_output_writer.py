"""pytest tests for evaluation.output_writer.

Uses pytest's tmp_path fixture to avoid writing to real experiment folders.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from evaluation.aggregator import EvalSetMetrics
from evaluation.config import EvalConfig
from evaluation.output_writer import (
    write_all_outputs,
    write_metrics_json,
    write_per_sample_parquet,
    write_run_meta_json,
)
from evaluation.scorer import ScoredCompletion, ScoredGeneration


@pytest.fixture
def fake_metrics() -> dict[str, EvalSetMetrics]:
    return {
        "gsm8k_test": EvalSetMetrics(
            eval_set_name="gsm8k_test",
            data_source="openai/gsm8k",
            num_samples=3,
            pass_at_1=0.667,
            maj_at_8=0.667,
            format_rate=0.667,
            total_reward_greedy=0.733,
            total_reward_sampled_mean=0.808,
        ),
    }


@pytest.fixture
def fake_scored() -> dict[str, list[ScoredGeneration]]:
    generation = ScoredGeneration(
        index=0,
        data_source="openai/gsm8k",
        ground_truth="42",
        greedy_completion="reason\n#### 42",
        sampled_completions=("a\n#### 42", "b\n#### 41"),
        greedy_score=ScoredCompletion(1.0, 1.0, 1.1),
        sampled_scores=(
            ScoredCompletion(1.0, 1.0, 1.1),
            ScoredCompletion(0.0, 1.0, 0.1),
        ),
    )
    return {"gsm8k_test": [generation]}


@pytest.fixture
def fake_config(tmp_path) -> EvalConfig:
    fake_yaml = tmp_path / "fake_eval_frozen.yaml"
    fake_yaml.write_text("version: v1\n")
    return EvalConfig(
        version="v1",
        frozen_at="2026-07-03",
        eval_sets=[],
        metrics={},
        sampling={},
        generation={},
        scoring={},
        engine={},
        yaml_path=fake_yaml,
        yaml_sha256="deadbeef" * 8,
        raw={},
    )


def test_write_metrics_json(tmp_path, fake_metrics):
    output_path = write_metrics_json(fake_metrics, tmp_path)
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"gsm8k_test"}
    assert payload["gsm8k_test"]["num_samples"] == 3
    assert payload["gsm8k_test"]["pass_at_1"] == pytest.approx(0.667)


def test_write_run_meta_json(tmp_path, fake_config):
    output_path = write_run_meta_json(
        config=fake_config,
        model_path="/some/model/path",
        tag="my_run",
        output_dir=tmp_path,
    )
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["tag"] == "my_run"
    assert payload["model_path"] == "/some/model/path"
    assert payload["config_yaml_sha256"] == "deadbeef" * 8
    assert payload["config_version"] == "v1"


def test_write_per_sample_parquet(tmp_path, fake_scored):
    output_path = write_per_sample_parquet(fake_scored, tmp_path)
    assert output_path.exists()
    table = pq.read_table(str(output_path))
    # 1 greedy + 2 sampled = 3 rows total
    assert table.num_rows == 3
    columns = set(table.column_names)
    expected_columns = {
        "eval_set", "sample_index", "data_source", "ground_truth",
        "completion_kind", "completion_index", "completion_text",
        "extracted_answer", "correctness", "format_ok", "total_reward",
    }
    assert expected_columns.issubset(columns)


def test_write_all_outputs_creates_run_subdir(
    tmp_path, fake_config, fake_metrics, fake_scored,
):
    written = write_all_outputs(
        config=fake_config,
        model_path="/some/model",
        tag="my_run",
        metrics_by_eval_set=fake_metrics,
        scored_by_eval_set=fake_scored,
        output_dir=tmp_path,
    )
    # All three files exist under a <timestamp>_my_run/ subdir.
    assert set(written.keys()) == {"metrics_json", "per_sample_parquet", "run_meta_json"}
    for path in written.values():
        assert Path(path).exists()
        assert path.parent.name.endswith("_my_run")