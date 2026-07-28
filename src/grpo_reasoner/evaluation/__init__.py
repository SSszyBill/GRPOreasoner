"""Frozen-criteria evaluation harness for GRPO experiments."""

from grpo_reasoner.evaluation.config import EvalConfig, load_eval_config
from grpo_reasoner.evaluation.data import EvalSample, load_eval_data
from grpo_reasoner.evaluation.generator import (
    SampleGeneration,
    build_greedy_sampling_params,
    build_sampled_sampling_params,
    build_vllm_engine,
    run_generation,
)
from grpo_reasoner.evaluation.scorer import (
    ScoredCompletion,
    ScoredGeneration,
    score_completion,
    score_generations,
)
from grpo_reasoner.evaluation.aggregator import (
    EvalSetMetrics,
    aggregate_eval_set,
    compute_format_rate,
    compute_maj_at_k,
    compute_pass_at_1,
    compute_total_reward_greedy,
    compute_total_reward_sampled_mean,
    extract_answer,
)
from grpo_reasoner.evaluation.output_writer import (
    write_all_outputs,
    write_metrics_json,
    write_per_sample_parquet,
    write_run_meta_json,
)

__all__ = [
    "EvalConfig", "load_eval_config",
    "EvalSample", "load_eval_data",
    "SampleGeneration", "build_vllm_engine", "build_greedy_sampling_params",
    "build_sampled_sampling_params", "run_generation",
    "ScoredCompletion", "ScoredGeneration", "score_completion", "score_generations",
    "EvalSetMetrics", "aggregate_eval_set",
    "compute_pass_at_1", "compute_maj_at_k", "compute_format_rate",
    "compute_total_reward_greedy", "compute_total_reward_sampled_mean",
    "extract_answer",
    "write_all_outputs", "write_metrics_json",
    "write_per_sample_parquet", "write_run_meta_json",
]

