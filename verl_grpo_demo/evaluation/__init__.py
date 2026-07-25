"""Frozen-criteria evaluation harness for GRPO experiments.

Public API:
    - EvalConfig, load_eval_config     : load and validate eval_frozen.yaml
    - EvalSample, load_eval_data       : parquet -> chat-templated samples
    - SampleGeneration, run_generation,
      build_vllm_engine,
      build_greedy_sampling_params,
      build_sampled_sampling_params    : vLLM offline batch generation

Later sub-steps will add:
    - Scorer      : route each generation through utils.reward_multi
    - Aggregator  : pass@1 / maj@8 / format_rate / total_reward
    - OutputWriter: dump metrics.json / per_sample.parquet / run_meta.json
"""
from evaluation.config import EvalConfig, load_eval_config
from evaluation.data import EvalSample, load_eval_data
from evaluation.generator import (
    SampleGeneration,
    build_greedy_sampling_params,
    build_sampled_sampling_params,
    build_vllm_engine,
    run_generation,
)
from evaluation.scorer import (
    ScoredCompletion,
    ScoredGeneration,
    score_completion,
    score_generations,
)
from evaluation.aggregator import (
    EvalSetMetrics,
    aggregate_eval_set,
    compute_format_rate,
    compute_maj_at_k,
    compute_pass_at_1,
    compute_total_reward_greedy,
    compute_total_reward_sampled_mean,
    extract_answer,
)
from evaluation.output_writer import (
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