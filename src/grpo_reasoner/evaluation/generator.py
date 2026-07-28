"""vLLM-based generation for eval: greedy for pass@1, sampled for maj@8.

vLLM imports are deferred so callers that only need config / data can import
`grpo_reasoner.evaluation` on a CPU-only machine without triggering GPU-side failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from grpo_reasoner.evaluation.config import EvalConfig
from grpo_reasoner.evaluation.data import EvalSample


@dataclass(frozen=True)
class SampleGeneration:
    """One sample's generations: single greedy + k independent samples.

    Kept in one object so downstream scoring / aggregation only iterates one list.
    """
    index: int
    data_source: str
    prompt_text: str
    ground_truth: str
    raw_question: str
    greedy_completion: str
    sampled_completions: tuple[str, ...]


def build_vllm_engine(model_path: str, engine_config: dict[str, Any]):
    """Instantiate vLLM offline engine with settings from eval_frozen.yaml.engine."""
    from vllm import LLM

    print(f"[generator] loading vLLM engine from: {model_path}")
    return LLM(
        model=model_path,
        dtype=engine_config.get("dtype", "bfloat16"),
        tensor_parallel_size=engine_config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=engine_config.get("gpu_memory_utilization", 0.85),
        trust_remote_code=engine_config.get("trust_remote_code", False),
        seed=42,
    )


def build_greedy_sampling_params(config: EvalConfig):
    """SamplingParams for pass@1: greedy, deterministic, single completion."""
    from vllm import SamplingParams

    greedy_config = config.sampling["pass_at_1"]
    return SamplingParams(
        n=greedy_config["n"],
        temperature=greedy_config["temperature"],
        top_p=greedy_config["top_p"],
        max_tokens=config.generation["max_new_tokens"],
        stop=config.generation.get("stop_tokens", []),
    )


def build_sampled_sampling_params(config: EvalConfig):
    """SamplingParams for maj@8: stochastic, fixed seed for reproducibility."""
    from vllm import SamplingParams

    sampled_config = config.sampling["maj_at_8"]
    return SamplingParams(
        n=sampled_config["n"],
        temperature=sampled_config["temperature"],
        top_p=sampled_config["top_p"],
        max_tokens=config.generation["max_new_tokens"],
        stop=config.generation.get("stop_tokens", []),
        seed=sampled_config.get("seed", 42),
    )


def run_generation(
    vllm_engine,
    eval_samples: Sequence[EvalSample],
    greedy_params,
    sampled_params,
) -> list[SampleGeneration]:
    """Two passes on the same prompts: greedy first, then sampled.

    vLLM batches internally, so we pass the full prompt list per call rather
    than looping per-sample (throughput reason).
    """
    prompt_texts = [sample.prompt_text for sample in eval_samples]

    print(f"[generator] greedy pass on {len(prompt_texts)} prompts ...")
    greedy_outputs = vllm_engine.generate(prompt_texts, greedy_params)

    print(f"[generator] sampled pass (n={sampled_params.n}) on {len(prompt_texts)} prompts ...")
    sampled_outputs = vllm_engine.generate(prompt_texts, sampled_params)

    generations: list[SampleGeneration] = []
    for eval_sample, greedy_output, sampled_output in zip(
        eval_samples, greedy_outputs, sampled_outputs, strict=True,
    ):
        generations.append(SampleGeneration(
            index=eval_sample.index,
            data_source=eval_sample.data_source,
            prompt_text=eval_sample.prompt_text,
            ground_truth=eval_sample.ground_truth,
            raw_question=eval_sample.raw_question,
            greedy_completion=greedy_output.outputs[0].text,
            sampled_completions=tuple(
                completion.text for completion in sampled_output.outputs
            ),
        ))
    return generations

