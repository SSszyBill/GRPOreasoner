"""Hydra entry point for python -m grpo_reasoner.evaluation."""
from future import annotations
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoTokenizer
from grpo_reasoner.evaluation import (
    aggregate_eval_set,
    build_greedy_sampling_params,
    build_sampled_sampling_params,
    build_vllm_engine,
    load_eval_config,
    load_eval_data,
    run_generation,
    score_generations,
    write_all_outputs,
)


# @hydra.main(config_path="../config/eval", config_name="eval_frozen_v1", version_base=None)
# def main(cfg: DictConfig) -> None:
#     pass
#         description="Frozen-criteria evaluation harness for GRPO experiments.",
#     )
#     parser.add_argument("--config", required=True, help="Path to eval_frozen.yaml")
#     parser.add_argument("--model", required=True, help="Model path or HF repo id")
#     parser.add_argument("--working-dir", default=".", help="Base dir for relative parquet paths in yaml")
#     parser.add_argument("--out", required=False, help="Output dir for baseline / post-training results")
#     parser.add_argument("--tag", required=False, default="untagged", help="Run tag, used in output subfolder name")
#     parser.add_argument("--skip-generation", action="store_true", help="Only run config + data loader; skip vLLM")
#     return parser.parse_args()


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    source_yaml = Path(hydra_cfg.runtime.config_sources[-1].path) / f"{hydra_cfg.job.config_name}.yaml"
    config = load_eval_config(cfg, source_yaml_path=source_yaml)
    print(f"[eval] config version={config.version}, "
        f"sha256={config.yaml_sha256[:16]}...")
    if cfg.model in (None, "???"):
        raise ValueError("model is required (pass model=/path/to/model on CLI)")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    print(f"[eval] tokenizer loaded from {cfg.model}")

    eval_data_by_set = load_eval_data(config, tokenizer, cfg.paths.rl_root)
    for eval_set_name, samples in eval_data_by_set.items():
        print(f"\n--- {eval_set_name} (n={len(samples)}) ---")
        first_sample = samples[0]
        print(f"  data_source: {first_sample.data_source}")
        print(f"  ground_truth: {first_sample.ground_truth!r}")
        print(f"  prompt (first 200 chars):")
        print(f"    {first_sample.prompt_text[:200]!r}...")

    if cfg.skip_generation:
        print("\n[run_eval] --skip-generation set, exiting after data load")
        return

    print("\n[run_eval] building vLLM engine ...")
    vllm_engine = build_vllm_engine(cfg.model, config.engine)
    greedy_params = build_greedy_sampling_params(config)
    sampled_params = build_sampled_sampling_params(config)
    print(f"[run_eval] greedy params : n={greedy_params.n}, "
          f"temperature={greedy_params.temperature}, max_tokens={greedy_params.max_tokens}")
    print(f"[run_eval] sampled params: n={sampled_params.n}, "
          f"temperature={sampled_params.temperature}, seed={sampled_params.seed}, "
          f"max_tokens={sampled_params.max_tokens}")

    scored_by_eval_set: dict = {}
    metrics_by_eval_set: dict = {}

    for eval_set_name, samples in eval_data_by_set.items():
        print(f"n[eval] generating for {eval_set_name} ...")
        generations = run_generation(vllm_engine, samples, greedy_params, sampled_params)
        print(f"[eval] {eval_set_name}: {len(generations)} generations done")
        scored = score_generations(generations)
        scored_by_eval_set[eval_set_name] = scored
        eval_set_spec = next(spec for spec in config.eval_sets if spec["name"] == eval_set_name)
        metrics = aggregate_eval_set(eval_set_name, eval_set_spec["data_source"], scored)
        metrics_by_eval_set[eval_set_name] = metrics
        print(
            f"[metrics] {eval_set_name}: "
            f"pass@1={metrics.pass_at_1:.3f}, maj@8={metrics.maj_at_8:.3f}, "
            f"format_rate={metrics.format_rate:.3f}"
        )

    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path(hydra_cfg.runtime.output_dir)
    written_paths = write_all_outputs(
        config=config,
        model_path=cfg.model,
        tag=cfg.tag,
        metrics_by_eval_set=metrics_by_eval_set,
        scored_by_eval_set=scored_by_eval_set,
        output_dir=out_dir,
    )
    print("n[eval] wrote outputs:")
    for artifact_kind, path in written_paths.items():
        print(f"  {artifact_kind}: {path}")
    print("n[eval] done.")

if __name__ == "__main__":
    main()

