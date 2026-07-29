#!/usr/bin/env bash

set -x

cd $RL_ROOT/repos/verl

export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$RL_ROOT/hf_cache
export TRANSFORMERS_CACHE=$RL_ROOT/hf_cache
export HF_DATASETS_CACHE=$RL_ROOT/hf_cache/datasets

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$RL_ROOT/data/gsm8k/train.parquet \
    data.val_files=$RL_ROOT/data/gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=verl_grpo_demo \
    trainer.experiment_name=4090_smoke_test \
    trainer.default_local_dir=$RL_ROOT/checkpoints/verl_grpo_demo/4090_smoke_test \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    2>&1 | tee $RL_ROOT/logs/run_grpo_gsm8k_4090.log
