#!/usr/bin/env bash
# scripts/run_grpo.sh
# Usage: bash scripts/run_grpo.sh <exp_dir> [hydra_overrides...]

set -x

EXP_DIR=$(cd $1 && pwd)
shift

source $EXP_DIR/env.sh
cd $VERL_REPO

export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE=$HF_HOME/datasets
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0



EXP_NAME_PREFIX=${EXP_NAME_PREFIX:-smoke}
EXP_NAME=${EXP_NAME_PREFIX}_$(date +%Y%m%d_%H%M)

python3 -m verl.trainer.main_ppo \
    --config-path=$EXP_DIR/configs \
    --config-name=train \
    \
    data.train_files=$TRAIN_FINAL_DIR/train.parquet \
    data.val_files=$TRAIN_FINAL_DIR/train.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    custom_reward_function.path=$REWARD_MULTI_PATH \
    \
    trainer.project_name=$(basename $EXP_DIR) \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=$EXP_DIR/checkpoints/$EXP_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=99999 \
    trainer.total_training_steps=80 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    "$@" \
    2>&1 | tee $RL_ROOT/verl_grpo_demo/logs/run_grpo_multisource_4090.log