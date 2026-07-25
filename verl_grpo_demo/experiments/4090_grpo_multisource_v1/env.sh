#!/usr/bin/env bash
# Environment for experiment: 4090_grpo_multisource_v1
# All paths and config for this experiment. Frozen with git.

# ---- infrastructure paths ----
export RL_ROOT=/root/autodl-tmp/rl_playground
export HF_CACHE_DIR=$RL_ROOT/hf_cache/hub
export HF_HOME=$RL_ROOT/hf_cache
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# ---- base model ----
export MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
export MODEL_DIR=$HF_CACHE_DIR/models--Qwen--Qwen2.5-1.5B-Instruct
export MODEL=$MODEL_DIR                    # alias for scripts that use $MODEL

# ---- data paths ----
export CANDIDATE_DIR=$RL_ROOT/data/candidate
export BUCKETED_DIR=$RL_ROOT/data/bucketed
export TRAIN_FINAL_DIR=$RL_ROOT/data/train_final

# ---- output paths ----
export RESULTS_DIR=$RL_ROOT/results

# ---- reward function ----
export REWARD_MULTI_PATH=$RL_ROOT/verl_grpo_demo/utils/reward_multi.py

# ---- verl repo ----
export VERL_REPO=$RL_ROOT/repos/verl

# ---- 恼人 warning 抑制 ----
export OMP_NUM_THREADS=1
unset TRANSFORMERS_CACHE   # legacy, use HF_HOME instead

# ---- Python path so `python -m evaluation` finds packages in verl_grpo_demo/ ----
export PYTHONPATH=$RL_ROOT/verl_grpo_demo:$PYTHONPATH

# 从 https://wandb.ai/authorize 复制 API key
export WANDB_API_KEY="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"
# AutoDL 国内可能连不通 wandb.ai，先 offline，训练完再 sync
# export WANDB_MODE=online
# 可选：wandb 本地缓存目录
export WANDB_DIR="$EXP_DIR/wandb"