#!/usr/bin/env bash
# 机器绑定的路径,换机器时改这里。其他一切都在 configs/ 里
export RL_ROOT=/root/autodl-tmp/rl_playground
export HF_HOME=$RL_ROOT/hf_cache
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export VERL_REPO=$RL_ROOT/repos/verl
export WANDB_MODE=${WANDB_MODE:-offline}