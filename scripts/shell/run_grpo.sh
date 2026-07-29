#!/usr/bin/env bash
# Usage:
#   bash scripts/run_grpo.sh grpo_multisource_v1
#   bash scripts/run_grpo.sh grpo_lora           +trainer.total_epochs=2   # 临时 override
set -euo pipefail
set -x

CONFIG_NAME="${1:-}"
if [ -z "$CONFIG_NAME" ]; then
    echo "usage: $0 <config_name> [+key=value ...]"
    echo "  config_name 是 configs/train/ 下的 yaml 文件名(不含 .yaml)"
    exit 1
fi
shift

# 加载机器环境
source "$(dirname "$0")/../env.sh"

# 让 grpo_reasoner 可 import (verl 加载 reward_adapter.py 时会 from grpo_reasoner.scoring)
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}

# 抑制杂音
export PYTHONUNBUFFERED=1
export SETUPTOOLS_USE_DISTUTILS=stdlib
export OMP_NUM_THREADS=1

cd $VERL_REPO

# Hydra 会自动创建 runs/<timestamp>_<experiment>/ 目录并把 stdout 也 tee 进去(通过我们下面的 tee)
python3 -m verl.trainer.main_ppo \
    --config-path=$(cd "$OLDPWD" && pwd)/configs \
    --config-name=train/${CONFIG_NAME} \
    "$@" \
    2>&1 | tee /tmp/latest_run.log

# hydra 已经把 run 目录写到 hydra.run.dir(base.yaml 里定义),把 tee 的 log 挪进去
RUN_DIR=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$OLDPWD/configs/train/${CONFIG_NAME}.yaml')
# 简单起见,不做完整 compose,只报告最新的 runs 目录
import glob, os
runs = sorted(glob.glob('$OLDPWD/runs/*_${CONFIG_NAME}' if 'experiment_name' not in dir() else '$OLDPWD/runs/*'))
print(runs[-1] if runs else '')
")
[ -n "$RUN_DIR" ] && cp /tmp/latest_run.log "$RUN_DIR/training.log"

echo "$RL_ROOT/checkpoints/${TRAINER_EXPERIMENT_NAME}" > "$RUN_DIR/checkpoint_path.txt"