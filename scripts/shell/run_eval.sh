#!/usr/bin/env bash
# Usage:
#   bash scripts/shell/run_eval.sh eval_frozen_v1 /path/to/model my_tag
#   bash scripts/shell/run_eval.sh eval_frozen_v1 /path/to/model my_tag +sampling.n=1
#
# Positional args:
#   $1 = config name under configs/eval/ (no .yaml suffix)
#   $2 = model path or HF repo id
#   $3 = tag (optional, default: untagged)
#   remaining = Hydra CLI overrides, e.g. +eval_sets.0.size=8

set -euo pipefail
set -x

CONFIG_NAME="${1:-}"
MODEL="${2:-}"
TAG="${3:-untagged}"

if [ -z "$CONFIG_NAME" ] || [ -z "$MODEL" ]; then
    echo "usage: $0 <config_name> <model_path> [tag] [+key=value ...]"
    exit 1
fi

# consume the three positional args; any leftover are hydra overrides
shift 3 2>/dev/null || shift $#

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO_ROOT/env.sh"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

python3 -m grpo_reasoner.evaluation \
    --config-path="$REPO_ROOT/configs" \
    --config-name="eval/${CONFIG_NAME}" \
    model="$MODEL" \
    tag="$TAG" \
    "$@"