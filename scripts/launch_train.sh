#!/usr/bin/env bash
# Generic train launcher. Tail-friendly progress goes to both stdout and
# experiments/<EXP_NAME>/run.log.
#
# Usage:
#   bash scripts/launch_train.sh configs/exp/cifar10_M.yaml
#   bash scripts/launch_train.sh configs/exp/dvsgesture_M.yaml
#
# Datasets cache to data/ on first run.
set -euo pipefail
cd /fs/nexus-scratch/haowenyu/SomeEventTemplate

CONFIG=${1:?"usage: $0 <config.yaml>"}
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

EXP_NAME=$(grep -E '^[[:space:]]*experiment_name:' "$CONFIG" | head -1 | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'")
EXP_NAME=${EXP_NAME:-train_run}

mkdir -p "experiments/${EXP_NAME}"
LOG="experiments/${EXP_NAME}/run.log"

echo "=== launch ==="
echo "config:        $CONFIG"
echo "experiment:    $EXP_NAME"
echo "log:           $LOG"
echo "git commit:    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "started:       $(date -Is)"
echo

PY=${PYTHON:-/fs/nexus-scratch/haowenyu/miniforge3/envs/torch/bin/python}
"$PY" train.py --config "$CONFIG" 2>&1 | tee "$LOG"
