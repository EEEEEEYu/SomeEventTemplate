#!/usr/bin/env bash
# Phase 1 — launch a CDLGN CIFAR-10 run.
#
# Usage:  bash scripts/p1_launch_cdlgn.sh configs/exp/02_cdlgn_cifar10_S.yaml
#
# The CIFAR-10 cache is downloaded under data/ on first run (~170 MB).
# Output (TensorBoard events + manifest.json + best/last checkpoints) goes to
# the experiment_name set in the yaml's LOGGER block.
set -euo pipefail
cd /fs/nexus-scratch/haowenyu/SomeEventTemplate

CONFIG=${1:?"usage: $0 <config.yaml>"}
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

EXP_NAME=$(grep -E '^[[:space:]]*experiment_name:' "$CONFIG" | head -1 | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'")
EXP_NAME=${EXP_NAME:-cdlgn_cifar10}

mkdir -p "experiments/${EXP_NAME}"
LOG="experiments/${EXP_NAME}/run.log"

echo "=== Phase 1 launch ==="
echo "config:        $CONFIG"
echo "experiment:    $EXP_NAME"
echo "log:           $LOG"
echo "git commit:    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "started:       $(date -Is)"
echo

PY=${PYTHON:-/fs/nexus-scratch/haowenyu/miniforge3/envs/torch/bin/python}
"$PY" train.py --config "$CONFIG" 2>&1 | tee "$LOG"
