#!/usr/bin/env bash
# P1 seed sweep — sequential, captures test_acc per seed to a CSV.
set -e
cd /fs/nexus-scratch/haowenyu/SomeEventTemplate
mkdir -p experiments/p1_results
LOG=experiments/p1_results/seed_sweep.csv
echo "seed,test_acc,test_loss" > $LOG
for s in 0 1 2 42 1337; do
  echo "=== seed=$s start $(date -Is) ==="
  out=experiments/p1_results/seed_${s}.log
  python train.py --config configs/exp/p1_seed_sweep/seed_${s}.yaml > $out 2>&1
  test_acc=$(grep -oE 'test_acc[[:space:]]*│[[:space:]]*[0-9.]+' $out | tail -1 | grep -oE '[0-9.]+$')
  test_loss=$(grep -oE 'test_loss[[:space:]]*│[[:space:]]*[0-9.]+' $out | tail -1 | grep -oE '[0-9.]+$')
  echo "$s,$test_acc,$test_loss" >> $LOG
  echo "=== seed=$s done test_acc=$test_acc ==="
done
echo "=== seed sweep complete ==="
cat $LOG
