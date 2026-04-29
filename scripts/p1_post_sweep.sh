#!/usr/bin/env bash
# Runs after the seed sweep finishes. Two cheap follow-ups in sequence:
#   1. P1 LR sensitivity (3 short 30-epoch runs)
#   2. P3 WordLogicLayer + ShiftedWordLogicLayer GPU parity tests
set -e
cd /fs/nexus-scratch/haowenyu/SomeEventTemplate
mkdir -p experiments/p1_results

# --- LR sensitivity ---
LR_LOG=experiments/p1_results/lr_sensitivity.csv
echo "lr,test_acc,test_loss,val_acc_at_30ep" > $LR_LOG
for lr_tag in 0_005 0_01 0_02; do
  echo "=== lr=$lr_tag start $(date -Is) ==="
  out=experiments/p1_results/lr_${lr_tag}.log
  python train.py --config configs/exp/p1_lr_sensitivity/lr_${lr_tag}.yaml > $out 2>&1
  test_acc=$(grep -oE 'test_acc[[:space:]]*│[[:space:]]*[0-9.]+' $out | tail -1 | grep -oE '[0-9.]+$' || echo "")
  test_loss=$(grep -oE 'test_loss[[:space:]]*│[[:space:]]*[0-9.]+' $out | tail -1 | grep -oE '[0-9.]+$' || echo "")
  val_acc=$(grep -oE 'val_acc[: ]+[0-9.]+' $out | tail -1 | grep -oE '[0-9.]+$' || echo "")
  echo "$lr_tag,$test_acc,$test_loss,$val_acc" >> $LR_LOG
  echo "=== lr=$lr_tag done test_acc=$test_acc ==="
done
echo "=== LR sensitivity complete ==="
cat $LR_LOG

# --- P3 / P4c GPU parity tests ---
echo "=== P3 + P4c GPU parity tests ==="
pytest tests/test_word_equivalence_forward.py tests/test_shifted_word_logic.py -v
echo "=== Phase P cheap workstreams complete ==="
