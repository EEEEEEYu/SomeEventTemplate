#!/usr/bin/env bash
# Phase P §P4d — launch the Tier 0 decisive experiment on DVS-Gesture.
#
# Long-running. First-time setup encodes DVS-Gesture to TBR (~minutes), then
# trains for max_epochs (~hours). Output in experiments/p4_tier0/
set -e
cd /fs/nexus-scratch/haowenyu/SomeEventTemplate
mkdir -p experiments/p4_tier0

LOG=experiments/p4_tier0/run.log
echo "=== P4d Tier 0 launch $(date -Is) ===" | tee $LOG
echo "config: configs/exp/P4_tier0_dvsgesture.yaml" | tee -a $LOG
python train.py --config configs/exp/P4_tier0_dvsgesture.yaml 2>&1 | tee -a $LOG
echo "=== P4d Tier 0 complete $(date -Is) ===" | tee -a $LOG
