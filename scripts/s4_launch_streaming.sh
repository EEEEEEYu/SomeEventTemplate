#!/usr/bin/env bash
# Stage 4 / Tier 1 streaming run launcher (proposal v2 §Stage 4).
# First streaming-architecture training pass: per-slice encoder + [N=32, M=32]
# buffer + difflogic16 cross-slice + WordLogicLayer decoder + GroupSum.
#
# `grad_factor=2.0` per proposal §Stage 1 task 2 (mitigates ~10-layer
# vanishing-grad chain). `tbptt_k=null` = full BPTT (Tier 1).
set -e
cd /fs/nexus-scratch/haowenyu/SomeEventTemplate
mkdir -p experiments/s4_streaming

LOG=experiments/s4_streaming/run.log
echo "=== S4 streaming launch $(date -Is) ===" | tee $LOG
echo "config: configs/exp/S4_streaming_dvsgesture.yaml" | tee -a $LOG
python train.py --config configs/exp/S4_streaming_dvsgesture.yaml 2>&1 | tee -a $LOG
echo "=== S4 streaming complete $(date -Is) ===" | tee -a $LOG
