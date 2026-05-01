# Differentiable Logic Gate Networks for Event Cameras

PyTorch Lightning training stack for differentiable logic gate networks
([torchlogix](https://github.com/ligerlac/torchlogix)) on two tasks:

- **Image classification** (CIFAR-10, paper-spec `ClgnCifar10*` from
  Petersen et al. 2024).
- **Event-camera action recognition** (DVS-Gesture, single-fused
  TBR encoding feeding a logic-conv classifier).

A flow-estimation path (`TorchlogixFlow`) is stubbed for Phase 2 (MVSEC).

The research plan and stage gates live in [proposal_v3.md](proposal_v3.md);
live execution state is in [STATUS.md](STATUS.md). The verification of
torchlogix and Stage-A→B refactor history is in
[docs/torchlogix_verification.md](docs/torchlogix_verification.md). The
v2.1 streaming-buffer experiments are preserved at git tag
`v2.1-streaming-buffer-relic`.

## Layout

```
proposal_v3.md      — research plan (immutable spec)
STATUS.md           — live execution state
train.py            — entry point; explicit model + data registries
configs/
  base.yaml         — shared defaults
  exp/
    cifar10_M.yaml      — CIFAR-10 classification (paper-spec ClgnCifar10Medium)
    dvsgesture_M.yaml   — DVS-Gesture classification (TBR M=128 → fused conv-LGN)
src/
  data/
    cifar10_dm.py     — torchvision CIFAR-10 + optional thermometer transform
    dvsgesture_dm.py  — DVS-Gesture (tonic) + TBR encoder (HDF5 cached)
    thermometer.py    — paper-faithful thermometer encoding (utility)
    tbr.py            — vectorized event → TBR tensor encoder
  models/
    torchlogix_classifier.py   — classification (CIFAR-10 + DVS-Gesture)
    torchlogix_flow.py         — flow estimation (Phase-2 stub)
  modules/
    torchlogix_backbones.py    — backbone factories: cifar10 / gesture / flow
  utils/                       — config, callbacks (PlainTextProgress + grad-norm
                                  logger), seeding, resume, manifest
tests/                — pytest; correctness gates for shared utilities
experiments/          — per-run manifest.json + run.log
data/                 — gitignored: raw datasets + cached encodings
docs/
  cdlgn_paper.md             — paper architecture summary (reference)
  torchlogix_verification.md — Stage-A verification report
```

## Setup

```bash
mamba activate torch
pip install -r requirements.txt
```

torchlogix ships pure-Python — no CUDA build step. CIFAR-10 + DVS-Gesture
both auto-download on first run.

## Run

```bash
# CIFAR-10 (4-GPU DDP):
bash scripts/launch_train.sh configs/exp/cifar10_M.yaml

# DVS-Gesture:
bash scripts/launch_train.sh configs/exp/dvsgesture_M.yaml

# Sanity check on a login node — builds Trainer, prints registry, no fit:
python train.py --config configs/exp/cifar10_M.yaml --dry-run
```

Each experiment is one yaml — no CLI hyperparameter overrides (proposal v3
§Cross-cutting rules: "One config per experiment").

## Tests

```bash
pytest tests/ -v
```

CPU-runnable; cluster-GPU tests skip on the login node.

## Status

[STATUS.md](STATUS.md) for the current phase + gate checklist.
