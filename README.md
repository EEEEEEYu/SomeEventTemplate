# Word-Level Logic Gate Networks for Event Cameras

PyTorch Lightning implementation of word-level differentiable logic gate networks
([difflogic](https://github.com/Felix-Petersen/difflogic)) trained on event-camera
data (N-MNIST, DVS-Gesture). The contribution is a cross-bit (shifted) operator
vocabulary that exploits the temporal structure in TBR-encoded event streams.

The full research plan, stage gates, and decision points live in [proposal.md](proposal.md).
Live execution state is tracked in [STATUS.md](STATUS.md).

## Layout

```
proposal.md     — research plan (immutable spec)
STATUS.md       — live execution state, per-stage gate checklist
train.py        — Hydra/OmegaConf entry point; explicit model/data registries
configs/
  base.yaml     — shared defaults
  exp/          — one yaml per experiment (stages 1–4)
src/
  data/         — Lightning DataModules + TBR encoder
  models/       — LightningModules
  modules/      — nn.Module building blocks (logic_blocks, word_logic)
  utils/        — config, callbacks, resume, seeding
tests/          — pytest; equivalence + correctness gates
experiments/    — per-stage write-ups + per-run manifest.json
data/           — gitignored: raw datasets + cached TBR tensors
difflogic/      — gitignored: cloned externally (see Setup)
```

## Setup

The cluster GPU is queued; phase A (scaffolding, TBR encoder, tests) runs on the
login node via the `torch` mamba env. Phase B (proposal stages 0–4) needs GPU.

```bash
mamba activate torch
pip install -r requirements.txt

# difflogic ships a CUDA extension. Clone it once a GPU node is available.
# Pin a known-good commit during Stage 0 (proposal §Stage 0 task 1) and record
# it in the run manifest.
git clone https://github.com/Felix-Petersen/difflogic.git
pip install -e ./difflogic       # builds difflogic_cuda — needs CUDA
python -c "import difflogic_cuda"   # must succeed before Stage 1
```

## Run a stage

```bash
# Sanity check on the login node — builds Trainer, prints registry, no fit:
python train.py --config configs/base.yaml --dry-run

# Real training (once GPU + a stage's config exists):
python train.py --config configs/exp/01_mnist_lightning.yaml
```

Each experiment is one yaml — there are no CLI overrides for hyperparameters
(proposal §Cross-cutting rules: "One config per experiment").

## Tests

```bash
pytest tests/ -v
```

On the login node: 10 TBR encoder tests pass; difflogic / word-logic equivalence
tests are in tree but skip cleanly until their prereqs land.

## Status

See [STATUS.md](STATUS.md) for the current phase, which gates have passed, and
the per-stage progress checklist.
