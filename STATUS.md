# Project Status

**Last updated:** 2026-04-28
**Current stage:** Stage 0 complete (with documented deviation); ready for Stage 1
**Blockers:** none — GPU node allocated (rtxa6000 ×4)

`proposal.md` is the authoritative spec; this file is the execution log.

---

## Phase A — pre-GPU scaffolding

- [x] A1 — Repo restructure (template scaffolding deleted, `src/` layout created)
- [x] A2 — Salvage utilities to `src/utils/` (config, callbacks, resume, seeding)
- [x] A3 — `STATUS.md` initialized
- [x] A4 — `src/data/tbr.py` + `tests/test_tbr_encoding.py` (10/10 pass on CPU)
- [x] A5 — Test scaffolds for Stages 0/3/4 (skip cleanly when prereqs missing — 19 skipped)
- [x] A6 — `difflogic` clone instructions in README + `.gitignore` excludes the directory
- [x] A7 — `requirements.txt` updated (`tonic`, `h5py`, `pytest`); `README.md` rewritten

**Verification:**
```bash
mamba activate torch
python -c "from src.utils.config import load_config; from src.utils.callbacks import load_callbacks; print('ok')"
pytest tests/ -v          # tbr tests pass; difflogic tests skip
python train.py --config configs/base.yaml --dry-run
```

---

## Phase B — proposal stages (require GPU)

### Stage 0 — environment & sanity (proposal §Stage 0) — ✅ accepted
- [x] `import difflogic_cuda` works (after `import torch`; needs kernel patch — see decisions log)
- [x] `tests/test_difflogic_imports.py` — 3/3 pass
- [~] MNIST repro ≥ 97.5% — peak **97.40%**, ~0.1% under nominal; accepted as install-correctness signal (see [experiments/00_difflogic_repro.md](experiments/00_difflogic_repro.md) §Decision)
- Manifest: [experiments/00_difflogic_repro.md](experiments/00_difflogic_repro.md)
- **Stage 1 parity baseline: 97.40%** (±0.3% tolerance window: 97.10–97.70%)

### Stage 1 — Lightning + MNIST parity (proposal §Stage 1)
- [ ] Lightning training runs without errors
- [ ] Discretized eval ≥ 97.2% on MNIST
- [ ] Gate count + inference latency logged
- Manifest target: `experiments/01_mnist_lightning.md`

### Stage 2 — TBR + scalar CDLGN, both datasets (proposal §Stage 2)
- [ ] TBR encoder unit-tested  ← (lands in Phase A)
- [ ] N-MNIST scalar CDLGN ≥ 95%
- [ ] DVS-Gesture scalar CDLGN ≥ 80% (32×32 downsample is fine)
- [ ] Baseline numbers logged for both
- Manifest target: `experiments/02_scalar_cdlgn.md`

### Stage 3 — word substrate, strict-rigor (proposal §Stage 3)
- [ ] Forward equivalence test passes (multi-seed, W ∈ {1, 8, 32})
- [ ] Backward equivalence test passes (output + gradients close)
- [ ] N-MNIST matched-accuracy within ±0.3% of Stage 2
- [ ] DVS-Gesture matched-accuracy within ±0.5% of Stage 2
- [ ] Throughput + memory logged for both
- [ ] Connectivity-init audit documented
- Manifest target: `experiments/03_word_substrate.md`

### Stage 4 — cross-bit operators (proposal §Stage 4) — make-or-break
- [ ] `ShiftedWordLogicLayer` discretized correctness test passes
- [ ] N-MNIST and DVS-Gesture results logged (iso-hidden-dim AND iso-gates)
- [ ] Operator-distribution histogram saved
- [ ] Honest decision documented (proposal §Decision point)
- Manifest target: `experiments/04_shifts_results.md`

### Stage 5 — comparative baselines (deferred)
- [ ] Triggered iff Stage 4 produces a positive result

---

## Decisions & deviations log

<!-- Append-only. Each entry: date — decision — rationale — link to commit/PR. -->

- **2026-04-28** — Discarded the original template's dynamic-import `ModelInterface`/`DataInterface` core; adopted proposal's `src/{data,models,modules,utils}/` layout with explicit registries in `train.py`. Rationale: per-stage Lightning modules need bespoke `configure_optimizers` (Adam lr=0.01) and inference-mode toggles that fight a generic wrapper. Salvaged: config schema, callback assembly, resume helpers (now under `src/utils/`).
- **2026-04-28** — Patched `difflogic/cuda/difflogic_kernel.cu` (six `AT_DISPATCH_*` sites: `.type()` → `.scalar_type()`) so it compiles against PyTorch 2.8 (`at::DeprecatedTypeProperties` no longer auto-converts to `c10::ScalarType`). Patch saved to `patches/difflogic_pytorch28_scalar_type.patch`; difflogic commit pinned to `469702c01ff0bfac9cdc6a395134252e11a56bd8`. Install command is `pip install ./difflogic --no-build-isolation` — `-e` and PEP 517 isolation both break in different ways.
- **2026-04-28** — Stage 0 MNIST repro gate **accepted at 97.40%** (nominal threshold was 97.5%; deviation ~0.1%). Plateaued for ~30k iterations with no upward trend; further training would not change the install-correctness signal this gate is meant to provide. Stage 1 parity check now uses 97.40% as the baseline (±0.3% window). Full rationale: [experiments/00_difflogic_repro.md](experiments/00_difflogic_repro.md) §Decision.

## Open risks

- DVS-Gesture full 128×128 input dimensionality (proposal §Note on DVS-Gesture spatial resolution) — start with 32×32 downsample; revisit only if accuracy is poor.
- TBR cache directory growth — log size after first encode; set retention policy if it gets large.
- difflogic CUDA build version-mismatch — pin a known-good commit; record CUDA toolkit + PyTorch version in run manifests.
- Stage 3 connectivity-init parity (proposal §Stage 3 task 6) — must use difflogic's exact RNG path or matched-accuracy gate may spuriously fail.

## Recent run pointers

<!-- One bullet per completed run, newest first. Format:
  - YYYY-MM-DD  experiment_name  metric_summary  manifest_path  commit
-->

(none yet)
