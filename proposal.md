# Implementation Plan: Word-Level Logic Gate Networks for Event Cameras

**Intended reader: a coding agent integrating `difflogic` into a PyTorch Lightning template.**

This plan is staged. **Do not skip stages.** Each stage has an explicit pass/fail gate. If a gate fails, stop and report — do not proceed to the next stage hoping the issue resolves itself.

User configuration choices encoded into this plan:
- **Datasets:** N-MNIST as the small/fast iteration gate, **DVS-Gesture validated alongside it** from Stage 2 onward.
- **Rigor mode:** *Minimal overall, with strict rigor on Stage 3.* Fastest honest path to a Stage 4 (cross-bit operator) result, but the word substrate in Stage 3 is the foundation Stage 4 builds on, so it gets multiple independent correctness checks. Comparative baselines (TBR-CNN, TBR-BNN) remain deferred to a post-Stage-4 polish pass.
- **TBR encoding:** Dataloader-side, encoded once and cached to disk using hdf5 format which is modern and managable.

What "minimal rigor" means in this plan: skip comparative baselines and most ablation runs in early stages, but **keep all correctness gates**. A correctness gate (accuracy threshold, equivalence test, encoder unit test) is early-failure detection, not paperwork — skipping it costs more time than it saves.

---

## Repository layout

Assume the user has:

```
project/
├── src/
│   ├── data/                  # Lightning DataModules
│   ├── models/                # LightningModules
│   ├── modules/               # nn.Module building blocks
│   └── utils/
├── configs/                   # Hydra/yaml configs
├── tests/
├── difflogic/                 # cloned from Felix-Petersen/difflogic
└── train.py
```

You will add code under `src/`. **Do not modify files inside `difflogic/`** — import from it. If you need to fix a bug there, fork into `src/modules/difflogic_patched/` and import that instead, with a comment explaining why.

---

## Stage 0 — Environment and sanity (target: 1–2 days)

### Goal
`difflogic` installs, builds its CUDA extension, and reproduces its own MNIST result inside the Lightning template (still using the original training loop, just invoked as a subprocess or one-off script).

### Tasks

1. **Install difflogic.** Run `pip install -e ./difflogic`. This compiles `difflogic_cuda` via setuptools. If compilation fails, check CUDA toolkit version and PyTorch CUDA version match. Do not proceed until the CUDA extension imports cleanly: `python -c "import difflogic_cuda"` must succeed.

2. **Smoke test.** Run their bundled MNIST training script with default hyperparameters. Document the achieved accuracy in `experiments/00_difflogic_repro.md`. Acceptance: ≥ 97.5% test accuracy on MNIST. If lower, the install is broken; debug before proceeding.

3. **Write `tests/test_difflogic_imports.py`** that imports `LogicLayer`, `GroupSum`, `CompiledLogicNet`, and asserts a forward pass works on a random tensor. This catches install regressions early.

### Gate
- [ ] `import difflogic_cuda` works
- [ ] MNIST repro ≥ 97.5%
- [ ] Test passes

If any gate fails: STOP. Report the failure. Do not start Stage 1.

---

## Stage 1 — Lightning wrapping of difflogic, MNIST parity (target: 3–4 days)

### Goal
A `LightningModule` wrapping difflogic's `LogicLayer` + `GroupSum` reproduces MNIST performance within ±0.3% of the original script. This is a **correctness gate**, not a contribution.

### Tasks

1. **Create `src/modules/logic_blocks.py`.** Thin re-exports of `difflogic.LogicLayer`, `difflogic.GroupSum`. Do not modify these. Add a docstring at top: "Stage 1: pass-through wrappers. Stage 3 introduces word-level variants here."

2. **Create `src/models/logic_classifier.py`.** A `LightningModule` with:
   - `__init__` taking `in_dim`, `hidden_dim`, `num_layers`, `num_classes`, `tau`, `connections` (default `"random"`). Build a `nn.Sequential` of `LogicLayer`s followed by `GroupSum(k=num_classes, tau=tau)`. Match Petersen's defaults: hidden_dim=8000–16000, num_layers=4–6.
   - `training_step` / `validation_step` using `nn.CrossEntropyLoss` on the GroupSum output.
   - `configure_optimizers` returning Adam with **lr=0.01** (NOT 0.001 — this is critical, difflogic's relaxation needs the higher LR).
   - An `on_validation_epoch_start` hook that switches LogicLayers to discretized inference mode, and switches back at end. **Verify in difflogic source how this is toggled** — read `difflogic/__init__.py` and `difflogic/functional.py`. Inference accuracy is what matters; train-time soft accuracy is a vanity number.

3. **Create `src/data/mnist_dm.py`.** Standard MNIST DataModule with:
   - Binarize inputs by thresholding at 0.5 (matches difflogic's default input encoding).
   - Flatten to 784-dim boolean vectors.
   - Standard 50k/10k/10k train/val/test splits.

4. **Create `configs/exp/01_mnist_lightning.yaml`** with the hyperparameters above.

5. **Train.** Acceptance: discretized test accuracy ≥ 97.2% (allowing 0.3% slack vs Stage 0). Log gate counts, training time, eval throughput.

### Gate
- [ ] Lightning training runs without errors
- [ ] Discretized eval ≥ 97.2% on MNIST
- [ ] Gate count and inference latency logged in `experiments/01_mnist_lightning.md`

If accuracy is below threshold: the bug is almost certainly in (a) the discretization toggle, (b) the LR, (c) the GroupSum tau, or (d) the input binarization. Debug in that order. Do not proceed to Stage 2 until parity is achieved.

---

## Stage 2 — Event datasets + TBR baseline with scalar CDLGN (target: 1.5 weeks)

### Goal
Scalar CDLGN trained on TBR-encoded **N-MNIST and DVS-Gesture**. These are the baselines your future contributions must beat. Not a paper, just numbers to point at.

DVS-Gesture is included now (rather than in Stage 4) because N-MNIST has weak temporal structure — saccade-induced events on a static digit. Validating only on N-MNIST risks reaching Stage 4 without ever testing the cross-bit operator hypothesis on data with real temporal structure.

### Tasks

1. **Add event-camera dependencies.** `pip install tonic` (https://github.com/neuromorphs/tonic) handles N-MNIST/DVS-Gesture/N-Caltech101 with a unified API.

2. **Implement TBR encoding** in `src/data/tbr.py`:
   - Function `encode_tbr(events, num_bins=32, bin_duration_us=1000, sensor_size=(H, W, 2)) -> torch.Tensor` returning a `[2, num_bins, H, W]` boolean tensor (polarity × bins × spatial).
   - Per pixel per polarity per bin: 1 if any event in that 1ms window, else 0.
   - Vectorize using `torch.bincount` or scatter operations. Do NOT use a Python loop over events — N-MNIST has ~5k events per sample, gesture data has ~1M+. A loop is unusable.
   - Unit test in `tests/test_tbr_encoding.py`: round-trip a synthetic event stream (events at known timestamps and pixels) and assert the resulting bin tensor matches the hand-computed expected output. Test edge cases: empty events, events outside time window, multiple events in same bin.

3. **Create both DataModules in parallel.**
   - `src/data/nmnist_dm.py`: 34×34 sensor, 2 polarities, 32 bins × 1ms. Output shape `[2, 32, 34, 34]` = 73,984 boolean inputs flattened.
   - `src/data/dvsgesture_dm.py`: 128×128 sensor (consider downsampling — see note below), 2 polarities, 32 bins × 1ms. Choose a fixed sample-window length (e.g., 500ms) and crop/pad the event stream to that.

4. **Cache encoded tensors to disk.** Encode once into `data/<dataset>_tbr_32x1ms_<resolution>/` as either `.pt` files per sample or webdataset shards. The DataLoader reads cached tensors. Document the cache directory structure and encoding parameters in a manifest file so re-encoding with different parameters doesn't silently overwrite.

5. **Note on DVS-Gesture spatial resolution.** Full 128×128 means 2·32·128·128 = 1,048,576 input bits, which is large for an LGN. Two acceptable starting points:
   - Downsample to 32×32 spatial (max-pool the encoded TBR tensor): 65,536 input bits, comparable to N-MNIST. **Recommended for first run.**
   - Keep 128×128 with smaller hidden_dim and fewer layers to manage compute. Document the choice and revisit later if accuracy is poor.

6. **Train.** For each dataset:
   - Hidden_dim 32k–64k, num_layers 5–6.
   - Acceptance: discretized test accuracy ≥ 95% on N-MNIST, ≥ 80% on DVS-Gesture (11-class, the TBR paper reports ~95%+ with Inception3D — your floor is much lower; you just need a working model).

7. **Log baseline numbers** in `experiments/02_scalar_cdlgn.md`: per-dataset test accuracy, gate count, inference throughput on one CPU core (use difflogic's `CompiledLogicNet` if accessible), training time, GPU memory.

### Gate
- [ ] TBR encoder unit-tested
- [ ] N-MNIST scalar CDLGN ≥ 95%
- [ ] DVS-Gesture scalar CDLGN ≥ 80% (downsampled-spatial version is fine)
- [ ] Baseline numbers logged for both datasets

If N-MNIST plateaus low: try doubling hidden_dim, longer training, or verify the TBR tensor isn't degenerate (mostly zeros suggests bin width is too short or sensor_size is wrong). If DVS-Gesture plateaus very low (e.g., near random 9%): the input is probably too high-dimensional — downsample more aggressively. Don't proceed past these gates — if scalar CDLGN can't clear them, your word-level extension can't either.

---

## Stage 3 — Word-level LGN, no shifts (target: 1.5–2 weeks)

### Goal
A word-level `LogicLayer` that processes 32-bit-wide tensors via the same 16 bitwise functions, producing **mathematically identical** results to scalar CDLGN run 32 times in parallel. The implementation is a refactor (no new expressivity), but it is **the substrate Stage 4 builds on**, so correctness must be established with high confidence — bugs here will look like "shifts don't help" in Stage 4 and you won't be able to tell.

**Strict-rigor mode for this stage.** Multiple independent correctness checks (mathematical equivalence + matched-accuracy training on both datasets + throughput characterization) before declaring the substrate ready. The cost is roughly a week beyond minimal rigor; the payoff is that Stage 4 results are interpretable.

### Tasks

1. **Create `src/modules/word_logic.py`** with:
   - `WordLogicLayer(nn.Module)`: takes input shape `[B, N, W]` where W is word width (32 default). Internally maintains a `(out_features, 16)` softmax over the 16 binary functions, just like difflogic. The forward pass applies the chosen function bitwise across the W dimension.
   - During training: real-valued relaxation over `[0,1]`, applied independently per bit position. This means the per-bit forward is identical to difflogic; you're just batching across the W dimension.
   - During inference (after discretization): use bitwise integer ops on packed 32-bit ints. PyTorch supports `torch.bitwise_and/or/xor` on int32 tensors. This gives the 32× wall-clock speedup.
   - For `WordGroupSum`: simplest path is to unpack words into bits at the very end and reuse difflogic's `GroupSum` directly. Pick this; do not optimize.

2. **Forward-pass equivalence test (discretized).** Write `tests/test_word_equivalence_forward.py`:
   - Fix connectivity (random but seeded). Fix operator choices (one-hot over the 16 ops per neuron, pre-discretized).
   - Generate random binary input of shape `[B, N, W]`.
   - Compute forward two ways: (a) `WordLogicLayer` on the packed input, (b) `LogicLayer` from difflogic run W times on each bit-slice, then re-stacked.
   - Assert bit-for-bit equality.
   - Run with at least 3 different seeds and `W ∈ {1, 8, 32}` to catch off-by-one and packing bugs.

3. **Backward-pass equivalence test (relaxed).** Write `tests/test_word_equivalence_backward.py`:
   - Same fixed connectivity, but now use **soft** operator distributions (random softmax weights, not one-hot) and continuous-valued inputs in `[0,1]`.
   - Compute output and backward gradients through both paths (word-level and difflogic-scalar-replicated).
   - Assert outputs are numerically close (`atol=1e-5`), and gradients w.r.t. operator logits are numerically close (`atol=1e-4`).
   - This catches relaxation bugs that the forward test misses — e.g., a subtly wrong T-norm formula will pass forward equivalence on hard inputs but produce different gradients during training.

4. **Matched-accuracy training, both datasets.** Re-run the Stage 2 experiments with `LogicLayer → WordLogicLayer`, same seeds, same hyperparameters, same connectivity initialization:
   - N-MNIST: discretized accuracy must match Stage 2 within ±0.3%.
   - DVS-Gesture: discretized accuracy must match Stage 2 within ±0.5% (more variance is acceptable on the larger, noisier dataset).
   - If either drifts beyond tolerance, something is wrong with gradient flow through the word substrate even if the unit tests pass — investigate before proceeding. Likely culprits: numerical precision in the relaxation (try float64 for the per-bit operator interpolation), incorrect handling of the per-bit independence assumption in the loss reduction, or a bug in the unpack-to-bits step before `WordGroupSum`.

5. **Throughput and memory characterization.** On a single CPU core, measure inference samples/sec for `LogicLayer` (Stage 2 model) vs `WordLogicLayer` (Stage 3 model) at the same accuracy point. Use difflogic's `CompiledLogicNet` for both if accessible. Also record peak GPU memory during training. The word version should be substantially faster at inference on bit-packed inputs and roughly similar in training memory (per-bit relaxation does not save memory during training; the speedup is an inference-time property).

6. **Connectivity-init audit.** difflogic uses pseudorandom fixed connectivity. Verify your `WordLogicLayer` uses the *same* connectivity-generation seed and procedure as difflogic for the parity comparisons in task 4. A different RNG path here will produce different (but equally valid) networks, which would spuriously fail the matched-accuracy gate even with a correct implementation. Document the seed and procedure in the test.

### Gate
- [ ] `test_word_equivalence_forward` passes (bit-for-bit, multiple seeds, multiple W)
- [ ] `test_word_equivalence_backward` passes (output and gradient numerical match)
- [ ] N-MNIST matched-accuracy run within ±0.3% of Stage 2
- [ ] DVS-Gesture matched-accuracy run within ±0.5% of Stage 2
- [ ] Throughput and training-memory numbers logged for both datasets in `experiments/03_word_substrate.md`
- [ ] Connectivity-init audit documented (seed and procedure verified to match difflogic)

Do NOT add new operators in this stage. Adding shifts here will entangle two debugging surfaces — you won't know if a future accuracy regression is from a buggy shift or a buggy word substrate. The whole point of this stage's strict rigor is that when Stage 4 produces a result (positive or negative), you can attribute it cleanly to the new operators.

---

## Stage 4 — Cross-bit operators (the actual contribution) (target: 2.5–3 weeks)

### Goal
Extend the operator vocabulary with bit-shift variants. These genuinely couple bits across the word dimension and cannot be expressed by a single scalar CDLGN gate. Test the hypothesis that they help on event data — particularly on DVS-Gesture, where temporal structure is real.

### Tasks

1. **Choose the operator vocabulary.** Start with the simpler "compound op" formulation: enumerate `{16 binary ops} × {shift_b ∈ {0, 1, 2, 4, 8}}` as `16 * 5 = 80` distinct operators per neuron. Each is `op(a, SHIFT_b(b))`. Softmax over 80 instead of 16. This requires no nested gumbel-softmax and is a one-line vocabulary expansion in `WordLogicLayer`.

2. **Implement in `src/modules/word_logic.py` as `ShiftedWordLogicLayer`.** Keep `WordLogicLayer` from Stage 3 untouched — you want both available for ablations.

3. **Discretized correctness test.** Write a small unit test that verifies discretized inference of `ShiftedWordLogicLayer` matches a hand-computed reference for a few `(op_idx, shift_b)` configurations. Cheap insurance; do not skip.

4. **Train on N-MNIST.** Same architecture skeleton as Stage 3 with `LogicLayer → ShiftedWordLogicLayer`. Compare:
   - Word-no-shifts (Stage 3) vs Word-with-shifts at iso-hidden-dim.
   - Word-with-shifts at smaller hidden_dim matching the gate count of Stage 3 (the iso-gates comparison — this is what reviewers care about).

5. **Train on DVS-Gesture.** Same comparison. **This is where the central hypothesis gets tested.** N-MNIST has weak temporal structure, so a shift advantage there would be surprising; DVS-Gesture has strong temporal structure (gestures evolve over hundreds of ms), so shifts should help if the inductive bias hypothesis is right at all.

6. **Critical analysis: operator distribution.** Log a histogram of which operators the trained networks pick. If shifts are rarely chosen (< 10% of neurons) on DVS-Gesture, that's strong evidence against the inductive bias hypothesis, and you need to confront that finding rather than push past it.

7. **Decision point.** After Stage 4 results are in, evaluate honestly:
   - **Shifts help on DVS-Gesture (iso-gates accuracy improvement, or iso-accuracy gate reduction):** the project has a real result. Move to Stage 5 to formalize.
   - **Shifts don't help on either dataset:** the central thesis is wrong as stated. Two honest responses: (a) try a richer cross-bit op formulation (per-neuron learnable shift via gumbel-softmax — slower but more expressive), or (b) acknowledge the negative result and consider pivoting. **Do not** keep adding tricks indefinitely until something works.
   - **Shifts help on DVS-Gesture but not N-MNIST:** consistent with the inductive bias story (cross-bit ops help when temporal structure is present, not on essentially-static data). This is a *good* outcome and a tighter framing than "shifts help everywhere."

### Gate
- [ ] `ShiftedWordLogicLayer` unit-tested for discretized correctness
- [ ] N-MNIST and DVS-Gesture results both logged (iso-hidden-dim and iso-gates)
- [ ] Operator distribution histogram saved
- [ ] Honest decision documented in `experiments/04_shifts_results.md`

This is the **make-or-break** stage of the project.

---

## Stage 5 (deferred under minimal rigor) — Comparative baselines and Pareto plot

Skipped from the critical path. Run after Stage 4 produces a positive result, before paper submission. At that point: add TBR-CNN and TBR-BNN baselines (small models, fair comparison, not SOTA-chasing), generate the gates-vs-accuracy Pareto plot across all five model families, write up.

If you need to make a quick external case for the project before this stage (e.g., for a workshop deadline), the Stage 4 iso-gates comparison against scalar CDLGN is enough on its own — that's the contribution; CNN/BNN baselines are context, not the result.

---

## Stage 6 (optional, follow-up paper) — Event-driven incremental inference

A separate project. The contribution is a runtime/compiler that propagates updates through a trained word-LGN only along gates whose inputs flipped between consecutive TBR words.

When you get there: read Ripple/Ripple++ (incremental GNN inference, 2025) and Multiply-and-Fire (event-driven sparse NN accelerator) to position the contribution. Search for any LGN incremental-inference work that may have appeared in the meantime — this space is moving.

---

## Cross-cutting rules for the agent

- **Reproducibility.** Every experiment seeds Python/NumPy/PyTorch/CUDA. Log the seed, full config, git commit hash, and difflogic commit hash to `experiments/<name>/manifest.json`.
- **Discretization is what matters.** Always report discretized inference accuracy, not soft training accuracy. The gap can be 5–10%.
- **Don't touch difflogic CUDA in Stages 1–4.** The pure-PyTorch path is fine for the word-level extensions; you can write CUDA later if/when the design is validated.
- **Stage gates are non-negotiable.** Even under minimal rigor, the accuracy thresholds and the equivalence test are early-failure detection. If a gate fails, STOP and report. Do not silently weaken acceptance thresholds.
- **One config per experiment.** No CLI overrides for important hyperparameters — they must live in versioned config files so experiments are reproducible later.
- **Keep `difflogic/` pristine.** All modifications go in `src/modules/`. If you must patch difflogic, fork the file and explain in a comment.

## Files the agent should produce, by stage

```
Stage 0:  experiments/00_difflogic_repro.md
          tests/test_difflogic_imports.py
Stage 1:  src/modules/logic_blocks.py
          src/models/logic_classifier.py
          src/data/mnist_dm.py
          configs/exp/01_mnist_lightning.yaml
          experiments/01_mnist_lightning.md
Stage 2:  src/data/tbr.py
          src/data/nmnist_dm.py
          src/data/dvsgesture_dm.py
          configs/exp/02_nmnist_scalar.yaml
          configs/exp/02_dvsgesture_scalar.yaml
          tests/test_tbr_encoding.py
          experiments/02_scalar_cdlgn.md
Stage 3:  src/modules/word_logic.py  (WordLogicLayer)
          tests/test_word_equivalence_forward.py
          tests/test_word_equivalence_backward.py
          configs/exp/03_nmnist_word.yaml
          configs/exp/03_dvsgesture_word.yaml
          experiments/03_word_substrate.md
Stage 4:  src/modules/word_logic.py  (extend with ShiftedWordLogicLayer)
          configs/exp/04_nmnist_shifts.yaml
          configs/exp/04_dvsgesture_shifts.yaml
          experiments/04_shifts_results.md
Stage 5:  (deferred — skip until after Stage 4 positive result)
```