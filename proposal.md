# Implementation Plan: Word-Level Logic Gate Networks for Event Cameras

**Intended reader: a coding agent integrating `difflogic` into a PyTorch Lightning template.**

This plan is staged. **Do not skip stages.** Each stage has an explicit pass/fail gate. If a gate fails, stop and report — do not proceed to the next stage hoping the issue resolves itself.

> **2026-04-28 — proposal v2.** Stages 0 and 1 are already complete and unchanged. Stages 2 / 3 / 4 are revised: the architecture pivots from "encode each event sample as one fused TBR tensor and process it in one forward pass" to a **streaming feature-buffer architecture** — a per-slice encoder, an `[N, M]` shift-register buffer (default `N=32` slices, `M=32` bits per slice), and a decoder whose first layer applies *cross-slice* operators (operators that read multiple rows of the buffer). Training uses Transformer-XL / MeMViT-style detached-memory + per-step loss — **no BPTT**. The original Stage 4 contribution (cross-bit shifts within a 32-bit *spatial* word) is replaced by this cross-slice operator vocabulary. HDC operators are demoted from a planned stage to a **contingency** that fires only if the default `difflogic16` cross-slice operators fail the Stage 4 gate. See §"Streaming feature-buffer architecture (v2)" near the end of this document for the full design.

**Architecture in one paragraph (v2).** A logic-gate **encoder** consumes a single TBR slice (one bin's worth of events) per forward step and emits an `M`-bit feature word. A buffer of shape `[N, M]` holds the most recent `N` such words as a shift register; on each step it is shifted down one row and the encoder writes row 0. A logic-gate **decoder** reads the full buffer and produces class logits. The decoder's first layer pulls from a *pluggable cross-slice operator registry*; default family is `difflogic16` (each output bit picks two buffer rows and one of the 16 standard binary ops, applied bitwise). When `N=1` the buffer collapses to a single word and the architecture reproduces Stage 1's per-frame model — this is the verification anchor for the buffer plumbing.

User configuration choices encoded into this plan:
- **Datasets:** N-MNIST as the small/fast iteration gate, **DVS-Gesture validated alongside it** from Stage 2 onward.
- **Rigor mode:** *Minimal overall.* Strict-rigor on the Stage 3 substrate has been **relaxed in v2** — the multi-`M` matched-accuracy test is dropped because there is no scalar baseline to match against in the streaming setting. Only the `(N=1, M=1)` degenerate forward-equivalence test remains. Comparative baselines (TBR-CNN, TBR-BNN) remain deferred to a post-Stage-4 polish pass.
- **TBR encoding:** Dataloader-side, encoded once and cached to disk using hdf5 format which is modern and managable. **v2 emits per-slice tensors** of shape `[2, H, W]`, with the dataloader returning `[T, 2, H, W]` per sample where `T` is the slice count. The original "fused `[2, num_bins, H, W]` per sample" framing is gone.

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

> *v2 note:* this stage builds the *single-slice* baseline — i.e., the `N=1` degenerate case of the v2 streaming architecture. No edits to the original task list; the result here remains the parity anchor for the streaming Stage 4.

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

## Optimization-tier strategy (v2.1) — read before Stage 2

> **2026-04-29 — proposal v2.1.** The "always detach, no BPTT" commitment from v2 was based on transformer-domain precedent (Transformer-XL / MeMViT). It does not survive contact with **LGN-domain** evidence: **RDDLGN** (Bührer et al., arXiv:2508.06097, Aug 2025) trains a recurrent DLGN with full BPTT through `S=3` timesteps for WMT'14 English-German translation, achieving 5.00 BLEU vs. GRU's 5.41, and explicitly reports "robust and uniform gradient flow throughout all layer groups — no vanishing or exploding gradients." BPTT through an LGN substrate works over short windows. This loosens the design space and reorders the project's optimization commitments.

The v2 streaming-buffer architecture (Stages 3–4 as written) is **one point** in a tier ladder, not the default path. **Try the lowest tier that works.** Each tier costs more engineering but tolerates more optimization difficulty; escalate only when the previous tier fails.

### Tier 0 — Long word, no buffer (try first)

`M=128` (or larger) at coarse bin granularity (e.g., 16ms per bin → ~2 seconds per word). Single fused tensor per sample, single forward pass, full gradient flow. Cross-bit operators (shifts, masks) provide temporal coupling within the word. **This is v1's architecture with a wider word — every advantage that matters is preserved**: tight LGN-specific contribution (cross-bit shift operators are inexpressible as a single scalar gate), no streaming complexity, no detached-memory question, no encoder-grad-starvation risk.

**Cost:** difflogic's int32 packing path is hardcoded; `M=128` requires a `[4 × int32]` representation under the hood (a few days of CUDA work, mechanically the same engineering as v2's "Stage 4.5" for `M ≠ 32`). Training itself is `M`-agnostic and works out of the box.

**Decisive experiment:** Tier 0 on DVS-Gesture, ~1.5 weeks. **If accuracy ≥ 80% with cross-bit shifts contributing meaningfully, the streaming pivot was unnecessary and Stages 3–4 below are deferred to future work.** Run this before anything else in the streaming chain.

### Tier 1 — Long word + small buffer + full BPTT

`M=64`, `N=4..8`. Total temporal range = `M · N · bin_duration` = up to several seconds. **Backprop through all `N` slices** — RDDLGN's `S=3` is direct evidence this is tractable on the LGN substrate. No detached memory anywhere. The "buffer" at `N=4..8` is not really a streaming shift register; it's a stacked input processed across time with shared encoder weights and a decoder reading the lot. Streaming-shaped, but with full gradient flow keeping the optimization tractable.

### Tier 2 — Long word + medium buffer + truncated BPTT

For larger `N` (e.g., `N=16..32`), BPTT-everything becomes expensive in memory. Truncated BPTT propagates gradient through the most recent `k` slices and detaches beyond. **Commit to `k ≥ 4` minimum, not `k=1`.** v2's original `k=1` commitment is the most fragile point of that design and is not supported by LGN-domain evidence. Make `k` a config flag with a sensible floor:

```yaml
tbptt_k: 4    # minimum supported
# k=N    : full BPTT (Tier 1 fallback for small N)
# k=4..8 : standard truncated BPTT (Tier 2 default)
# k=1    : v2's original detach-everything (kept for ablation only)
```

Tier 2 is what the v2 Stage 4 architecture becomes after this update. **Drop the "always detach" non-negotiable from cross-cutting rules.**

### Tier 3 — Encoder pretrain + decoder fine-tune (fallback only)

If after Tiers 0–2 the encoder still fails to learn (verified via gradient-flow diagnostics, see below), pretrain the encoder at `N=1` with single-slice classification (full gradient, easy problem), freeze it, then train the decoder on top of the buffer. Decouples two optimization problems entirely; sidesteps grad-starvation. The encoder cannot co-adapt to the decoder, but you always have a working encoder.

Implement Tier 3 as a code path; **do not use it unless Tiers 0–2 have empirically failed**. Reaching Tier 3 means the project's central optimization story needs revisiting before the paper is written.

### Always-on additions (regardless of tier)

These are cheap, applicable to every tier including Tier 0, and serve as both regularizers and diagnostics. Implement them once in the training loop and gate them with config flags rather than tier-by-tier.

1. **Auxiliary per-slice loss on encoder output.** A small classifier head reading directly from the encoder's output (not the buffer), trained with the same labels. Provides strong gradient signal to the encoder regardless of the buffer-mediated path. Acts as a regularizer in Tier 0 (where encoder grad is already strong) and as a lifeline in Tiers 2–3.

2. **Encoder warm-up.** First 5–10 epochs train at `N=1` only (single-slice classification), then enable the buffer / cross-slice decoder. Bootstraps encoder representations before buffer dynamics make optimization harder. Costs nothing; helps in every tier ≥ 1.

3. **Gradient-flow diagnostics.** Log per-layer-group gradient norms at the end of each epoch (RDDLGN does this in their Table 5). **Diagnostic rule of thumb: if encoder gradient norm is more than 10× smaller than decoder gradient norm at convergence, the encoder is not learning, regardless of what the loss curve says.** This single diagnostic will save weeks of misdiagnosed failures and is non-optional for Tiers ≥ 1.

### The unique inductive-bias contribution — *encoding a temporal structure prior*

> **This is the project's clearest claim to LGN-architecture novelty in the streaming setting and should be foregrounded in the paper if Stages 3–4 produce results.**

A scalar CDLGN with random connectivity over a flattened `[N · M]` buffer can in principle learn temporal patterns, but the connectivity prior is uniform — any input bit is as likely to be sampled as any other. The decoder has to *learn* the temporal alignment via lucky connections. This is hard and statistically wasteful when `N · M` is large.

**The cross-slice operator family encodes a temporal structure prior into the connectivity itself.** Each output bit's two inputs are constrained to be *row-aligned* — both come from the same bit position across two (potentially different) slice rows of the buffer. This forces the layer to express temporal comparisons cleanly: "compare slice `t` to slice `t-k` at corresponding bit positions." The decoder no longer has to learn the alignment; the alignment is structural.

This is a real (if modest) inductive-bias contribution and it is **specific to the LGN substrate**. In a CNN or transformer, positional information is added via embeddings or convolutional structure — soft, learned mechanisms layered on top of a generic computation. In an LGN, you cannot add a positional embedding (there are no real-valued representations to embed into); you can only constrain the *connectivity* to encode positional structure. The cross-slice operator's row-aligned input pairing is the LGN-native form of a temporal positional prior.

Two extensions that make this contribution sharper:

- **Per-row positional bits.** Concatenate a small learned tag (4–8 bits) to each buffer row indicating its temporal position. Costs almost nothing; lets the decoder learn position-aware operators ("when slice age is `≤ k`, do X; otherwise Y") via the standard 16-op vocabulary. The LGN-native analogue of transformer positional embedding.
- **Diagonal / strided connectivity priors.** Beyond the default `(i, j)` row pair sampling, expose connectivity priors that bias toward small `|i - j|` (local temporal coupling) or fixed-stride pairs (`j - i = k` for learned `k`). Each prior is a hypothesis about what kind of temporal structure matters; ablating across them is a clean experimental design.

If Stage 4 succeeds, the headline framing of the paper is **not** "LGNs work for event cameras" (too generic) and **not** "we built a streaming LGN" (architecture-only, generic across model classes). The headline is **"We introduce a temporal-structure prior for LGNs that is inexpressible in CNNs and transformers because it operates on connectivity rather than embeddings, and we show this prior delivers measurable inductive bias on event-camera data."** That framing is LGN-specific, event-camera-specific, and falsifiable — the right shape for a contribution.

### How this updates the staged plan below

- **Stage 2** is unchanged — the per-slice TBR encoding and concat baseline serve all tiers.
- **Stage 3** is unchanged — the word substrate is required regardless of tier.
- **Stage 4** as written below is **Tier 2 with `k=1`**. Update before executing: (a) try Tier 0 first as a one-shot decisive experiment (`M=128`, no buffer); (b) if Tier 0 fails, run Stage 4 but with `k=4` minimum and always-on additions enabled; (c) keep Tier 3 as a fallback code path.
- **The "Detached-memory pattern is non-negotiable" rule in cross-cutting rules is dropped.** The detach-vs-BPTT choice is now a tunable config (`tbptt_k`) with `k ≥ 4` as the default for Tier 2.

---

## Stage 2 — Event datasets + per-slice TBR + concat baseline (target: 1.5 weeks)

> *v2 reframe.* The dataloader now emits **per-slice** TBR tensors (one slice = one bin's worth of events), not a single fused per-sample tensor. The Stage 2 scalar-CDLGN baseline becomes the *concat baseline*: stack all `T` per-slice tensors along the channel dimension and run one forward through scalar CDLGN. This is the temporal-aggregation lower bound that the v2 Stage 4 streaming buffer architecture must beat.

### Goal
Scalar-CDLGN concat baseline trained on per-slice-TBR-encoded **N-MNIST and DVS-Gesture**. These are the baselines the v2 streaming architecture must beat. Not a paper, just numbers to point at.

DVS-Gesture is included now (rather than in Stage 4) because N-MNIST has weak temporal structure — saccade-induced events on a static digit. Validating only on N-MNIST risks reaching Stage 4 without ever testing the cross-slice operator hypothesis on data with real temporal structure.

### Tasks

1. **Add event-camera dependencies.** `pip install tonic` (https://github.com/neuromorphs/tonic) handles N-MNIST/DVS-Gesture/N-Caltech101 with a unified API.

2. **Implement TBR encoding** in `src/data/tbr.py` *(already complete; tests pass on CPU)*:
   - Function `encode_tbr(events, num_bins=N, bin_duration_us=…, sensor_size=(H, W, 2)) -> torch.Tensor` returning a `[2, num_bins, H, W]` boolean tensor (polarity × bins × spatial).
   - Per pixel per polarity per bin: 1 if any event in that bin, else 0.
   - Vectorize using `torch.bincount` or scatter operations. Do NOT use a Python loop over events — N-MNIST has ~5k events per sample, gesture data has ~1M+. A loop is unusable.
   - Unit test in `tests/test_tbr_encoding.py`: round-trip a synthetic event stream and assert the resulting bin tensor matches the hand-computed expected output. Edge cases: empty events, events outside time window, multiple events in same bin.

3. **Per-slice slicing in the DataModules.** The encoder receives one slice at a time during streaming (Stage 4); the concat baseline at this stage stacks all slices into one tensor. Both consumers want the same `[T, 2, H, W]` boolean tensor at the dataloader boundary.
   - `src/data/nmnist_dm.py`: 34×34 sensor, 2 polarities. Default `T=N=32` slices. `bin_duration_us` chosen so that `T · bin_duration_us` covers the full sample (~300ms). Per-slice shape `[2, 34, 34]`; per-sample `[32, 2, 34, 34]`.
   - `src/data/dvsgesture_dm.py`: 128×128 sensor (consider downsampling — see note below), 2 polarities. Fixed sample window length 500ms; default `T=N=32`, so `bin_duration_us = 500_000/32 ≈ 15625`. Pad/crop short and long event streams to the window.

4. **Cache encoded tensors to disk.** Encode once into `data/<dataset>_tbr_<T>x<bin_us>us_<resolution>/` as hdf5 (recommended) or `.pt` per sample. The DataLoader reads cached tensors. Document the cache directory structure and encoding parameters in a manifest file so re-encoding with different parameters doesn't silently overwrite.

5. **Note on DVS-Gesture spatial resolution.** Full 128×128 with `T=32` means 2·32·128·128 = 1,048,576 input bits, which is large for an LGN — but the *concat baseline* sees this whole tensor in one forward; the streaming Stage 4 architecture only sees `[2, H, W]` per slice (= 32,768 bits at full 128×128, or 2,048 bits at 32×32). Two starting points for the concat baseline:
   - Downsample to 32×32 spatial (max-pool the encoded TBR tensor): 65,536 input bits with `T=32`, comparable to N-MNIST. **Recommended for first run.**
   - Keep 128×128 with smaller hidden_dim and fewer layers to manage compute. Document the choice and revisit later if accuracy is poor.
   - For future utility, let's *also* encode 128×128 full resolution to disk so streaming experiments at higher spatial resolution don't require re-encoding.

6. **Train (concat baseline).** For each dataset, the model is a scalar CDLGN consuming the flattened `[T·2·H·W]` per-sample tensor:
   - Hidden_dim 32k–64k, num_layers 5–6.
   - Acceptance (loose; revisit after first run): discretized test accuracy ≥ 95% on N-MNIST, ≥ 80% on DVS-Gesture (11-class). These are *floors*; the real gate is "v2 Stage 4 beats this number".

7. **Log baseline numbers** in `experiments/02_scalar_cdlgn.md`: per-dataset test accuracy, gate count, inference throughput on one CPU core (use difflogic's `CompiledLogicNet` if accessible), training time, GPU memory.

### Gate
- [x] TBR encoder unit-tested *(already complete)*
- [ ] N-MNIST per-slice DataModule + concat baseline ≥ 95%
- [ ] DVS-Gesture per-slice DataModule + concat baseline ≥ 80% (32×32 downsample is fine)
- [ ] Baseline numbers logged for both datasets

If N-MNIST plateaus low: try doubling hidden_dim, longer training, or verify the TBR tensor isn't degenerate (mostly zeros suggests bin width is too short or sensor_size is wrong). If DVS-Gesture plateaus very low (e.g., near random 9%): the input is probably too high-dimensional — downsample more aggressively. Don't proceed past these gates — if scalar CDLGN can't clear them on the concat baseline, the streaming Stage 4 won't either.

---

## Stage 3 — Word-level LGN as the buffer substrate (target: ~1 week, v2-relaxed)

> *v2 reframe.* The `[N, M]` shift-register buffer that the Stage 4 streaming architecture uses *is* this stage's word substrate — the storage layout is identical. The strict-rigor multi-`M` matched-accuracy test is dropped because there is no scalar reference for the *streaming* setting (you can't run scalar CDLGN frame-by-frame and ask it to track a buffer of past features). Only the `(N=1, M=1)` degenerate forward-equivalence test remains as the parity anchor.

### Goal
A word-level `WordLogicLayer` that processes `[B, N, M]` packed-bit tensors via the same 16 bitwise functions, with `(N, M)` as **independent** knobs. Default `(N=32, M=32)` matches difflogic's existing int32 packing path so fast inference works out of the box; the layer is correct for arbitrary `(N, M)` at training time. This is the substrate the Stage 4 streaming buffer architecture builds on.

### Tasks

1. **Create `src/modules/word_logic.py`** with:
   - `WordLogicLayer(nn.Module)`: input shape `[B, in_features, M]` where `M` is the bit-width per slice (32 default). Internally maintains a `(out_features, 16)` softmax over the 16 binary functions, just like difflogic. The forward pass applies the chosen function bitwise across the `M` dimension.
   - **`(N, M)` independence:** `N` (slice count) lives in the dataloader / buffer and `M` (bits per slice) lives in the layer. This file owns `M`; the buffer (Stage 4) owns `N`.
   - During training: real-valued relaxation over `[0,1]`, applied independently per bit position. The per-bit forward is identical to difflogic; we're just batching across `M`. **Trivially `M`-agnostic.**
   - During inference: at `M=32`, use bitwise integer ops on packed int32 (difflogic's existing fast path). For `M ∈ {8, 16, 64}`, accuracy is reported but fast inference is **out of scope here** — see proposal v2 §"Stage 4.5".
   - For final readout: simplest path is to unpack words into bits at the very end and reuse difflogic's `GroupSum` directly. Pick this; do not optimize.

2. **Forward-pass equivalence test (discretized, single anchor).** `tests/test_word_equivalence_forward.py`:
   - Fix connectivity (random but seeded). Fix operator choices (one-hot over the 16 ops per neuron, pre-discretized).
   - Generate random binary input of shape `[B, in_features, M]`.
   - Compute forward two ways: (a) `WordLogicLayer` on the packed input, (b) `LogicLayer` from difflogic run `M` times on each bit-slice, then re-stacked.
   - Assert bit-for-bit equality.
   - Run with at least 3 different seeds and `M ∈ {1, 8, 32}` to catch off-by-one and packing bugs.
   - **The `M=1` case is the load-bearing parity check** — at `M=1` the WordLogicLayer is supposed to be exactly `difflogic.LogicLayer`.

3. ~~Backward-pass equivalence test~~ — **dropped in v2.** Without a meaningful scalar reference for the streaming pipeline, this test would only re-validate the same bit operations the forward test already covers. The `M=1` forward equivalence is the parity anchor; deeper gradient checks are deferred until/unless Stage 4 fails the gate and we need to localise.

4. ~~Matched-accuracy training on both datasets~~ — **dropped in v2.** No scalar baseline exists in the streaming setting; the only meaningful matched-accuracy test (`N=1` reproducing Stage 1's 97.36%) lives in the Stage 4 verification block.

5. **Throughput and memory characterization.** On a single CPU core, measure inference samples/sec for `LogicLayer` (Stage 1 single-frame baseline) vs `WordLogicLayer` (`M=32`). Use difflogic's `CompiledLogicNet` if accessible. Also record peak GPU memory during training at `(N, M) ∈ {(1, 32), (32, 32)}`. The word version should be substantially faster at inference on bit-packed inputs and roughly similar in training memory.

6. **Connectivity-init audit.** difflogic uses pseudorandom fixed connectivity. Verify your `WordLogicLayer` uses the *same* connectivity-generation seed and procedure as difflogic for the `M=1` parity comparison in task 2. A different RNG path here will produce different (but equally valid) networks, spuriously failing parity. Document the seed and procedure in the test.

### Gate
- [ ] `test_word_equivalence_forward` passes (bit-for-bit, multiple seeds, `M ∈ {1, 8, 32}`)
- [ ] Throughput and training-memory numbers logged for `(N, M) ∈ {(1, 32), (32, 32)}` in `experiments/03_word_substrate.md`
- [ ] Connectivity-init audit documented (seed and procedure verified to match difflogic)
- [ ] Document `(N, M)` as independent config knobs

Do NOT add new operators in this stage. Cross-slice operators live in the Stage 4 decoder, not in this layer. The whole point of separating substrate (here) from operators (Stage 4) is that when Stage 4 produces a result, you can attribute it cleanly to the operator vocabulary, not to a buggy substrate.

---

## Stage 4 — Streaming feature buffer + cross-slice operators (v2 contribution) (target: 2.5–3 weeks)

### Goal
Build the streaming architecture: per-slice encoder → `[N, M]` shift-register buffer (default `N=32`, `M=32`) → decoder with a *cross-slice operator* in its first layer. Train end-to-end with **detached-memory + per-step loss** (Transformer-XL / MeMViT pattern; no BPTT). Test the hypothesis that cross-slice operators beat the Stage 2 concat baseline on event data — particularly on DVS-Gesture, where the temporal structure is real.

### Architecture

```
                                        ┌──────────┐
TBR slice  ─►  Encoder E (logic-gate) ─►│  B[0, :] │   row 0 (newest, grad-attached)
   [2,H,W]                              ├──────────┤   B : [N, M]
                                        │  B[1, :] │   detached
                                        │    ...   │
                                        │  B[N-1,:]│   detached (oldest, about to fall off)
                                        └────┬─────┘
                                             │
                                             ▼
                                       Decoder D (logic-gate)
                                       — cross-slice operator in layer 0 —
                                             │
                                             ▼
                                          GroupSum(k=num_classes)
                                             │
                                             ▼
                                           logits_t
```

### Tasks

1. **Implement the buffer (`src/modules/buffer.py`).** A small module owning a `torch.Tensor` of shape `[N, M]` (booleans during inference, soft-relaxed floats during training). Each `step(f_t)` call:
   - Detaches the current buffer (`B = B.detach()`).
   - Shifts down: `B = torch.roll(B, shifts=1, dims=0)`.
   - Clones (so the in-place write below doesn't break autograd elsewhere).
   - Writes `B[0, :] = f_t` — only this row carries gradient back to the encoder.
   - Returns the new `B`.
   - Owns a `reset()` that zeroes the buffer at the start of each sample.

2. **Implement the cross-slice operator registry (`src/modules/cross_slice_ops.py`).** A registry keyed by family name. Each family declares `arity`, `vocabulary_size`, an inference `apply(buffer, idx_a, idx_b, op_idx)`, and a soft-relaxed `relax(buffer_soft, op_logits)`. Register the default family `difflogic16`: each output bit picks two slice rows `(i, j) ∈ [0, N)` and one of the 16 standard binary ops, applied bitwise across the `M`-bit dimension to `B[i, :]` and `B[j, :]`. **Start with the pruned form** that fixes `j = 0` (latest-vs-history pair-up) — vocabulary size `16 · N`, much cheaper than `16 · N²` and easier to interpret.

3. **Implement the streaming classifier (`src/models/streaming_classifier.py`).** A `LightningModule` that owns the encoder, buffer, and decoder. `training_step` runs the per-sample loop:
   ```
   buffer.reset()
   loss_acc = 0
   for t in range(T):
       f_t = encoder(x[t])                    # grad-attached
       B = buffer.step(f_t)                   # only B[0,:] carries grad
       logits_t = decoder(B)                  # grad to D and to f_t
       if t >= warmup_steps:
           loss_acc += CE(logits_t, y)
   loss = loss_acc / max(T - warmup_steps, 1)
   ```
   Default `warmup_steps = N` (don't backprop until the buffer is fully populated). Make this a config flag (could also be `0` for "loss at every step" or `T-1` for "loss only at end-of-sample").

4. **Discretized correctness test for the cross-slice operator.** Verify that the discretized inference of a single `difflogic16` cross-slice neuron matches a hand-computed reference for a few `(op_idx, i, j)` configurations on synthetic random buffers. Cheap insurance; do not skip.

5. **Buffer mechanics tests** (`tests/test_buffer_mechanics.py`):
   - After `T` calls to `buffer.step(...)`, `B[0, :] == encoder(x_T)` and `B[t, :] == encoder(x_{T-t})` for `t < min(T, N)`.
   - After step, `B[1:, :].requires_grad == False` (detached past).
   - `loss.backward()` produces non-zero gradient on encoder parameters at every step (verified by hooking into the encoder's grad).

6. **Single-slice degenerate case.** With `N=1`, the v2 streaming model trained on single-slice MNIST must reproduce **Stage 1's 97.36% within ±0.3%**. This catches bugs in the buffer plumbing on a problem whose answer we already know.

7. **Train on N-MNIST.** Architecture: encoder (small LogicLayer stack, `M=32` output) + buffer `(N=32, M=32)` + decoder (cross-slice `difflogic16` first layer, then standard `WordLogicLayer` stack, then `GroupSum`). Compare against Stage 2's concat baseline:
   - Streaming-with-cross-slice-ops vs concat-baseline at *iso-gates*.
   - Streaming-with-cross-slice-ops vs concat-baseline at *iso-flops*.
   - Ablation: streaming-without-cross-slice-ops (decoder layer 0 = identity / fixed `op = LATEST`) — isolates the contribution from the streaming structure alone.

8. **Train on DVS-Gesture.** Same comparison. **This is where the central hypothesis gets tested.** DVS-Gesture has real temporal structure (gestures evolve over hundreds of ms) so cross-slice ops should help if the inductive bias hypothesis is right at all.

9. **`(N, M)` ablation matrix.** Sweep `N ∈ {8, 16, 32, 64, 128}` with `M=32` (full inference path), and `M ∈ {8, 16, 32, 64}` with `N=32` (training-only for `M ≠ 32`; document the fast-inference gap as the trigger for optional Stage 4.5). Report accuracy + parameter count + (where applicable) inference throughput. This shows the architecture is general for arbitrary `(N, M)`, not tied to 32.

10. **Critical analysis: operator distribution.** Log a histogram of which `(i, j, op_idx)` combinations the trained decoder picks. If cross-slice ops are rarely chosen (e.g., `j = 0` collapses to identity-on-latest in > 90% of neurons) on DVS-Gesture, that's strong evidence against the inductive bias hypothesis. Confront that finding rather than push past it.

11. **Decision point.** After Stage 4 results are in, evaluate honestly:
    - **Cross-slice ops beat the concat baseline on DVS-Gesture (iso-gates):** the project has a real result. Move to Stage 5 to formalise.
    - **Cross-slice ops don't help on either dataset:** trigger the **Contingency: HDC operator vocabulary** below before pivoting away from the architecture. If HDC also fails, the central thesis is wrong as stated; acknowledge the negative result.
    - **Cross-slice ops help on DVS-Gesture but not N-MNIST:** consistent with the inductive bias story (temporal ops help when temporal structure exists, not on essentially-static data). This is a *good* outcome and a tighter framing than "they help everywhere".

### Gate
- [ ] `cross_slice_ops.difflogic16` unit-tested for discretized correctness
- [ ] Buffer mechanics tests pass (shift order, detached gradient, encoder grad once-per-step)
- [ ] `N=1` degenerate case reproduces Stage 1's 97.36% within ±0.3%
- [ ] N-MNIST and DVS-Gesture streaming results both logged (iso-gates and iso-flops)
- [ ] `(N, M)` ablation matrix logged
- [ ] Operator-choice histogram saved
- [ ] Honest decision documented in `experiments/04_streaming_buffer.md`

This is the **make-or-break** stage of the project.

---

## Stage 4.5 (optional engineering follow-up) — Arbitrary-`M` fast inference

Triggered only if Stage 4 succeeds *and* the `M ≠ 32` accuracy ablation shows interesting patterns (e.g., `M=64` substantially improves accuracy at modest gate-count overhead). Extend difflogic's `tensor_packbits_cuda_kernel` so the inference fast path supports `M ∈ {8, 16, 64}` natively (one-day patch per `M`, multiple kernels). Without this stage, `M ≠ 32` accuracy numbers are reported but their fast-inference numbers are not.

---

## Contingency — HDC operator vocabulary (NOT a numbered stage)

Triggered only if Stage 4's `difflogic16` cross-slice ops fail to beat the Stage 2 concat baseline on either dataset. **Last-resort tool**, not a planned experiment.

Add an `hdc` family to the cross-slice operator registry:
- `xor_bind`: each output bit reads two slice rows `B[i, :]` and `B[j, :]` and computes `B[i, :] XOR B[j, :]`. This is HDC's *binding* primitive — it produces a representation in which knowing one operand and the result lets you recover the other.
- `bit_majority_bundle`: each output bit reads `K ≥ 3` slice rows `B[i_1, :], ..., B[i_K, :]` and computes the bit-wise majority. This is HDC's *bundling* primitive — superpose multiple slices into one summary representation.

Retrain the decoder with the HDC family and compare against Stage 4's `difflogic16` numbers. Decision point: if HDC ops also don't help, acknowledge the negative result on the architectural hypothesis and consider pivoting away. Do not keep adding operator families indefinitely.

---

## Stage 5 (deferred under minimal rigor) — Comparative baselines and Pareto plot

Skipped from the critical path. Run after Stage 4 produces a positive result, before paper submission. At that point: add TBR-CNN and TBR-BNN baselines (small models, fair comparison, not SOTA-chasing), generate the gates-vs-accuracy Pareto plot across all five model families, write up.

If you need to make a quick external case for the project before this stage (e.g., for a workshop deadline), the Stage 4 iso-gates comparison against scalar CDLGN is enough on its own — that's the contribution; CNN/BNN baselines are context, not the result.

---

## Stage 6 (optional, follow-up paper) — Event-driven incremental inference

> *v2 reframe.* The streaming buffer architecture from v2 Stage 4 is *already* incremental by construction — each forward step processes only one new TBR slice and updates one row of the buffer. Stage 6 therefore shifts from "build incremental inference from scratch" to "exploit the architectural fact for runtime gains" — an inference-runtime contribution that propagates updates only through gates whose inputs actually flipped between consecutive slices.

When you get there: read Ripple/Ripple++ (incremental GNN inference, 2025) and Multiply-and-Fire (event-driven sparse NN accelerator) to position the contribution. Search for any LGN incremental-inference work that may have appeared in the meantime — this space is moving.

---

## Streaming feature-buffer architecture (v2)

> Detailed design for the v2 architecture used in Stages 3–4.

**Why pivot from v1.** The original Stage 4 contribution (cross-bit shifts within a 32-bit *spatial* word) treated each event sample as a single fused TBR tensor and processed it in one forward pass. This loses the streaming nature of event data and makes Stage 6's incremental-inference contribution feel bolted-on. The v2 architecture instead processes events one slice at a time, holds the last `N` slice features in a shift register, and lets the decoder reach across the buffer with cross-slice operators. Each forward step touches only one new slice; the architecture is incremental by construction.

**Prior art (no-BPTT training).** The "detached memory + per-step loss" pattern is the canonical no-BPTT approach in language and video modelling:
- **Transformer-XL** (Dai et al. 2019, [paper](https://aclanthology.org/P19-1285.pdf)) — caches a sequence of hidden states across segments, applies attention to cached + current, but **stops gradients at the cache boundary**.
- **MeMViT** (Wu / Feichtenhofer, CVPR 2022, [paper](https://arxiv.org/abs/2201.08383)) — vision-domain analogue. Caches "memory" online during inference and training; achieves 30× longer temporal support than non-streaming baselines with only 4.5% more compute. Memory is detached; per-segment loss; no BPTT.

The v2 Stage 4 architecture uses exactly this paradigm.

**Two independent axes.** `N` (slice history depth) and `M` (bits per slice) are independent and the architecture is general for arbitrary `(N, M)`. Defaults `(32, 32)` line up with difflogic's existing int32 packing path. The cross-`(N, M)` ablation matrix in Stage 4 task 9 demonstrates generality.

**M as a parameter — what's general, what's `M=32`-specific:**

| Path | Constraint on `M` | Implementation cost |
|---|---|---|
| Training (real-valued relaxation) | **Any positive int.** Per-bit relaxation; `M` is just a tensor shape. | Zero — works out of the box. |
| Inference, slow path (per-bit float ops) | **Any positive int.** | Zero. |
| Inference, fast path (bit-packed int32 ops) | **Hardcoded `M=32`** in difflogic's `tensor_packbits_cuda_kernel`. | Non-trivial; gated as Stage 4.5 if needed. |

This gives a clean three-tier story: the architecture-level claim (default `M=32`, full inference path), the generality claim (cross-`M` accuracy ablation, training-only for `M ≠ 32`), and the engineering follow-up (Stage 4.5).

**Concerns flagged for the implementer:**

1. **Sparse encoder gradients.** With detached buffer the encoder receives gradient at every step but only via row 0; effective gradient signal is `1/N` of the decoder's gradient. If the encoder fails to learn, mitigations are: (a) per-slice classification warm-up (pretrain encoder for K epochs at `N=1` before adding the buffer), (b) auxiliary per-slice loss on encoder output. **Do not** fall back to BPTT — the no-BPTT commitment is architectural.

2. **Per-step vs per-sample loss ambiguity.** Per-step loss is more informative early in training but "leaks" the label across slices. For DVS-Gesture (non-IID-across-time within a sample) per-step makes sense; for N-MNIST (IID-static-digit-with-saccade) it might be redundant. Make `warmup_steps` and `loss_at_every_step` config flags.

3. **No iso-gates scalar reference for Stage 4.** The original v1 comparison (scalar CDLGN vs word-with-shifts at iso-gates) is gone in the streaming setting. Replacement is the Stage 2 *concat baseline* (which is also scalar CDLGN, just over the concatenated `[T·2·H·W]` tensor) — apples-to-apples by parameter budget but apples-to-oranges by inference cost. Document both numbers.

4. **DVS-Gesture variable sample length.** Sample lengths range from ~500ms to ~2s. Default plan: fix `N=32` slices per sample at `bin_duration_us = sample_window_us / N`, padding short samples with zeros at the start.

## Cross-cutting rules for the agent

- **Reproducibility.** Every experiment seeds Python/NumPy/PyTorch/CUDA. Log the seed, full config, git commit hash, and difflogic commit hash to `experiments/<name>/manifest.json`.
- **Discretization is what matters.** Always report discretized inference accuracy, not soft training accuracy. The gap can be 5–10%.
- **Don't touch difflogic CUDA in Stages 1–4.** The pure-PyTorch path is fine for the word-level extensions; you can write CUDA later if/when the design is validated.
- **Stage gates are non-negotiable.** Even under minimal rigor, the accuracy thresholds and the equivalence test are early-failure detection. If a gate fails, STOP and report. Do not silently weaken acceptance thresholds.
- **Truncated-BPTT depth is a tunable, with a floor (v2.1).** v2's original "always detach, no BPTT" rule is replaced. `tbptt_k` is a config flag with `k ≥ 4` minimum for Tier 2 default; `k = N` is the Tier 1 fallback (full BPTT, viable for small N per RDDLGN evidence); `k = 1` (the original v2 setting) is retained as an ablation only. If Tier 2 with the always-on additions still fails to train the encoder, escalate to Tier 3 (encoder pretrain + frozen-encoder decoder fine-tune) — *not* to BPTT removal.
- **One config per experiment.** No CLI overrides for important hyperparameters — they must live in versioned config files so experiments are reproducible later.
- **Keep `difflogic/` pristine.** All modifications go in `src/modules/`. If you must patch difflogic, fork the file and explain in a comment. *Exception:* Stage 0's `AT_DISPATCH_*` patch in `difflogic_kernel.cu` is required for PyTorch 2.8 compatibility and is documented in `patches/difflogic_pytorch28_scalar_type.patch`.

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
Stage 3:  src/modules/word_logic.py  (WordLogicLayer; M-agnostic)
          tests/test_word_equivalence_forward.py  (M=1 parity anchor)
          experiments/03_word_substrate.md
Stage 4:  src/modules/buffer.py                  (NEW — [N, M] shift register)
          src/modules/cross_slice_ops.py         (NEW — registry; default difflogic16)
          src/models/streaming_classifier.py     (NEW — encoder + buffer + decoder)
          tests/test_buffer_mechanics.py
          tests/test_cross_slice_ops.py
          configs/exp/04_nmnist_streaming.yaml
          configs/exp/04_dvsgesture_streaming.yaml
          experiments/04_streaming_buffer.md
Stage 4.5: (optional, only if Stage 4 succeeds AND M ≠ 32 ablation is interesting)
          patches/difflogic_packing_M*.patch     (extends packbits kernel for M ∈ {8,16,64})
Contingency (HDC):
          src/modules/cross_slice_ops.py         (extend with hdc family)
          configs/exp/04_*_hdc.yaml
          experiments/04_streaming_buffer.md     (append HDC results to same file)
Stage 5:  (deferred — comparative TBR-CNN/BNN baselines, only after Stage 4 positive result)
Stage 6:  (optional follow-up — incremental inference runtime)
```