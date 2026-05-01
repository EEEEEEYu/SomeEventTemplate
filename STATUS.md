# STATUS

## Current phase
**Phase 1 closed (gate not formally met; documented-honest-gap accepted, see below).
Moving to Phase 2 — event-flow pipeline de-risking.**

## Phase progress
- [x] Phase 1: CDLGN reproduction on CIFAR-10 — best M = **65.99%** vs paper 71.01%
      (gate ≥69% missed by 3.0 pp; -5.0 pp from paper). Closed with documented gap
      per proposal §Phase 1 honesty rule; remaining gap is generalization, not
      capacity or undertraining (see findings + table below).
- [ ] Phase 2: Event-flow pipeline de-risking — **BLOCKED on dataset upload**.
- [ ] Phase 3: Full flow baseline
- [ ] Phase 4: BFS novelty exploration

## Active work — Phase 2 readiness
Per proposal v3 §Phase 2, we need a small flow benchmark for the "does it work at
all" experiment. Recommended (in order of preference for fast iteration):

1. **MVSEC** (Multi-Vehicle Stereo Event Camera dataset) — `indoor_flying1` or
   `outdoor_day1` sequence is enough for Phase 2. ~3–10 GB per sequence. GT
   optical flow available. Standard entry point in event-flow literature.
2. **DSEC-Flow** — modern SOTA benchmark, ~100+ GB. Heavier; better for Phase 3.
3. **A downsampled subset of either** — works fine for Phase 2.

**Suggested upload location:** `data/mvsec/<sequence_name>/` (gitignored already).
Format-flexible — common public packagings are HDF5 (MVSEC) or rosbag → HDF5
conversion. Once a sequence is uploaded, `src/data/event_flow_dm.py` and a
small `cdlgn_flow.py` head module follow the existing CDLGNClassifier pattern.

## Blockers
- **2026-04-30: Phase 2 blocked on dataset upload.** No event-flow data on disk yet.

## Recent results

### Phase 1 sweep summary (CIFAR-10, AdamW lr=0.02, wd=0.002)

| Config                                            | S test  | M test  |
|---------------------------------------------------|---------|---------|
| Plain (no groups, 6 ep, no cosine)                | 51.74%  | 51.59%  |
| + conv `channel_groups=k/8`, 20 ep                | 56.20%  | 65.99%  |
| + cosine LR (T=40), 40 ep                         | 56.83%  | 65.25%  |
| + dense1/2 grouped, dense3 ungrouped, 40 ep       | 55.84%  | **65.51%** |
| All 3 dense layers grouped (no recombination)     | 44.02%  |   —     |
| **Paper**                                         | 60.38%  | 71.01%  |
| **Phase-1 gate (paper − 2 pp)**                   | (58.4)  | **69.0** |

Best M run **65.99%**; Phase-1 gate (≥69%) **NOT MET** (-3.0 pp).

### Findings
- **Channel grouping** (paper §A.3, k/8 throughout the network) verified
  structurally: dense1/2 grouped + dense3 ungrouped lands within ±0.5 pp of
  the un-grouped baseline → confirms paper's "does not reduce accuracy"
  claim, but also means **this divergence was not the source of the gap**.
- **Cosine LR schedule**: no-op (-0.7 pp on M, +0.6 pp on S), within noise.
- **`_apply_bin_op_s` bilinear-form rewrite**: numerically equivalent to
  `difflogic.functional.bin_op_s` (unit-tested) → not a regression source.
- **Block-1 RF fix** (5×5 → 3×3 per paper §A.1.1): +0 pp.
- **Capacity**: M's train_acc reaches 96–98% (plenty of capacity), val_acc
  stuck at 65–66% → the gap is **generalization**, not optimization or capacity.
- **Training duration**: 40 epochs × 352 steps = 14,080 steps, ~7× the paper's
  2,000-step budget. Not undertrained.

### Remaining hypotheses (deferred — revisit after Phase 2 produces a flow
   number, since the same backbone is reused)
1. **Discretization gap.** Test accuracy is `model.eval()` (hard one-hot per
   gate). If soft accuracy is meaningfully higher, τ tuning is the issue.
   Cheap to verify by logging both.
2. **Random-connectivity seed luck.** difflogic's `connections="random"` is
   set once at construction. Try 3 seeds.
3. **Some subtle architectural detail** in the conv tree forward we have not
   spotted yet (e.g., paper's actual gate-forward ordering inside the tree,
   or an undocumented training detail like LR warmup).

### Phase 1 lock-in
Configs frozen at the **paper-faithful** recipe (dense1/dense2 grouped,
dense3 ungrouped, cosine LR T=40, channel_groups=k/8). This is within ±0.5 pp
of the absolute-best M run (65.99%, no dense grouping, 20 epochs no cosine)
and is the cleanest baseline to inherit into Phase 2 — same backbone class,
just different head.
- **2026-04-30 _apply_bin_op_s ↔ difflogic.bin_op_s equivalence test:** PASSED.
  Bilinear-form rewrite is numerically identical → not a regression source.
- **2026-04-30 M run (4-GPU DDP, bilinear bin_op_s, bs=32×4=128):** test_acc =
  **51.59%** after 6 epochs. Paper M 71.01%; gate ≥69%. **MISSED gate by 17.4 pp.**
  Train_acc 69.0% / val_acc 52.0% → overfit but val plateaued near S. ~57 s/epoch.
- **2026-04-30 S re-run (block-1 3×3 fix + 4-GPU DDP):** 51.74%. Block-1 fix
  had ~0 effect.
- **2026-04-30 S sanity (1st pass, buggy block-1 5×5):** test_acc 51.66%.
- **2026-04-30 smoke (stub backbone):** 33.2% in 2 epochs → pipeline working.
- **2026-04-30:** 26 CPU tests pass (CDLGN module correctness, thermometer
  encoding, CIFAR-10 datamodule shapes, gradient-norm logger).
- **Infrastructure 2026-04-30:** `PlainTextProgress` (newline-per-update); 4-GPU
  DDP wired up with `_ensure_difflogic_tensors_on_device` hook in
  `CDLGNClassifier` working around difflogic's unregistered `indices` /
  `given_x_indices_*` attrs; bilinear-form `_apply_bin_op_s` (16 op-tensors
  per tree level → 4) to fit M in 48 GB at full bs=32 per GPU.

## Phase-1 status: GATE NOT MET
Both S (k=32) and M (k=256) plateau at ~52% test acc — capacity-independent.
Strongly suggests the issue is NOT undertraining or model size; something
about the architecture or training recipe systematically caps generalization.

### Hypotheses to investigate (next session)
Ordered by suspected impact + effort to test:

1. **No channel grouping (paper §A.3 uses k/8).** Without it the ConvLogicLayer
   may be effectively disconnected — leaves sampled uniformly across the full
   RF can ignore most of the input channels. Implementing `channel_groups`
   per the paper's k/8 convention could change the regularization profile.
   **Effort:** small (already supported by `ConvLogicLayer`; just set it in
   the YAML and re-run).
2. **`_apply_bin_op_s` bilinear-form equivalence to `difflogic.functional.bin_op_s`.**
   The rewrite should be exact, but no numerical equivalence test exists. Add
   one before relying on the rewrite.
   **Effort:** trivial (one unit test).
3. **Train longer.** 6 epochs (~2,100 steps) matches the paper budget but val
   was still climbing slowly. Try 20–40 epochs. **Effort:** trivial (config edit).
4. **Eval-time discretization gap.** I haven't separately reported soft vs
   hard accuracy on val. If the soft accuracy is much higher than reported
   test (which is hard via `model.eval()`), the gap is in the discretization
   sharpness, suggesting τ tuning is needed.
   **Effort:** small (add a soft-acc log).
5. **OR-pool relaxation.** Paper says "max t-norm" which is `F.max_pool2d`. I
   use exactly that. Sanity-check by ablating OR-pool to plain max-pool to see
   if anything moves; should be identical.
6. **Subtle architectural mismatch.** Re-read §A.1.1 with the implementation
   open side-by-side and verify each layer dim, padding, and connectivity rule.

### What's solid
- Pipeline: data, Lightning, DDP, manifest, checkpointing, plain-text logging.
- Phase 1 file scaffold landed and tested (26 CPU tests pass).
- Repo cleaned of v1/v2/v2.1 dead code.

## Phase-1 root-cause findings (S gap)

**2026-04-30 paper re-read of §A.1.1.** Caught one architecture bug and confirmed
several non-issues:

- **BUG (fixed):** block 1 had RF 5×5 in our impl, but the paper says all four
  CIFAR conv blocks use 3×3 RF with padding=1. The 5×5 came from the agent's
  initial PDF read confusing the MNIST first-block (5×5) with the CIFAR
  first-block (3×3). Fixed in [src/modules/cdlgn/tree_net.py](src/modules/cdlgn/tree_net.py).
  Re-running S now.
- **Confirmed correct:** dense head 128k→1280k→640k→320k (the `*` footnote in
  §A.1.1 says 2× gates only for B/L sizes); τ values (S=20, M=40); LR=0.02;
  weight decay=0.002; thermometer encoding (n_bits=3 = paper's "2 input bits");
  4 OR-pools with stride 2.
- **Memory:** M run at bs=128 OOMed at 47 GB; the pure-PyTorch ConvLogicLayer
  is far less memory-efficient than the paper's fused CUDA kernel. Switched to
  bs=32 + grad-accumulation×4 for iso-effective-batch.
- **Channel grouping:** paper §A.3 uses k/8 groups; we still default to 1.
  Expected ≤2 pp effect; revisit only if M still misses after the block-1 fix.

## Open questions
- Encoding spec double-confirm (proposal §Phase 1 Task 1): plan defaults to
  thermometer thresholds `(i+1)/(n_bits+1)` per the difflogic reference and paper
  §A.1.1; n_bits=3 for S/M, 31 for B/L/G. Will proceed unless told otherwise.
- Channel grouping (paper §A.3): default `channel_groups=1`. The faithful
  paper repro uses `k/8`; defer until S sanity lands and we know whether the
  un-grouped baseline closes the 2% gap.
