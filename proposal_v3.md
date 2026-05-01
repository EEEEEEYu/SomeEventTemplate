# Proposal v3 — CDLGN Reproduction and Event-Camera Flow Estimation

> **Direction.** This proposal supersedes v1 (fused-word architecture for classification) and v2/v2.1 (streaming-buffer architecture). The project pivots to **optical/normal flow estimation on event cameras**, with a clean reproduction of CDLGN as the foundational step and a deliberately-deferred novelty phase. Classification remains a compatible secondary path; the codebase is structured so that adding a classification head later requires no refactor.

---

## Why this direction

Three realizations from the v2.1 review converge on this plan:

1. The strongest published LGN baseline (CDLGN, Petersen et al. 2024) has no public implementation. Every accuracy number we produce against MLP-difflogic alone carries an asterisk. We need CDLGN ourselves before we can claim anything.
2. Optical flow on event cameras is a more natural fit for an LGN substrate than gesture classification. The data is binary, the substrate is binary, and flow is fundamentally about temporal correspondence between binary events. The framing writes itself once the substrate is in place.
3. Flow is dense per-pixel prediction, which makes spatial convolutional structure essential. Pure MLP-difflogic flattens away the 2D structure we need; CDLGN preserves it. The two pivots (toward CDLGN, toward flow) are coupled.

We commit to this direction without claiming the novelty design upfront. **Phases 1–3 establish a baseline. Phase 4 explores novelty only after the baseline is real.** This ordering is non-negotiable: a novelty story without a strong baseline is a story we can't defend.

---

## Plan overview

| Phase | Goal | Gate |
|---|---|---|
| 1 | Reproduce CDLGN on CIFAR-10 | Match paper accuracy within reasonable tolerance |
| 2 | Build event-flow data pipeline + flow head | Non-trivial EPE on a small subset |
| 3 | Full CDLGN-flow baseline | Documented baseline numbers on a standard benchmark |
| 4 | BFS-style novelty exploration | Concrete improvement criteria, set in advance |

Phases 1–3 are baseline engineering. Phase 4 is the research contribution and only begins once Phase 3 has a working number to beat.

---

## Repository structure

**Keep the existing folder structure as-is.** All existing modules (`src/data/`, `src/models/`, `src/modules/`, `configs/`, `tests/`, `experiments/`) remain. New work slots in alongside what's there:

- `src/modules/cdlgn/` — convolutional logic layers, OR-pooling, residual init (Phase 1).
- `src/data/cifar10_dm.py` — for CDLGN reproduction (Phase 1).
- `src/data/event_flow_dm.py` — for flow estimation (Phase 2). The exact dataset (MVSEC / DSEC-Flow / something smaller) is decided at the start of Phase 2; the DataModule abstraction is the same regardless.
- `src/models/cdlgn_classifier.py` — Phase 1 deliverable.
- `src/models/cdlgn_flow.py` — Phase 2/3 deliverable. Designed so the backbone is shared with the classifier and only the head differs.

**Compatibility with classification.** The architecture is split as `backbone (CDLGN) → head (classifier or flow regressor)`. Switching tasks is a head swap. No code path is flow-only; no code path is classification-only. This costs us nothing now and protects optionality later.

---

## Phase 1 — Reproduce CDLGN on CIFAR-10

**Goal.** Implement convolutional differentiable logic gate networks from scratch and reproduce the CIFAR-10 numbers from Petersen et al. 2024.

**Tasks.**

1. **Confirm input encoding before implementing.** The 2024 paper uses thermometer encoding for CIFAR-10. The user (you) will double-confirm the exact encoding (bits per channel, threshold spacing) from the paper before code is written. Do not start implementation on assumed parameters.

2. **Implement convolutional logic layers.** Each output gate's two inputs are sampled from a fixed receptive field, not random across the whole input. The connectivity is fixed at initialization (matching difflogic's pattern, but spatially structured). The 16-op softmax per gate is unchanged from MLP-difflogic.

3. **Implement OR-pooling.** Replaces max-pool with disjunction over a window. One-function addition to the layer set.

4. **Implement residual initialization.** Petersen 2024 added this specifically because plain init didn't train well. Don't skip; it's load-bearing.

5. **Train on CIFAR-10.** Target the paper's reported accuracy. A reproduction within ~2% is acceptable; document the gap honestly if larger.

**Gate.**
- [ ] CIFAR-10 test accuracy within 2% of paper-reported number.
- [ ] If gap is larger, root cause documented (most likely culprits: connectivity sampling distribution within receptive field, OR-pool gradient handling, residual init scaling).

**Time.** 4–5 weeks. Budget includes ~1 week of "loss is decreasing but accuracy is stuck" debugging — this is normal for from-scratch reimplementation against a paper, not a sign of failure.

**Honesty rule.** If a paper detail is ambiguous, implement the most likely interpretation, document the choice, and flag it. Do not invent details to close the gap. A documented 4% gap is fine; a falsified-up baseline is not.

---

## Phase 2 — Event-flow pipeline and de-risking

**Goal.** Confirm the LGN substrate can produce non-trivial flow predictions before committing to a full benchmark. This is a "does it work at all" experiment, not a results phase.

**Tasks.**

1. **Pick a small flow benchmark.** Decide between MVSEC, DSEC-Flow, or a downsampled subset of either. Decide based on data-pipeline simplicity, not benchmark prestige — at this stage we want fast iteration, not SOTA conditions.

2. **Build the event-to-input pipeline.** Whatever event representation we use as input (TBR, voxel grid, event count image — open question) needs to fit the CDLGN input format. Defer the "what's the best representation" question; pick a defensible default and move on.

3. **Add a flow head.** Hybrid architecture: CDLGN backbone produces Boolean features densely per-pixel, a small float head converts these to a 2D flow vector per pixel. Trained with standard L1 or L2 loss on flow vectors. The head is small enough that the network remains "Boolean except at the readout."

4. **Run on a small subset.** Goal is to confirm EPE is meaningfully better than predict-zero. Not to match SOTA, not even to match a CNN baseline. Just: does training converge and does the output mean anything?

**Gate.**
- [ ] Training converges (loss decreases, doesn't diverge).
- [ ] EPE is meaningfully below the predict-zero baseline on the subset.
- [ ] If either fails, debug before proceeding to Phase 3. Likely failure modes: float head too small to bridge the substrate gap, event representation discards too much, gradient flow through the discrete-output backbone is too weak for regression.

**Time.** 2–3 weeks. Most of this is data-pipeline and integration work; the head itself is small.

---

## Phase 3 — Full flow baseline

**Goal.** A documented CDLGN-based flow baseline on a standard event-camera flow benchmark. This is the number that Phase 4 must beat.

**Tasks.**

1. **Scale up from Phase 2.** Full benchmark (not subset), full resolution (or a clearly-documented downsampling), proper train/val/test splits.
2. **Tune to convergence.** Standard hyperparameter search: LR, batch size, hidden dim, number of layers, head width. Document what was tried.
3. **Compare against a small CNN baseline on the same data.** Not because we expect to win, but because reviewers will ask. A simple U-Net-like CNN with comparable parameter count, trained on the same input representation, gives us a reference point.
4. **Log everything.** Per-pixel EPE, mean EPE, timing, gate count, memory footprint. These become the reference numbers Phase 4 reports against.

**Gate.**
- [ ] Documented EPE on a standard benchmark with a fixed evaluation protocol.
- [ ] Comparison to small-CNN baseline reported honestly (we may lose; document the gap).
- [ ] All numbers reproducible from a seed and config.

**Time.** 3–4 weeks.

---

## Phase 4 — BFS-style novelty exploration

> **The discipline that makes this phase work: explore broad and shallow, not narrow and deep.** Multiple candidate improvements, each tested with minimum viable implementation, no premature optimization. Only the ones that show signal get a second pass.

**Goal.** Find architectural or algorithmic improvements that meaningfully improve flow performance over the Phase 3 baseline.

**Method.**

1. **At the start of Phase 4, list 5–8 candidate improvements.** Each one is a hypothesis: "doing X should improve EPE by reason Y." Don't filter aggressively; include speculative ideas alongside obvious ones. (Examples that would be on the list: cross-bit / shift operators on temporally-binned input; richer event representations; learned vs. random connectivity; quantization-aware regularizers inspired by the Sinkhorn paper's diagnosis; output-side improvements like flow-residual prediction; positional encoding in the receptive field.)

2. **Define the bar in advance.** What counts as "leveraging performance"? Concrete thresholds: e.g., ≥X% EPE reduction at iso-gates, or ≥Y% gate reduction at iso-EPE. Write these down before running any Phase 4 experiment. **Do not move the goalposts later.**

3. **Implement each candidate at minimum viable scope.** Smallest version that tests the hypothesis. No CUDA optimization, no hyperparameter tuning beyond the obvious. Goal: a signal in or signal out.

4. **Run all candidates against the Phase 3 baseline.** Same data, same protocol, same metric. Tabulate results.

5. **Promote only candidates that pass the bar.** These get a second pass with proper tuning. Candidates that don't pass go in a "tried, didn't work" appendix — they're useful negative results, not failures to hide.

6. **Iterate.** A second BFS round may emerge from what worked in the first. Combinations of two passing candidates are themselves new candidates.

**Gate.**
- [ ] At least one candidate passes the pre-defined bar against the Phase 3 baseline.
- [ ] If no candidate passes after one full BFS round and one targeted second round, the project's contribution claim is in trouble. Honest options: weaken the claim, pivot framing, or stop.

**Time.** Open-ended; budget at least 6 weeks but be willing to extend if signal is appearing.

---

## Status tracking

**Delete the existing `STATUS.md` file.** Create a new one with this structure:

```
# STATUS

## Current phase
Phase 1 — CDLGN reproduction

## Phase progress
- [ ] Phase 1: CDLGN reproduction on CIFAR-10
- [ ] Phase 2: Event-flow pipeline de-risking
- [ ] Phase 3: Full flow baseline
- [ ] Phase 4: BFS novelty exploration

## Active work
(what's being implemented or run right now)

## Blockers
(anything stalling, with date noted)

## Recent results
(short bullet log of experiment outcomes, newest first)

## Open questions
(things that need a decision before proceeding)
```

Update STATUS.md at the start of every working session and at every gate transition. Keep it short — long status docs become stale.

---

## What's deferred (and why it's listed here)

These are explicitly out of scope for v3, but listed so we can return to them with intent rather than rediscovery:

- **Word-level / cross-bit operators (v1's contribution).** May reappear as a Phase 4 candidate; not a starting commitment.
- **Streaming buffer / temporal aggregation (v2's contribution).** Same — Phase 4 candidate, not a starting commitment.
- **Sinkhorn-Fourier substrate.** Cited as related work. Its empirical track record (76% on n=3 with gradient descent vs. 100% with exhaustive search) does not justify building on it without further evidence. The diagnostic insight (relaxations don't quantize cleanly) may inform a Phase 4 candidate (quantization-aware regularizer) without adopting the full machinery.
- **Classification benchmarks.** The architecture stays compatible. If Phase 4 produces a flow result, we can add classification numbers as a secondary table without re-architecting.

---

## Cross-cutting rules

- **Reproducibility.** Every experiment seeds RNG, logs git commit, logs full config to `experiments/<name>/manifest.json`.
- **Discretization is what matters.** Always report discretized inference accuracy/EPE, not soft training values.
- **Gates are non-negotiable.** If a phase gate fails, stop and report. Don't silently weaken thresholds.
- **One config per experiment.** Hyperparameters live in versioned configs, not CLI flags.
- **STATUS.md gets updated.** Per session, per gate transition.