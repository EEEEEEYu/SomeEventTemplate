# CDLGN — Convolutional Differentiable Logic Gate Networks (Petersen et al., NeurIPS 2024)

> Self-contained reference for the CDLGN paper (`ConvLGN.pdf` at the repo root).
> Purpose: future agents working on this repo should not need to re-open the PDF.
> Section/page pointers refer to the published NeurIPS 2024 version.

---

## TL;DR

CDLGN extends differentiable logic gate networks (Petersen et al. 2022, "difflogic")
from random fully-connected layers to **convolutional layers with fixed local
connectivity**, augmented with **OR-pooling** instead of max-pool and a
**residual initialization** trick that makes deep stacks trainable. On CIFAR-10
the largest model (LogicTreeNet-G, 61.0M gates) reaches **86.29%** test acc,
matching XNOR-Net (86.28%, 1.78B gates) at **~29× fewer gates** and FPGA latency
of nanoseconds rather than microseconds.

The architectural ingredients we must reproduce:
1. Convolutional logic-gate layers using **logic-gate trees** of depth `d=3` as kernels.
2. **Local, fixed-at-init** input connectivity within each receptive field.
3. **OR-pooling** (logical disjunction over a window) replacing max-pool.
4. **Residual initialization** of the per-gate softmax over the 16 binary ops.
5. **Thermometer encoding** at the input.
6. **Discretized inference** (argmax over the 16 ops at eval).

---

## 1. Architecture

### Logic-gate tree as the kernel

Each kernel is a **binary tree of depth d** with logic gates at the internal nodes
and binary signals at the leaves (Section 3, Page 3–4). For d=3 there are
`2^d − 1 = 7` gates per kernel; the tree's single output is the kernel's value
at one spatial location.

Each gate node performs one of the **16 two-input Boolean ops** (AND, OR, NOT-A,
NOT-B, XOR, NAND, …, the constants 0 and 1). During training the choice is a
**softmax distribution** over the 16 ops; during inference the softmax is
replaced by `argmax → one-hot` (Equation 1, Page 3, and `difflogic.functional.bin_op_s`).

### Convolutional structure & connectivity

For each output channel of a conv-logic layer the paper instantiates one
tree-kernel; the kernel is then **applied at every spatial location** (standard
2D convolution mechanics: stride, padding) — Page 3–4, Eq. 2–3.

The **two inputs of each gate node are randomly sampled, once at initialization,
from within the receptive field** (kernel_size × kernel_size × in_channels).
After init, the connectivity is **frozen**: only the per-gate softmax weights
are learned. Quote (Page 4): *"the connections remain fixed and the learning
task comprises the choice of logic gate at each node."*

### Channel grouping (Appendix §A.3) — verified 2026-04-30

Quoting §A.3 directly:
> "we select connections such that the model could be split into k/8 separated
>  models that are only recombined at the stage of output gates after
>  accumulation, akin to using grouped convolutions with a constant number of
>  groups throughout the network."

Two key points:
1. **k/8 groups.** For S (k=32) → 4 groups; for M (k=256) → 32 groups; for B
   (k=512) → 64 groups; for L (k=1024) → 128 groups; for G (k=2560) → 320 groups.
2. **Throughout the network**, including the **dense logic layers**. The
   recombination happens *only* at the final GroupSum. So the dense head
   should preserve the k/8-group split as well, not just the conv blocks.

In our implementation, only `ConvLogicLayer` honors `channel_groups`; the three
`difflogic.LogicLayer` dense layers in `LogicTreeNet` still use unrestricted
random connectivity. Adding grouped dense layers would require either
subclassing `LogicLayer` or instantiating one `LogicLayer` per group and
concatenating outputs. Tracked as a follow-up.

### Layer geometry (CIFAR-10, Appendix A.1.1)

**Verified against ConvLGN.pdf §A.1.1 directly (2026-04-30).** All four CIFAR
conv blocks use 3×3 RF with padding=1; block 1 is **not** a 5×5 first
layer (the agent-extracted summary was wrong about this — fixed in our code).

For width parameter `k`:

```
Input        : 32×32×3, thermometer-encoded → 32×32×(3·n_bits)
Conv-block 1 : k channels, RF 3×3, d=3, padding=1   →  32×32×k
OR-Pool 1    : 2×2, stride 2                         →  16×16×k
Conv-block 2 : 4k channels, RF 3×3, d=3, padding=1   →  16×16×4k
OR-Pool 2    : 2×2, stride 2                         →  8×8×4k
Conv-block 3 : 16k channels, RF 3×3, d=3, padding=1  →  8×8×16k
OR-Pool 3    : 2×2, stride 2                         →  4×4×16k
Conv-block 4 : 32k channels, RF 3×3, d=3, padding=1  →  4×4×32k
OR-Pool 4    : 2×2, stride 2                         →  2×2×32k
Flatten      :                                       →  128k
Dense Logic 1: 128k  → 1280k                         (× ox for B/L)
Dense Logic 2: 1280k → 640k                          (× ox for B/L)
Dense Logic 3: 640k  → 320k                          (× ox for B/L)
GroupSum     : 320k → 10 (k_groups=10, temperature τ)
```

`n_bits = 3` for S/M (paper "# input bits = 2" → 2-bit precision = 3
thermometer thresholds), `n_bits = 31` for B/L/G ("# input bits = 5").

`ox` (output gate factor, Table 6): 1 for S/M/G, 2 for B/L. The `(*)` footnote
in §A.1.1 says "for the B & L size CIFAR models, we use 2× as many gates in
the final layers" — i.e., the dense head's output dims are doubled.

For MNIST the architecture has only 3 conv-blocks (28×28 input) and the
**first** MNIST conv block uses 5×5 RF without padding (this is where the
5×5 came from in the agent summary's first read of the paper).

### Channel grouping (Appendix A.3) — gotcha

For hardware-routability the paper restricts cross-channel connectivity:
channels are split into **k/8 groups** and gate inputs are sampled only from
within their group (`in_channels_per_group` instead of full `in_channels`).
This is **not in the main text**; reproducing without it gives slightly inflated
gate counts and accuracy and breaks comparison.

---

## 2. OR-pooling

Replaces max-pooling. Pool window is **2×2 with stride 2** for CIFAR-10
(Section 3.1, Page 4–5).

- **Definition (Boolean):** logical OR over the window — `a ∨ b ∨ c ∨ d`.
- **Continuous relaxation (training):** maximum t-norm — `max(a, b, c, d)`,
  i.e. ordinary `F.max_pool2d` on `[0, 1]`-valued activations. (For Boolean
  inputs `max == OR`, so the same kernel call works in both modes.)
- **Gradient handling:** standard max-pool gradient (only the argmax position
  receives gradient).

Empirical observation (Figure 4, Page 5): post-conv activations sit around 50%
density during training; OR-pooling pushes activations higher (any 1 in the
window → 1). No explicit sparsity regularizer is needed; the density behaviour
self-regulates.

---

## 3. Residual initialization

### What it does

For each gate's 16-way softmax over ops, the paper biases initialization toward
op index that corresponds to **"identity on input A"** (i.e. the gate output
just passes input A through). With this bias, an untrained network behaves like
a (random-permuted) feedforward signal path, so gradients can flow through deep
stacks before training shapes the gates.

### Why plain init failed (Section 3.2, Page 5)

With Gaussian init of the per-op logits `z`, the softmax is roughly uniform —
each of the 16 ops has ~6.25% mass. The expected output of a gate becomes
roughly 0.5 (mean of the 16 op outputs over random binary inputs), and applied
recursively this drives all activations toward 0.5 layer by layer, shrinking
gradients exponentially in depth.

### The fix

Set logit `z_3 = 5.0` (the index of the "A" passthrough op in the paper's
enumeration) and the other 15 logits to 0. After softmax this gives
`P(op_3) ≈ exp(5)/(exp(5)+15) ≈ 0.908` and `P(other_i) ≈ 0.0061`.

### Sensitivity (Appendix A.5, Figure 11)

- Recommended range: `2 ≤ z_3 ≤ 6`.
- `z_3 < 2`: training is unstable.
- `z_3 > 6`: performance plateaus.
- Optimal `z_3` differs slightly between MNIST and CIFAR-10.

### Ablation (Table 5)

Removing residual init drops CIFAR-10 LogicTreeNet-L accuracy from 84.99% →
**76.18% (−8.81 pp)**. This is the single most important non-obvious
implementation detail.

> ⚠️ **Op-index gotcha.** The paper says "A" but does not state which softmax
> index corresponds to the canonical-A op — verify against `difflogic.functional`
> (specifically `bin_op_s`, which enumerates the 16 ops in a fixed order) before
> hard-coding `z[3]`. The right thing to do is search the op enumeration for
> the function `(a, b) ↦ a` and set the logit at that index.

---

## 4. Input encoding (thermometer)

### CIFAR-10 (Appendix A.1.1)

Thermometer encoding turns each pixel value `x ∈ [0, 1]` into a fixed-length
binary vector by comparison with a set of evenly-spaced thresholds:

```
bit_i = 1 iff x > thresholds[i]
```

| Model size | n_bits per channel | thresholds                     | total input dims         |
|------------|--------------------|--------------------------------|--------------------------|
| S, M       | 3                  | (i+1)/4 for i in {0,1,2}       | 3 × 3 × 32 × 32 = 9,216  |
| B, L, G    | 31                 | (i+1)/32 for i in {0,…,30}     | 31 × 3 × 32 × 32 = 94,272|

### MNIST (Appendix A.1.2)

Single channel, 28×28. S/M use 3-bit, L uses larger. Total: e.g. 2,352 for S.

### Encoding gates are counted

The published gate counts **include the thermometer comparison gates**
(Page 6). When reporting our own gate counts we must do the same to compare
fairly.

> ⚠️ No data augmentation is described in the paper — only `ToTensor()` and
> thermometer encoding. Reproducers who add CIFAR-10 standard augmentation
> (random crop + horizontal flip) will get different (typically higher) numbers
> and break the comparison.

---

## 5. Training recipe (Appendix A.2, Table 6)

| Aspect            | CIFAR-10                                 | MNIST                |
|-------------------|------------------------------------------|----------------------|
| Optimizer         | AdamW                                    | AdamW                |
| Learning rate     | 0.02 (all sizes)                         | 0.01                 |
| Weight decay      | 0.002                                    | 0.002                |
| Batch size        | 128                                      | 512                  |
| Steps             | 2,000 (eval every 200)                   | 50,000 (eval every 5,000) |
| Train/val split   | 45,000 / 5,000 (from official train)     | 50,000 / 10,000      |
| Loss              | Cross-entropy on `softmax(logits / τ)`   | same                 |
| Batchnorm         | **None**                                 | **None**             |
| Dropout           | **None**                                 | **None**             |
| Augmentation      | **None reported**                        | **None reported**    |

### Softmax temperature τ

τ is applied inside the loss to the GroupSum logits. The paper varies τ with
model size; rule of thumb stated qualitatively as **τ ∝ √(n_output_gates)**.

| Model | τ (CIFAR-10) | τ (MNIST) |
|-------|--------------|-----------|
| S     | ~20          | ~6.5      |
| M     | ~40          | ~10       |
| B     | ~280         | —         |
| L     | ~350         | ~35       |
| G     | ~450         | —         |

> ⚠️ τ is critical. With τ too small, the logits saturate the softmax and
> gradients die; too large, the softmax is too flat and discretized accuracy
> drops. There is no closed-form; expect to grid-search τ when the model size
> changes.

---

## 6. Discretization at inference

- **Train forward (soft):** for each gate, output `Σ_i softmax(z)[i] · op_i(a, b)`,
  a scalar in `[0, 1]`.
- **Eval forward (hard):** replace softmax with one-hot at `argmax(z)`. Output
  is a deterministic Boolean. Code path: `difflogic/difflogic.py` lines around
  106 and 128 — `F.one_hot(weights.argmax(-1), 16)`.

The train↔discretized accuracy gap is small (<1%) once training has converged
and τ is right (Figure 10, Appendix A.4). Early in training the gap can be
3–5%. **For Phase 1 reporting we always use the discretized number** (proposal
v3 §Cross-cutting rules: "Discretization is what matters").

---

## 7. Reported results

### CIFAR-10 (Table 1, Page 8)

| Method                | Test acc | Gate count |
|-----------------------|----------|------------|
| DiffLogic (Petersen 2022, small)  | 57.39%   | 0.51 M    |
| DiffLogic (largest)               | 62.14%   | 5.12 M    |
| **LogicTreeNet-S** (k=32)         | **60.38%** | **0.40 M** |
| **LogicTreeNet-M** (k=256)        | **71.01%** | **3.08 M** |
| **LogicTreeNet-B** (k=512)        | **80.17%** | **16.0 M** |
| **LogicTreeNet-L** (k=1024)       | **84.99%** | **28.9 M** |
| **LogicTreeNet-G** (k=2560)       | **86.29%** | **61.0 M** |
| Conv. TTNet (small)               | 50.10%   | 0.57 M    |
| Conv. TTNet (large)               | 70.75%   | 189 M     |
| FINN CNN                          | 80.10%   | 901 M     |
| XNOR-Net                          | 86.28%   | 1,780 M   |

### CIFAR-10 FPGA latency (Table 2)

LogicTreeNet-S/M: 9 ns. LogicTreeNet-B: 24 ns. FINN CNN: 45.6 µs (≈ 1900× slower
for the same accuracy class). These are Xilinx Vivado HLS synthesis estimates,
not measured silicon.

### MNIST (Table 3)

| Method               | Test acc | Gate count |
|----------------------|----------|------------|
| DiffLogic (largest)  | 98.47%   | 384 K      |
| LogicTreeNet-S       | 98.46%   | 147 K      |
| LogicTreeNet-M       | 99.23%   | 566 K      |
| LogicTreeNet-L       | 99.35%   | 1.27 M     |

### Ablations on CIFAR-10 LogicTreeNet-L (Table 5)

| Ablation                  | Acc Δ (vs 84.99%) |
|---------------------------|-------------------|
| Tree depth d=1 everywhere | **−4.01 pp**      |
| Tree depth (1,1,2,2)      | −2.31 pp          |
| Tree depth (2,2,2,2)      | −1.67 pp          |
| Tree depth (2,3,3,3)      | −0.86 pp          |
| **No OR-pool**            | **−3.54 pp**      |
| **Gaussian init**         | **−8.81 pp**      |
| No weight decay           | −1.05 pp          |
| 8 input channels (instead of 2-bit precision) | −1.46 pp |

Take-away: tree depth 3, OR-pool, and residual init are each load-bearing.

---

## 8. Map to the existing repo

### Reusable as-is from `difflogic/`

| Component               | What we use it for                                |
|-------------------------|---------------------------------------------------|
| `LogicLayer`            | The 3 dense logic layers after conv blocks (MLP head). |
| `GroupSum`              | Final readout into 10 class scores with τ.        |
| `bin_op_s`              | The 16-op softmax — same math we need inside ConvLogicLayer. |
| Discretization (`argmax → one_hot`) | Same eval-mode behaviour for our conv layer. |
| `PackBitsTensor`, `CompiledLogicNet` | Optional later — discrete-only inference & FPGA-style packing. |

### New code needed

| Module                          | Why                                              |
|---------------------------------|--------------------------------------------------|
| `src/modules/cdlgn/conv_logic.py` (`ConvLogicLayer`) | Local-RF, fixed-at-init connectivity. `difflogic.LogicLayer` is global FC. |
| `src/modules/cdlgn/or_pool.py` (`OrPool2d`) | `F.max_pool2d` wrapped with right semantics for both train & eval. |
| `src/modules/cdlgn/init.py` (`residual_init_`) | Set logits so canonical-A op has ~0.9 softmax mass. |
| `src/modules/cdlgn/tree_net.py` (`LogicTreeNet`) | Container assembling the layer-by-layer architecture above. |
| `src/data/thermometer.py` (`thermometer_encode`) | Reusable encoder for CIFAR-10 (and later flow inputs). |
| `src/data/cifar10_dm.py` (`CIFAR10DataModule`) | Lightning data plumbing. |
| `src/models/cdlgn_classifier.py` | Lightning wrapper around backbone + CE loss. |

### Partial / minor code

The existing `difflogic/main.py` already has a thermometer-encoding code path
for CIFAR-10 (look for `cifar-10-31-thresholds`); it can be cribbed into
`src/data/thermometer.py` rather than rewritten.

---

## 9. Reproduction risks & gotchas

1. **Op-index for "A".** The paper writes `z_3 = 5` but does not pin down which
   index in `difflogic`'s 16-op enumeration is the A-passthrough. Verify with
   a small unit test before relying on it (apply the gate to known `(a, b)` and
   check the output equals `a`).
2. **Channel grouping.** In Appendix A.3 only. Reproducing without it inflates
   accuracy. Make `channel_groups` a config arg.
3. **τ scaling.** Rule is √n_output_gates but constant of proportionality is
   per-experiment. Expect to tune.
4. **No augmentation.** Adding standard CIFAR-10 augmentation breaks
   comparability. Keep augmentation off in our reproduction.
5. **No batchnorm.** Logic networks rely on residual init + OR-pool for
   stability. Don't add BN "to help convergence" — it changes the architecture.
6. **OR-pool gradient.** Paper says max-t-norm relaxation; this is exactly
   `F.max_pool2d` on real-valued inputs and produces standard max-pool gradients.
   No custom autograd needed.
7. **Discretization gap.** Always report discretized eval acc, not soft. Per
   proposal §Cross-cutting rules.
8. **Fixed connectivity.** Easy to accidentally make `idx_a, idx_b` learnable;
   they must be `register_buffer`s, not parameters.
9. **Thermometer threshold convention.** `(i+1)/(n+1)` not `i/n` — check
   boundary behaviour at `x = 0` and `x = 1` against the existing difflogic
   code.
10. **Gate-count accounting.** Paper counts thermometer encoding gates in the
    total. Our gate-count utility must do the same to compare apples-to-apples.

---

## 10. Quick reference — defaults to use for Phase 1

| Knob                   | Default                                  |
|------------------------|------------------------------------------|
| Tree depth d           | 3                                        |
| Conv blocks            | 4 (CIFAR-10)                             |
| Conv RFs (block 1..4)  | 3×3, 3×3, 3×3, 3×3 (CIFAR-10)            |
| Channel widths         | k, 4k, 16k, 32k                          |
| OR-pool                | 2×2 stride 2 after each block            |
| Dense logic head       | 128k → 1280k → 640k → 320k → GroupSum-10 |
| Channel groups         | k / 8                                    |
| n_bits (S/M)           | 3                                        |
| n_bits (B/L/G)         | 31                                       |
| Optimizer              | AdamW                                    |
| LR                     | 0.02                                     |
| Weight decay           | 0.002                                    |
| Batch size             | 128                                      |
| Steps                  | 2,000 (eval every 200)                   |
| Residual init z[A]     | 5.0                                      |
| τ (M, k=256)           | ~40 (start here; tune if needed)         |
| Augmentation           | None                                     |
| Batchnorm / dropout    | None                                     |

---

## 11. Citations within this doc

All section/figure/table/page references point to Petersen, Borgerding, Petersen,
Welleck, Ermon, Kuehne, "Convolutional Differentiable Logic Gate Networks",
NeurIPS 2024 (`ConvLGN.pdf` at the repo root).

The 2022 predecessor — Petersen, Borgerding, Kuehne, Ermon, "Deep Differentiable
Logic Gate Networks" — is the basis of `difflogic/`.
