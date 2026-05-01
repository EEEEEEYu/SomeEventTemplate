# torchlogix verification + refactor proposal

**Status (2026-04-30):** torchlogix 0.1.1 installed and verified. CPU + GPU
smoke tests pass. Paper-spec `ClgnCifar10Small` constructs and runs on the
allocated GPU node. **No old code removed yet** — refactor scope below
awaiting decision.

Verification harness: [scripts/torchlogix_verify.py](../scripts/torchlogix_verify.py).
Run via `python scripts/torchlogix_verify.py` on a GPU node.

## What torchlogix provides

`pip install torchlogix==0.1.1` (Python ≥ 3.10, PyTorch ≥ 1.6, MIT,
https://github.com/ligerlac/torchlogix).

| Component                          | Our equivalent                                    | Status |
|------------------------------------|---------------------------------------------------|--------|
| `torchlogix.layers.LogicConv2d`    | `src/modules/cdlgn/conv_logic.py` ConvLogicLayer  | replaceable |
| `torchlogix.layers.OrPooling2d`    | `src/modules/cdlgn/or_pool.py` OrPool2d           | replaceable |
| `torchlogix.layers.LogicDense`     | `difflogic.LogicLayer` (vendored)                 | replaceable |
| `torchlogix.layers.GroupSum`       | `difflogic.GroupSum`                              | replaceable |
| `torchlogix.layers.FixedBinarization` | `src/data/thermometer.py` + transform           | replaceable |
| `connections_kwargs.channel_group_size` | `GroupedLogicLayer` + `channel_groups` arg   | replaceable, **interpretation differs** |
| `torchlogix.models.conv.ClgnCifar10{Small,Medium,Large}` | `LogicTreeNet` | replaceable (drop-in) |
| Multiple `parametrization` kinds (`raw`, `warp`, `light`) | `_apply_bin_op_s` bilinear-form | torchlogix wins (more options) |
| `CompiledLogicNet`, `PackBitsTensor` | (we don't have) | bonus |
| `LearnableBinarization`, `SoftBinarization` | (we don't have) | bonus |

## Key API differences

- `LogicConv2d(in_dim, channels, num_kernels, tree_depth, receptive_field_size, padding, ...)`. **`in_dim` is the spatial size H/W (or `(H, W)` tuple), not in-channels.** Our `ConvLogicLayer` took `in_channels, out_channels, kernel_size`.
- `OrPooling2d(kernel_size, stride, padding=0)` — interface matches ours exactly.
- `LogicDense(in_dim, out_dim, ...)` — interface matches `difflogic.LogicLayer`.
- **Channel grouping configured via `connections_kwargs={"channel_group_size": N}`**, where `N` is **channels per group** (not number of groups). Their CIFAR-10 reference uses `group_size=2`. Our previous interpretation set `channel_groups=k/8` which means 32 groups of 8 channels each for M; torchlogix's literal reading of "k/8 separated models" gives 128 groups of 2 channels each for M. **More restrictive.**

## Verification results

```
STEP 1 — API inspection                                      OK
STEP 2 — CPU forward+backward+train/eval discretization       OK
   tiny CDLGN (k=8, 1 conv block + dense + GroupSum):
     forward (2, 9, 32, 32) → (2, 10), all 4 params got grad
     |train - eval| max delta = 30.9 (random weights, expected)
STEP 3 — GPU + paper-spec ClgnCifar10Small build              OK
   tiny CDLGN GPU: loss=5.40, eval works
   ClgnCifar10Small (k_num=32, group_size=2): 1.34M params
     forward (1, 3, 32, 32) → (1, 10)
```

## Risks worth probing before full refactor

1. **DDP compatibility.** Our current code has a `_ensure_difflogic_tensors_on_device` hook in `CDLGNClassifier.on_fit_start` that walks vendored difflogic LogicLayers and moves their unregistered `indices` / `given_x_indices_*` attrs to the rank's GPU. torchlogix has its own `connections.py` and likely stores indices differently — we need to verify a 4-GPU DDP run doesn't hit `illegal memory access`.
2. **Memory profile.** Our pure-PyTorch ConvLogicLayer needed the bilinear-form `_apply_bin_op_s` rewrite (16 → 4 op-tensors per tree level) to fit M at bs=32 on 48 GB. torchlogix's LogicConv2d uses an `einsum`-based forward — memory profile unknown. Need to confirm it fits at our M config without bs reductions.
3. **Reproducibility of our 65.99% baseline.** Their `ClgnCifar10Medium` (k=256, group_size=2, τ=40) is the closest analog. To declare equivalence we'd want a matched run — not for accuracy parity (the user said no perf requirement) but to confirm convergence works end-to-end with our LightningDataModule.

## Proposed refactor plan (Phase 1.5 — pre-Phase-2)

Two-stage to keep risk bounded:

### Stage A — adapter only, both implementations live (1-2 hr)

Create a **new** model class `src/models/torchlogix_cdlgn_classifier.py`
that uses torchlogix's `ClgnCifar10*` (or our own `nn.Sequential` of
torchlogix primitives if we want finer control). Register it in `train.py`
under a new key, e.g. `"torchlogix_cdlgn_classifier"`. Add a config
`configs/exp/04_torchlogix_cifar10_M.yaml` mirroring the existing M recipe
but pointing at the new model. **Keep all current code untouched.** Run on
4-GPU DDP; verify no crash, training converges, eval discretization works.

### Stage B — remove old code (only after Stage A is green)

If Stage A succeeds (functional, converges, fits in memory under DDP):

- Remove `src/modules/cdlgn/{conv_logic,or_pool,grouped_logic,init,tree_net}.py`.
- Remove `src/models/cdlgn_classifier.py` (replaced by torchlogix variant).
- Remove `_ensure_difflogic_tensors_on_device` (assuming torchlogix needs no equivalent).
- Remove `tests/test_cdlgn_modules.py` (19 tests covering the homegrown primitives).
- Keep `src/data/{cifar10_dm.py,thermometer.py}` either as-is (our thermometer feeds torchlogix nicely) or replace the transform with `FixedBinarization` (style choice).
- Keep `difflogic/` vendored for now — it's a torchlogix dependency? **Check.** Actually torchlogix builds its own conv/dense without difflogic; if no torchlogix code path imports `difflogic.*`, we can remove the vendored repo and its CUDA-build patch in `patches/`.
- Update `STATUS.md` and `docs/cdlgn_paper.md` to point at torchlogix.

### Don't yet (deferred)

- Don't try to "match" 65.99% with torchlogix — no perf requirement.
- Don't adopt `LearnableBinarization` / non-`raw` parametrizations — interesting but Phase-4 territory.
- Don't remove vendored `difflogic/` until we confirm nothing in torchlogix transitively imports from it.

## Recommendation

Proceed with **Stage A only** for now: write the adapter, run a smoke (S size, 1 epoch, 4 GPU DDP), confirm it works. Stage B is a separate decision after Stage A is green.
