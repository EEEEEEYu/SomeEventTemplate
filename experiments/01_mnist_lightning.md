# Stage 1 — Lightning + MNIST parity

**Status:** ✅ passed.
**Gate (proposal §Stage 1, anchored to Stage 0):** discretized test_acc ≥ **97.10%**
(Stage 0 baseline 97.40% −0.3%).

## Result

| Metric | Value |
|---|---|
| **test_acc (MNIST 10k test, discretized)** | **0.9736 (97.36%)** |
| test_loss | 0.0872 |
| val_acc (10k held-out from MNIST train, discretized) | 0.970 |
| train_acc_epoch (eval mode) | 0.997 |
| Δ vs Stage 0 baseline (97.40%) | −0.04% (well inside ±0.3%) |
| Total logic gates | 40,000 (6 layers × 8000) |
| Trainable params | 768 K |
| Wall-clock (fit + test) | ~5 min on 1×rtxa6000 |
| Train throughput | ~145–155 it/s @ bs=100 |
| Eval throughput (test) | 270 it/s @ bs=100 ≈ 27 k samples/s |

Run log: [01_mnist_lightning/train.log](01_mnist_lightning/train.log).
Manifest: `lightning_logs/<run>/version_pending/manifest.json`.

## Setup

- Config: [configs/exp/01_mnist_lightning.yaml](../configs/exp/01_mnist_lightning.yaml)
- Hyperparameters: hidden_dim=8000, num_layers=6, tau=10, lr=0.01, bs=100, max_epochs=100
- Data: MNIST 50k/10k/10k (50k train + 10k val from `random_split(seed=42)`; standard 10k MNIST test).
- Inputs: float [0, 1] from `transforms.ToTensor()`. **No** dataloader-side binarization. Eval-time inputs are `.round()`-ed inside `LogicClassifier._shared_step` to mirror Petersen's `eval()` (`difflogic/experiments/main.py:189`).
- `connections="random"` (proposal default), not `"unique"` (Petersen's default). Did not affect parity at this scale.

## Notes / surprises

1. **First attempt undertrained.** Initial 30-epoch run hit 96.7% val_acc — undertrained, *not* a wiring bug. Stage 0 only crossed 97% at iter ~7k and plateaued at iter ~36k; 30 epochs × 500 steps = 15k iters lands near 96.5–97.0% on the val split. Bumped `max_epochs` to 100 (50k iters); cleared the gate cleanly.

2. **val_acc < test_acc.** The 10k val split is sampled from the 60k MNIST train set, so it shares the train distribution exactly. The 10k MNIST test set is independently distributed. With this much regularization (LGN's discretization is its own regularizer), test marginally beats val. Don't read into it.

3. **No `on_validation_epoch_start` hook needed.** Lightning's automatic `model.eval()` flips `self.training=False` on every `LogicLayer`, which is what `LogicLayer.forward` reads to dispatch to the discretized eval branch (`difflogic/difflogic.py:103,111`). The proposal's mention of a manual hook is defensive — not required when the underlying module honors `self.training`.

4. **lr=0.01 is hardcoded as the LightningModule default**, not just in the optimizer config — the proposal flags this as a critical correctness issue (the relaxation needs the higher LR). Putting the default in code prevents a config typo from silently breaking parity.

## Stage 2 prep

Throughput / accuracy reference for the scalar-CDLGN baselines:
- Treat 97.36% as the **N-MNIST** scalar-CDLGN sanity benchmark only loosely — different dataset, different input dim. The proposal Stage 2 gate is N-MNIST ≥95%, which is a weaker bar than what we cleared here.
- Eval throughput numbers above are for fp32 on GPU. The proposal §Stage 2 task 7 wants per-dataset CPU inference throughput numbers using `CompiledLogicNet`; defer that to the Stage 2 / Stage 3 throughput characterization.
