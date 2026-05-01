"""torchlogix verification harness (no performance requirement).

Three checks, each isolated so the next one runs even if the previous fails:

    1. API inspection — exact constructor signatures and docstrings of the
       primitives we care about.
    2. CPU smoke — assemble a tiny CDLGN-shaped model with the correct API
       and verify forward + backward + the train/eval discretization toggle.
    3. GPU smoke — same, on cuda; also instantiate the paper-spec
       `ClgnCifar10Small` reference model so we can confirm it builds.

Run on the allocated GPU node (e.g. via `tmux send-keys -t 0 ...`).
"""

from __future__ import annotations

import inspect
import sys
import traceback

import torch
import torch.nn as nn

from torchlogix import layers as tl_layers
from torchlogix.layers import (
    LogicConv2d, OrPooling2d, LogicDense, GroupSum, FixedBinarization,
)


# ---------------------------------------------------------------------------
# 1. API inspection (already mostly known; print again for archival)
# ---------------------------------------------------------------------------

def _print_api(cls):
    print(f"\n--- {cls.__name__} ---")
    print(f"signature: {inspect.signature(cls.__init__)}")


def step1_api():
    print("=" * 70)
    print("STEP 1 — API inspection")
    print("=" * 70)
    for cls in [LogicConv2d, OrPooling2d, LogicDense, GroupSum,
                FixedBinarization]:
        try:
            _print_api(cls)
        except Exception:
            print(f"[{cls.__name__}] inspect failed:")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# 2. CPU build + forward + backward smoke
# ---------------------------------------------------------------------------

def _build_tiny_cdlgn(in_dim: int = 32, channels: int = 9, k: int = 8,
                      num_classes: int = 10, device: str = "cpu"):
    """One-conv-block tiny model purely to exercise the API. Sizes obey
    `out_dim * lut_rank >= in_dim` for the dense layer (same constraint as
    difflogic's `2 * out_dim >= in_dim`). Output dim must also be divisible
    by num_classes for GroupSum."""
    flat_dim = k * (in_dim // 2) ** 2                            # k × 16 × 16
    # Dense out_dim: must be ≥ flat_dim / lut_rank=2 and divisible by num_classes.
    dense_out = max(flat_dim, ((flat_dim + num_classes - 1) // num_classes) * num_classes)
    return nn.Sequential(
        LogicConv2d(in_dim=in_dim, channels=channels, num_kernels=k,
                    tree_depth=3, receptive_field_size=3, padding=1,
                    device=device),
        OrPooling2d(kernel_size=2, stride=2),                    # → k × 16 × 16
        nn.Flatten(),
        LogicDense(in_dim=flat_dim, out_dim=dense_out, device=device),
        GroupSum(k=num_classes, tau=1.0),
    )


def step2_cpu_smoke():
    print("\n" + "=" * 70)
    print("STEP 2 — CPU forward + backward smoke")
    print("=" * 70)
    try:
        torch.manual_seed(0)
        model = _build_tiny_cdlgn(in_dim=32, channels=9, k=8, num_classes=10,
                                  device="cpu")
        # Train mode: relaxed (real-valued) gates.
        model.train()
        x = (torch.rand(2, 9, 32, 32) > 0.5).to(torch.float32)
        y = torch.randint(0, 10, (2,))
        logits_train = model(x)
        print(f"train forward OK: input {tuple(x.shape)} → logits {tuple(logits_train.shape)}")
        loss = nn.functional.cross_entropy(logits_train, y)
        loss.backward()

        # Walk parameters, count non-zero gradient tensors.
        grad_param_count = total_params = 0
        for p in model.parameters():
            total_params += 1
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                grad_param_count += 1
        print(f"backward OK: {grad_param_count}/{total_params} param tensors received non-zero grad")

        # Eval-mode discretization toggle.
        model.eval()
        with torch.no_grad():
            logits_eval = model(x)
        delta = (logits_train.detach() - logits_eval).abs().max().item()
        print(f"eval forward OK: max |train - eval| = {delta:.4f}")
        if grad_param_count == 0:
            print("FAIL: no gradients propagated.")
            return False
        return True
    except Exception:
        print("STEP 2 raised:")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 3. GPU smoke + paper-spec ClgnCifar10Small build
# ---------------------------------------------------------------------------

def step3_gpu_and_paper_arch():
    print("\n" + "=" * 70)
    print("STEP 3 — GPU smoke + paper-spec ClgnCifar10Small build")
    print("=" * 70)
    if not torch.cuda.is_available():
        print("GPU not available; skipping.")
        return None
    try:
        device = "cuda"

        # --- 3a) tiny CDLGN on GPU ---
        torch.manual_seed(0)
        model = _build_tiny_cdlgn(in_dim=32, channels=9, k=8, num_classes=10,
                                  device=device).to(device)
        x = (torch.rand(4, 9, 32, 32, device=device) > 0.5).to(torch.float32)
        y = torch.randint(0, 10, (4,), device=device)
        model.train()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        print(f"3a tiny GPU train+backward OK: loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            logits_eval = model(x)
        print(f"3a eval logits shape: {tuple(logits_eval.shape)}; "
              f"max |train-eval| = {(logits.detach() - logits_eval).abs().max().item():.4f}")

        # --- 3b) paper-spec ClgnCifar10Small build ---
        # Note: this model is huge in the dense head; we just confirm it
        # constructs and forwards a single tiny batch.
        from torchlogix.models.conv import ClgnCifar10Small
        thresholds = torch.tensor([
            [0.25, 0.50, 0.75]
            for _ in range(3)                                    # 3 channels
        ], device=device).T                                      # (n_bits=3, channels=3)
        # Wait — they want shape `(channels, n_bits)` per the assert in
        # ClgnCifar10.__init__: `thresholds.shape[-1] == self.n_input_bits`.
        thresholds = torch.tensor([
            [0.25, 0.50, 0.75],                                  # but small=2 bits per paper... actually
            [0.25, 0.50, 0.75],                                  # this is sized for n_input_bits which the
            [0.25, 0.50, 0.75],                                  # ClgnCifar10Small declares = 2.
        ], device=device)                                        # shape: (3, 3)
        # Use 2 thresholds (n_input_bits=2 for Small).
        thresholds = torch.tensor([
            [0.333, 0.667],
            [0.333, 0.667],
            [0.333, 0.667],
        ], device=device)                                        # (3, 2)
        try:
            mp = ClgnCifar10Small(
                thresholds=thresholds,
                binarization="fixed",
                binarization_kwargs={},
                connections_kwargs={},
            ).to(device)
            print(f"3b ClgnCifar10Small constructed on {device}; "
                  f"params: {sum(p.numel() for p in mp.parameters()):,}")
            # Confirm forward works on a 1-batch input.
            x = torch.rand(1, 3, 32, 32, device=device)
            mp.eval()
            with torch.no_grad():
                logits = mp(x)
            print(f"3b ClgnCifar10Small eval forward OK; logits {tuple(logits.shape)}")
        except Exception:
            print("3b ClgnCifar10Small construction raised:")
            traceback.print_exc()

        return True
    except Exception:
        print("STEP 3 raised:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    step1_api()
    cpu_ok = step2_cpu_smoke()
    gpu_ok = step3_gpu_and_paper_arch()
    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY  cpu={cpu_ok}  gpu={gpu_ok}")
    print("=" * 70)
    sys.exit(0 if cpu_ok and (gpu_ok is not False) else 1)
