"""Residual initialization for differentiable logic gate networks.

Petersen et al. 2024 §3.2 / Appendix A.5: the per-gate softmax over 16 ops
should be biased at init toward op index 3 — the canonical "A" passthrough
(`f(a, b) = a`, see the table at the top of `difflogic.functional`). This
keeps deep stacks trainable: an unrolled gate behaves like an identity
forward path before training shapes it.

Default `z3 = 5.0` reproduces the paper's CIFAR-10 setting:
    softmax(z)[3] = exp(5) / (exp(5) + 15 * exp(0)) ≈ 0.908
The remaining 15 ops share ~0.092 / 15 ≈ 0.0061 each.

Sensitivity: paper Figure 11 shows accuracy is roughly flat for `z3 ∈ [2, 6]`
and drops outside that range.
"""

from __future__ import annotations

import torch.nn as nn

from src.modules.cdlgn.conv_logic import ConvLogicLayer

# Index of the "A" passthrough op in difflogic's enumeration. Verified against
# the table at the top of difflogic.functional: row id=3 has output column
# "0 0 1 1" matching f(A, B)=A regardless of B.
CANONICAL_A_OP_INDEX = 3


def residual_init_(module: nn.Module, z3: float = 5.0) -> None:
    """In-place: bias the per-gate softmax toward the canonical-A op.

    Sets the logits at `CANONICAL_A_OP_INDEX` to `z3` and the others to 0.
    Applies to every `ConvLogicLayer` and `difflogic.LogicLayer` reachable from
    `module`.
    """
    # Local import to avoid circular: tree_net imports difflogic.LogicLayer
    # via difflogic; init.py is imported by tree_net.
    from difflogic import LogicLayer as DiffLogicLayer

    for sub in module.modules():
        if isinstance(sub, ConvLogicLayer):
            with _no_grad():
                sub.weights.zero_()
                sub.weights[..., CANONICAL_A_OP_INDEX] = z3
        elif isinstance(sub, DiffLogicLayer):
            with _no_grad():
                sub.weights.zero_()
                sub.weights[..., CANONICAL_A_OP_INDEX] = z3


class _no_grad:
    """Tiny shim — `torch.no_grad()` as a context manager. Imported lazily
    so this module doesn't drag torch in at import time when used purely as a
    type-hint target."""

    def __enter__(self):
        import torch
        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)
