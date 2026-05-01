"""Convolutional logic-gate layer (CDLGN, paper §3, eqs. 2-3).

Each output channel owns a logic-gate tree of depth `tree_depth`:
    - `2 ** tree_depth` leaves drawn — once at init — from positions inside
      the receptive field `(kernel_size, kernel_size)` over the input
      channels of this output channel's group.
    - `2 ** tree_depth - 1` internal gate nodes, each parameterized by a
      softmax over the 16 two-input Boolean ops (paper Equation 1, also the
      `bin_op_s` implementation in difflogic.functional).

The tree's gate softmax weights are shared across spatial locations (this is
the convolutional weight sharing); the leaf indices and gate connectivity are
**fixed at init** as registered buffers — only the softmax logits are learned.

Eval-mode forward replaces the softmax with `argmax → one_hot`, matching
`difflogic.LogicLayer.forward_python`'s discretization (lines 105-107). On
Boolean inputs this produces Boolean outputs (the bin_op multiplications
preserve `{0,1}` since `0·anything=0` and `1·1=1`).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflogic.functional import bin_op


# Polynomial coefficients for the 16 binary ops, derived from the bin_op table
# in `difflogic.functional`. Every two-input Boolean op evaluates to a
# polynomial in {1, a, b, ab} with integer coefficients, so the weighted sum
# Σᵢ wᵢ · binᵢ(a, b) collapses to W₀ + W₁·a + W₂·b + W₃·(a·b) where
# Wⱼ = Σᵢ wᵢ · cⱼ[i]. This brings memory per tree level from ~16 intermediate
# tensors down to 4, which lets larger models fit in 48 GB.
_BIN_OP_COEFFS = torch.tensor([
    # c0  c1  c2  c3
    [ 0,  0,  0,  0],   # 0  : 0
    [ 0,  0,  0,  1],   # 1  : ab
    [ 0,  1,  0, -1],   # 2  : a - ab
    [ 0,  1,  0,  0],   # 3  : a
    [ 0,  0,  1, -1],   # 4  : b - ab
    [ 0,  0,  1,  0],   # 5  : b
    [ 0,  1,  1, -2],   # 6  : a + b - 2ab  (xor)
    [ 0,  1,  1, -1],   # 7  : a + b - ab   (or)
    [ 1, -1, -1,  1],   # 8  : 1 - (a+b-ab) (nor)
    [ 1, -1, -1,  2],   # 9  : 1 - xor      (xnor)
    [ 1,  0, -1,  0],   # 10 : 1 - b
    [ 1,  0, -1,  1],   # 11 : 1 - b + ab   (b implies a)
    [ 1, -1,  0,  0],   # 12 : 1 - a
    [ 1, -1,  0,  1],   # 13 : 1 - a + ab   (a implies b)
    [ 1,  0,  0, -1],   # 14 : 1 - ab       (nand)
    [ 1,  0,  0,  0],   # 15 : 1
], dtype=torch.float32)


class ConvLogicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        tree_depth: int = 3,
        stride: int = 1,
        padding: int = 0,
        channel_groups: int = 1,
        grad_factor: float = 1.0,
    ):
        super().__init__()
        if in_channels % channel_groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by "
                f"channel_groups ({channel_groups})."
            )
        if out_channels % channel_groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"channel_groups ({channel_groups})."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.tree_depth = tree_depth
        self.stride = stride
        self.padding = padding
        self.channel_groups = channel_groups
        self.grad_factor = grad_factor

        self.n_leaves = 2 ** tree_depth                 # leaves per tree
        self.n_gates = 2 ** tree_depth - 1              # gates per tree

        self.in_channels_per_group = in_channels // channel_groups
        self.out_channels_per_group = out_channels // channel_groups
        rf_size = self.in_channels_per_group * kernel_size * kernel_size

        if rf_size < 2:
            raise ValueError(
                f"Receptive field is too small ({rf_size} positions) to feed a tree "
                f"with {self.n_leaves} leaves. Increase kernel_size, in_channels_per_group, "
                f"or reduce tree_depth."
            )

        # Per-gate 16-op logits, shared across spatial locations.
        # Shape: (out_channels, n_gates, 16). Initialized with small Gaussian by
        # default; `residual_init_` should be called after construction to bias
        # the softmax toward the canonical-A op (op index 3).
        self.weights = nn.Parameter(torch.randn(out_channels, self.n_gates, 16))

        # Sample leaves — for each output channel, draw n_leaves indices into
        # its group's RF (size rf_size). Stored as a registered buffer so they
        # do not become learnable and travel with .to(device).
        # Sampling is "with replacement" — the paper does not require distinct
        # leaves, and forbidding it would constrain rf_size >= n_leaves which
        # is restrictive for small layers. (For typical configs rf_size > n_leaves
        # so duplicates are rare in practice.)
        leaves_local = torch.randint(0, rf_size, (out_channels, self.n_leaves))
        self.register_buffer("leaves_local", leaves_local, persistent=True)

        # Group offsets: output channel c belongs to group (c // out_per_group);
        # its leaves index into the unfolded tensor at offset
        # group * (in_per_group * K * K). Pre-compute the absolute indices.
        group_of_oc = torch.arange(out_channels) // self.out_channels_per_group
        group_offsets = group_of_oc * rf_size                        # (out_channels,)
        leaves_abs = leaves_local + group_offsets.unsqueeze(-1)      # (oc, n_leaves)
        self.register_buffer("leaves", leaves_abs, persistent=True)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, k={self.kernel_size}, "
            f"d={self.tree_depth}, stride={self.stride}, padding={self.padding}, "
            f"groups={self.channel_groups}"
        )

    def _output_hw(self, h_in: int, w_in: int) -> Tuple[int, int]:
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        return h_out, w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W); got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: layer expects {self.in_channels}, got {x.shape[1]}"
            )

        # Cast Boolean inputs to float for unfold and the bin_op multiplications;
        # the output is still in {0, 1} at eval time because each gate is
        # one-hot and bin_op preserves {0, 1} on {0, 1} inputs.
        if x.dtype == torch.bool:
            x = x.to(self.weights.dtype)
        else:
            x = x.to(self.weights.dtype)

        if self.grad_factor != 1.0:
            x = _GradFactor.apply(x, self.grad_factor)

        B, _, H, W = x.shape
        H_out, W_out = self._output_hw(H, W)
        L = H_out * W_out

        # patches: (B, in_channels * K * K, L)
        patches = F.unfold(
            x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride
        )

        # Gather leaves: for each output channel select n_leaves positions from
        # its group's RF slice. We index along axis=1 with `leaves` (flat into
        # in_channels*K*K).
        # leaves: (oc, n_leaves) → flatten to (oc*n_leaves,)
        flat_idx = self.leaves.reshape(-1)
        # (B, oc*n_leaves, L)
        gathered = patches.index_select(1, flat_idx)
        gathered = gathered.view(B, self.out_channels, self.n_leaves, L)

        # Per-gate 16-op weights. Train mode: softmax. Eval mode: one-hot argmax.
        if self.training:
            op_w = F.softmax(self.weights, dim=-1)                  # (oc, n_gates, 16)
        else:
            op_w = F.one_hot(self.weights.argmax(-1), 16).to(self.weights.dtype)

        # Tree forward: at each level pair adjacent values and apply the 16-op
        # softmax. Gates are consumed in tree-order: the first 2^(d-1) gates are
        # the leaf-pair gates, the next 2^(d-2) gates are the next level, etc.
        vals = gathered                                             # (B, oc, 2^d, L)
        gate_offset = 0
        for level in range(self.tree_depth):
            n_pairs = vals.shape[2] // 2
            a = vals[:, :, 0::2, :]                                 # (B, oc, n_pairs, L)
            b = vals[:, :, 1::2, :]
            w_level = op_w[:, gate_offset : gate_offset + n_pairs, :]   # (oc, n_pairs, 16)
            # Broadcast w_level against (B, oc, n_pairs, L): need (1, oc, n_pairs, 1, 16)
            w_b = w_level.view(1, self.out_channels, n_pairs, 1, 16)
            vals = _apply_bin_op_s(a, b, w_b)                       # (B, oc, n_pairs, L)
            gate_offset += n_pairs

        assert gate_offset == self.n_gates, (gate_offset, self.n_gates)
        # vals: (B, oc, 1, L) → (B, oc, H_out, W_out)
        out = vals.squeeze(2).view(B, self.out_channels, H_out, W_out)
        return out


def _apply_bin_op_s(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted sum over the 16 binary ops, computed as a polynomial in a, b.

    Σᵢ wᵢ · binᵢ(a,b) = W₀ + W₁·a + W₂·b + W₃·(a·b)
    with Wⱼ = Σᵢ wᵢ · cⱼ[i] read from `_BIN_OP_COEFFS`.

    Args:
        a, b: tensors of shape `(B, oc, n_pairs, L)`.
        w:    weights of shape `(1, oc, n_pairs, 1, 16)` summing to 1 over dim -1.

    Returns:
        `(B, oc, n_pairs, L)`.
    """
    coeffs = _BIN_OP_COEFFS.to(device=w.device, dtype=w.dtype)        # (16, 4)
    # w · coeffs → (1, oc, n_pairs, 1, 4); split into the four W's.
    W = (w @ coeffs)                                                  # (..., 4)
    W0 = W[..., 0]                                                    # (1, oc, n_pairs, 1)
    W1 = W[..., 1]
    W2 = W[..., 2]
    W3 = W[..., 3]
    ab = a * b
    return W0 + W1 * a + W2 * b + W3 * ab


class _GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None
