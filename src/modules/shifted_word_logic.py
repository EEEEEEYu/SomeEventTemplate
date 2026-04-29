"""Shifted word logic layer (Phase P §P4c / Tier 0 expressivity hook).

The resurrected v1 Stage 4 contribution (proposal v1 §Stage 4 / v2.1 §Tier 0
line 106): each output bit picks two input words and a bit-rotate amount
`s`, then computes

    out_bit_j = op(word_a_j, word_b_{(j - s) mod M})

across all `j ∈ [0, M)`. Rotation provides cross-bit temporal coupling within
a single forward pass — Tier 0's whole reason to exist.

Three trainable choices per neuron:
    1. Connectivity (a, b) — fixed pseudorandom, identical to WordLogicLayer
       (RNG ordering matters: see word_logic.py for the connectivity-init audit).
    2. 16-op vocabulary — softmax over `weights ∈ [out, 16]`, same as difflogic.
    3. Shift amount — softmax over `shift_weights ∈ [out, K]` where `K` is
       either `M` (full alphabet, default) or `len(shift_lut)` (a fixed
       reduced vocabulary like `(0, 1, 2, 4, 8, 16, 32, 64, 127)`).

`shift = 0` for all neurons makes the layer functionally identical to a
`WordLogicLayer` (test pins this down — `tests/test_shifted_word_logic.py`).
This invariant requires `shift_lut[0] == 0` when an LUT is in use.

Memory: training-time forward materialises `[B, out_dim, K, M]` for the soft-
shift sum. With the full alphabet at `M=128, out_dim=4000, B=8` this is
~8 GB; with a `K=9` LUT it drops to ~600 MB — the headline reason the LUT
exists (proposal v2.1 §Tier 0 + Phase P §P4d optimisation plan).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflogic.functional import bin_op_s, get_unique_connections, GradFactor


def _build_shift_gather_indices(
    M: int,
    device: torch.device,
    shift_lut: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Returns `[K, M]` int64 tensor where `idx[k, j] = (j - shift_k) mod M`.

    With `shift_lut=None` we use `K = M` and `shift_k = k` (the full rotation
    alphabet). With a non-empty `shift_lut`, `K = len(shift_lut)` and each row
    `k` corresponds to the rotation by `shift_lut[k]`.
    """
    positions = torch.arange(M, device=device)
    if shift_lut is None:
        shifts = torch.arange(M, device=device)
    else:
        shifts = torch.tensor(list(shift_lut), dtype=torch.int64, device=device)
    return (positions.unsqueeze(0) - shifts.unsqueeze(1)) % M  # [K, M]


class ShiftedWordLogicLayer(nn.Module):
    """Word logic layer with per-neuron bit-rotation of the second operand.

    Args mirror `WordLogicLayer` plus:
        train_shift_relaxation: if True (default), shifts are softmax-mixed
            during training. False forces the layer through the discretized
            (one-hot argmax) path always — useful for ablation runs where you
            want to disable the soft path.

    At `M=1` the layer collapses to `WordLogicLayer` (the only shift is 0).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        M: int = 32,
        device: str = "cuda",
        grad_factor: float = 1.0,
        connections: str = "random",
        train_shift_relaxation: bool = True,
        shift_lut: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if M < 1:
            raise ValueError(f"M must be >= 1; got {M}")
        if 2 * out_dim < in_dim:
            raise ValueError(
                f"out_dim ({out_dim}) must be >= in_dim/2 ({in_dim/2}) — same "
                f"constraint as difflogic.LogicLayer."
            )
        if connections not in ("random", "unique"):
            raise ValueError(f"connections must be 'random' or 'unique'; got {connections!r}")
        if shift_lut is not None:
            shift_lut = tuple(int(s) for s in shift_lut)
            if len(shift_lut) == 0:
                raise ValueError("shift_lut must be non-empty when provided")
            if shift_lut[0] != 0:
                raise ValueError(
                    f"shift_lut[0] must be 0 (the shift=0 ⇔ WordLogicLayer parity "
                    f"invariant requires LUT index 0 to map to identity); got {shift_lut[0]}"
                )
            if any(not (0 <= s < M) for s in shift_lut):
                raise ValueError(f"every entry of shift_lut must be in [0, {M}); got {shift_lut}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.M = M
        self.device = device
        self.grad_factor = grad_factor
        self.connections = connections
        self.train_shift_relaxation = train_shift_relaxation
        self.shift_lut = shift_lut
        # K = M when full alphabet, len(shift_lut) when a reduced LUT is in use.
        K = M if shift_lut is None else len(shift_lut)
        self.K = K

        # RNG order matches WordLogicLayer (which matches difflogic.LogicLayer)
        # for the first two calls. The shift_weights call is new and lives at
        # the end so it doesn't perturb the connectivity parity for shared
        # weight/index inits between the two layer classes.
        self.weights = nn.Parameter(torch.randn(out_dim, 16, device=device))
        a_idx, b_idx = self._init_connections(connections, device)
        self.register_buffer("indices_a", a_idx, persistent=False)
        self.register_buffer("indices_b", b_idx, persistent=False)
        # shift_weights has K entries (= M for full alphabet, = len(shift_lut) for LUT).
        self.shift_weights = nn.Parameter(torch.randn(out_dim, K, device=device))

        # Cache the gather indices on the right device.
        gather = _build_shift_gather_indices(M, torch.device(device), shift_lut=shift_lut)
        self.register_buffer("_shift_gather_idx", gather, persistent=False)

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def _init_connections(self, connections: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if connections == "random":
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        return get_unique_connections(self.in_dim, self.out_dim, device)

    def _materialise_shifted_b(self, b: torch.Tensor) -> torch.Tensor:
        """`b` is `[B, out_dim, M]`. Return `[B, out_dim, K, M]` where the K
        axis enumerates right-rotations from the alphabet (full M-alphabet
        when `shift_lut is None`, otherwise the LUT entries).
        """
        # Advanced indexing along the last axis: b[..., gather_idx] reshapes
        # the last axis from `[M]` to `[K, M]`.
        return b[..., self._shift_gather_idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.in_dim or x.shape[2] != self.M:
            raise ValueError(
                f"ShiftedWordLogicLayer expects input shape [B, {self.in_dim}, {self.M}]; "
                f"got {tuple(x.shape)}"
            )

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        a = x[:, self.indices_a, :]              # [B, out_dim, M]
        b = x[:, self.indices_b, :]              # [B, out_dim, M]

        # Op-choice weights.
        if self.training:
            w_op = F.softmax(self.weights, dim=-1)
        else:
            w_op = F.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
        w_op_bcast = w_op.unsqueeze(1).unsqueeze(1)   # [out_dim, 1, 1, 16] — broadcasts over batch + shift + bit

        # Shift-choice weights.
        if self.training and self.train_shift_relaxation:
            w_shift = F.softmax(self.shift_weights, dim=-1)             # [out_dim, K]
        else:
            w_shift = F.one_hot(self.shift_weights.argmax(-1), self.K).to(x.dtype)  # [out_dim, K]

        # Discretized fast path: only one shift per neuron is active. Avoid
        # materialising the full [B, out, K, M] tensor by gathering b at the
        # chosen shift directly. `chosen_shift_idx` is an LUT index into
        # `_shift_gather_idx` which already encodes the actual rotation.
        if not (self.training and self.train_shift_relaxation):
            chosen_shift_idx = self.shift_weights.argmax(-1)        # [out_dim]
            per_neuron_idx = self._shift_gather_idx[chosen_shift_idx]   # [out_dim, M]
            per_neuron_idx_b = per_neuron_idx.unsqueeze(0).expand(b.shape[0], -1, -1)  # [B, out_dim, M]
            b_shifted = torch.gather(b, dim=2, index=per_neuron_idx_b)                 # [B, out_dim, M]
            return bin_op_s(a, b_shifted, w_op.unsqueeze(1))         # [B, out_dim, M]

        # Soft path: build the full shift tensor and sum-weight. `bin_op` in
        # difflogic.functional asserts shape equality (no implicit broadcasting),
        # so we expand a explicitly to match b_all.
        b_all = self._materialise_shifted_b(b)                      # [B, out_dim, K, M]
        a_b = a.unsqueeze(2).expand_as(b_all)                       # [B, out_dim, K, M]
        per_shift = bin_op_s(a_b, b_all, w_op_bcast)                # [B, out_dim, K, M]
        # Weight by shift softmax along the K axis and sum.
        # w_shift is [out_dim, K]; broadcast to [1, out_dim, K, 1].
        w_shift_bcast = w_shift.unsqueeze(0).unsqueeze(-1)          # [1, out_dim, K, 1]
        return (per_shift * w_shift_bcast).sum(dim=2)               # [B, out_dim, M]

    def force_zero_shift_(self) -> None:
        """Set all shift logits so the argmax picks shift=0. Used by the
        parity test: `ShiftedWordLogicLayer` with shift=0 everywhere should
        match `WordLogicLayer` bit-for-bit (in eval mode, with shared
        weights+indices). When a `shift_lut` is in use, the constructor
        already enforced `shift_lut[0] == 0`, so picking LUT index 0 is the
        same as picking shift=0."""
        with torch.no_grad():
            self.shift_weights.zero_()
            self.shift_weights[:, 0] = 1.0  # one-hot at LUT index 0 after argmax

    def extra_repr(self) -> str:
        return f"{self.in_dim}, {self.out_dim}, M={self.M}, {'train' if self.training else 'eval'}"
