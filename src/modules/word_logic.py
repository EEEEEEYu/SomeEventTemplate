"""Word-level logic layer (proposal v2 §Stage 3, pulled forward in Phase P §P3).

`WordLogicLayer` extends difflogic's `LogicLayer` to operate on `[B, in_dim, M]`
tensors instead of `[B, in_dim]`. The 16-op vocabulary, weight relaxation, and
discretization are the same; the only difference is that the bitwise op is
applied across an extra `M`-bit "word" axis.

Two design constraints:
  1. **`(N, M)` independence.** This file owns `M` (bits per word). `N` (slice
     count for the streaming buffer) lives in Stage 4's buffer module — it is
     none of this layer's business. (Proposal v2 §Stage 3 task 1; v2.1 line 396.)
  2. **`M=1` bit-for-bit parity with `difflogic.LogicLayer`.** With identical
     seeds, an `M=1` WordLogicLayer and a `LogicLayer` initialised the same
     way must produce bit-identical outputs in `eval()` mode. This is the
     only scalar parity check still meaningful in v2 (proposal v2 §Stage 3
     task 2; lines 219, 233).

To get parity-by-construction we mirror `difflogic.LogicLayer.__init__`'s RNG
calls in the exact same order:
    1. `torch.randn(out_dim, 16, device=device)` for `self.weights`
    2. `torch.randperm(2 * out_dim) % in_dim`
    3. `torch.randperm(in_dim)`
A different RNG path here would produce a different (but equally valid) network
and the parity test would fail on connectivity differences, not bugs. The test
docstring records the procedure.

Inference fast path: at `M=32` with bit-packed inputs we could call
`difflogic_cuda.eval` directly. For now we only implement the slow training/eval
path (per-bit float ops); the fast path is part of the existing difflogic flow
when `M=32` and is exercised by Stage 3's throughput task. For `M ∈ {64, 128}`
the fast path requires Phase P §P4a's kernel extension.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflogic.functional import bin_op_s, get_unique_connections, GradFactor


class WordLogicLayer(nn.Module):
    """Differentiable logic-gate layer over `[B, in_dim, M]` word tensors.

    At `M=1` reproduces `difflogic.LogicLayer` bit-for-bit when initialised
    under the same RNG state.

    Args:
        in_dim:      input feature count.
        out_dim:     output feature count (= number of logic neurons).
        M:           bits per word (>= 1). Default 32 matches difflogic's int32
                     packing path.
        device:      "cuda" or "cpu".
        grad_factor: per-layer gradient scaling (proposal §Stage 1 task 2).
        connections: "random" or "unique" — same vocabulary as
                     `difflogic.LogicLayer`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        M: int = 32,
        device: str = "cuda",
        grad_factor: float = 1.0,
        connections: str = "random",
    ) -> None:
        super().__init__()
        if M < 1:
            raise ValueError(f"M must be >= 1; got {M}")
        if 2 * out_dim < in_dim:
            raise ValueError(
                f"out_dim ({out_dim}) must be >= in_dim/2 ({in_dim/2}) so all "
                f"inputs can be referenced — same constraint as difflogic.LogicLayer."
            )
        if connections not in ("random", "unique"):
            raise ValueError(f"connections must be 'random' or 'unique'; got {connections!r}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.M = M
        self.device = device
        self.grad_factor = grad_factor
        self.connections = connections

        # RNG order MUST match difflogic.LogicLayer.__init__ exactly so that
        # `M=1` parity holds under a shared seed. See file docstring.
        self.weights = nn.Parameter(torch.randn(out_dim, 16, device=device))
        a_idx, b_idx = self._init_connections(connections, device)
        self.register_buffer("indices_a", a_idx, persistent=False)
        self.register_buffer("indices_b", b_idx, persistent=False)

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def _init_connections(self, connections: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if connections == "random":
            # VERBATIM from difflogic.LogicLayer.get_connections (random branch).
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        return get_unique_connections(self.in_dim, self.out_dim, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward over `[B, in_dim, M]` → `[B, out_dim, M]`.

        At `M=1` and `eval()` mode this matches `difflogic.LogicLayer.forward`
        on `x.squeeze(-1)` bit-for-bit (the parity test pins this down).
        """
        if x.dim() != 3 or x.shape[1] != self.in_dim or x.shape[2] != self.M:
            raise ValueError(
                f"WordLogicLayer expects input shape [B, {self.in_dim}, {self.M}]; "
                f"got {tuple(x.shape)}"
            )

        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        a = x[:, self.indices_a, :]   # [B, out_dim, M]
        b = x[:, self.indices_b, :]   # [B, out_dim, M]

        if self.training:
            w = F.softmax(self.weights, dim=-1)                                       # [out_dim, 16]
        else:
            w = F.one_hot(self.weights.argmax(-1), 16).to(x.dtype)                    # [out_dim, 16]

        # bin_op_s does `r += i_s[..., i] * u` where u has shape [B, out_dim, M].
        # We need i_s[..., i] to broadcast against [B, out_dim, M] across the M axis,
        # so reshape w to [out_dim, 1, 16] -> i_s[..., i] is [out_dim, 1]
        # -> broadcasts to [1, out_dim, 1] -> [B, out_dim, M]. ✓
        w_bcast = w.unsqueeze(1)
        return bin_op_s(a, b, w_bcast)

    def extra_repr(self) -> str:
        return f"{self.in_dim}, {self.out_dim}, M={self.M}, {'train' if self.training else 'eval'}"
