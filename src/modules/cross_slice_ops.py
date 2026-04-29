"""Cross-slice operator registry + default `difflogic16` family.

Proposal v2 §Stage 4 task 2. The first decoder layer in the streaming
architecture pulls from this registry; later word layers in the decoder are
plain `WordLogicLayer`s.

A family is a self-contained `nn.Module` that:
  - declares its `arity` and `vocabulary_size` (read by callers for logging),
  - owns its own learnable parameters (op-choice softmax, slice-row choice
    softmax, etc.),
  - implements `forward(buffer)` with the same train/eval split as
    `WordLogicLayer`: softmax-relaxed during training, one-hot argmax during
    eval (giving discrete binary outputs).

Adding a new family is a one-decorator job:

    @register_family("hdc_xor_bind")
    class HDCXorBindFamily(CrossSliceOpFamily):
        ...

Configs name the family by string (`cross_slice_family: difflogic16` in YAML),
and `get_family_class(name)` looks it up. This is the seam for the HDC
contingency (proposal §Contingency) and any future operator vocabularies.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflogic.functional import bin_op_s


# ----------------------------------------------------------------------
# Registry

_FAMILY_REGISTRY: Dict[str, Type["CrossSliceOpFamily"]] = {}


def register_family(name: str) -> Callable[[Type["CrossSliceOpFamily"]], Type["CrossSliceOpFamily"]]:
    """Decorator: register a `CrossSliceOpFamily` subclass under `name`."""

    def decorator(cls: Type["CrossSliceOpFamily"]) -> Type["CrossSliceOpFamily"]:
        if name in _FAMILY_REGISTRY:
            raise ValueError(f"cross-slice family {name!r} already registered")
        _FAMILY_REGISTRY[name] = cls
        return cls

    return decorator


def get_family_class(name: str) -> Type["CrossSliceOpFamily"]:
    if name not in _FAMILY_REGISTRY:
        raise KeyError(
            f"cross-slice family {name!r} not registered. Known: {sorted(_FAMILY_REGISTRY)}"
        )
    return _FAMILY_REGISTRY[name]


def list_families() -> list[str]:
    """For diagnostics: list all registered family names."""
    return sorted(_FAMILY_REGISTRY)


# ----------------------------------------------------------------------
# Abstract base


class CrossSliceOpFamily(nn.Module):
    """Abstract base for a family of cross-slice operators.

    Subclasses implement `forward(buffer)` and declare metadata. The streaming
    classifier holds an instance of one family as its first decoder layer.

    Constructor convention: every subclass takes
    `(N: int, M: int, out_dim: int, **family_kwargs)`. `N` and `M` are the
    buffer shape; `out_dim` is the number of output neurons (= bits per output
    word, which is also `M`). Family-specific kwargs (e.g. connectivity prior)
    pass through as keyword arguments.
    """

    arity: int = 2          # default: binary ops; HDC bundling overrides this
    family_name: str = "?"  # set by subclasses for logging

    def __init__(self, N: int, M: int, out_dim: int) -> None:
        super().__init__()
        if N < 1:
            raise ValueError(f"N must be >= 1; got {N}")
        if M < 1:
            raise ValueError(f"M must be >= 1; got {M}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1; got {out_dim}")
        self.N = N
        self.M = M
        self.out_dim = out_dim

    @property
    @abstractmethod
    def vocabulary_size(self) -> int:
        """Total per-neuron choice cardinality. For diff-logic16-pruned this
        is `16 · N`; for the full pair form it would be `16 · N · N`. Used
        for diagnostics + reporting."""
        ...

    @abstractmethod
    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """Buffer `[B, N, M]` -> output word `[B, out_dim, M]`."""
        ...


# ----------------------------------------------------------------------
# Default family: difflogic16 (pruned, j=0)


@register_family("difflogic16")
class DiffLogic16Family(CrossSliceOpFamily):
    """Each output neuron picks a slice index `i ∈ [0, N)` and a binary op
    `op ∈ [0, 16)`, then computes

        out_bit_b = bin_op[op](buffer[B, i, b], buffer[B, 0, b])
                                                       ^^^
                                  the "latest" slice (j=0) — pruned form
                                  per proposal §Stage 4 task 2 line 296.

    Vocabulary size per neuron = `16 · N`. Two trainable softmaxes per neuron:

      - `op_logits  ∈ [out_dim, 16]`  → which of the 16 binary ops
      - `idx_logits ∈ [out_dim, N]`   → which slice row to pair with row 0

    Train: convex combination over `(op, i)`. Eval: argmax → discrete
    `(op, i)` per neuron, output is one-hot binary (matching `WordLogicLayer`
    eval semantics — verified by `test_eval_mode_outputs_are_binary` in
    `test_cross_slice_ops.py`).

    `i = 0` for all neurons makes the layer functionally a `WordLogicLayer`
    on row 0 of the buffer (the "compare latest with latest" degenerate
    case). At `N = 1`, the only choice is `i = 0` and the layer reduces
    bit-for-bit to a scalar `WordLogicLayer` — this is the streaming
    architecture's parity anchor against Stage 1.
    """

    family_name = "difflogic16"
    arity = 2

    def __init__(
        self,
        N: int,
        M: int,
        out_dim: int,
        connections: str = "random",
        device: str = "cuda",
    ) -> None:
        super().__init__(N=N, M=M, out_dim=out_dim)
        if connections not in ("random", "unique"):
            raise ValueError(f"connections must be 'random' or 'unique'; got {connections!r}")
        self.connections = connections
        self.device = device

        # Op softmax (16-way) — same as WordLogicLayer's `weights`.
        self.op_logits = nn.Parameter(torch.randn(out_dim, 16, device=device))
        # Slice-row softmax (N-way). At N=1 it's a single trivial bin.
        self.idx_logits = nn.Parameter(torch.randn(out_dim, N, device=device))

    @property
    def vocabulary_size(self) -> int:
        return 16 * self.N

    # ------------------------------------------------------------------

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        if buffer.dim() != 3 or buffer.shape[1] != self.N or buffer.shape[2] != self.M:
            raise ValueError(
                f"DiffLogic16Family expects buffer shape [B, {self.N}, {self.M}]; "
                f"got {tuple(buffer.shape)}"
            )
        B = buffer.shape[0]

        # row_0 is buffer[:, 0, :] — the "latest" slice, shape [B, M].
        # All neurons see the same row 0 (broadcasted), so we compute
        # `b = row_0` once and pair it with a soft mixture over slice rows.
        row_0 = buffer[:, 0, :]                                # [B, M]

        # Softmax over slice rows. Train: convex combination; eval: one-hot.
        if self.training:
            w_idx = F.softmax(self.idx_logits, dim=-1)                            # [out_dim, N]
        else:
            w_idx = F.one_hot(self.idx_logits.argmax(-1), self.N).to(buffer.dtype)  # [out_dim, N]

        # Softmax over the 16 binary ops — identical to WordLogicLayer.
        if self.training:
            w_op = F.softmax(self.op_logits, dim=-1)                              # [out_dim, 16]
        else:
            w_op = F.one_hot(self.op_logits.argmax(-1), 16).to(buffer.dtype)      # [out_dim, 16]

        # Soft "operand a" via convex combination over rows of the buffer.
        # buffer:    [B, N, M]
        # w_idx:     [out_dim, N]
        # We want:   a[b, n, j] = sum_i w_idx[n, i] * buffer[b, i, j]
        # einsum:    'bnm, on -> bom'
        a = torch.einsum("bnm,on->bom", buffer, w_idx)         # [B, out_dim, M]

        # b operand is row 0, broadcast across out_dim.
        b = row_0.unsqueeze(1).expand(B, self.out_dim, self.M).contiguous()  # [B, out_dim, M]

        # Apply 16-op soft mixture per (B, out_dim, j). bin_op_s expects
        # i_s broadcastable to [B, out_dim, M]; w_op is [out_dim, 16] →
        # unsqueeze to [out_dim, 1, 16] which broadcasts across batch + bit.
        return bin_op_s(a, b, w_op.unsqueeze(1))               # [B, out_dim, M]
