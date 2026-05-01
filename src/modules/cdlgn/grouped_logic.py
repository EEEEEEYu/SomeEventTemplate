"""Grouped wrapper around `difflogic.LogicLayer` (CDLGN, paper §A.3).

The paper restricts connectivity such that the model "could be split into k/8
separated models that are only recombined at the stage of output gates after
accumulation". The conv blocks already honor this via `ConvLogicLayer`'s
`channel_groups`. This module adds the matching restriction to the dense head:

    GroupedLogicLayer(in_dim, out_dim, num_groups)
        ≡ num_groups parallel LogicLayer(in_dim/g, out_dim/g)
        applied along disjoint slices of the input/output channel axes.

Each output gate's two inputs are sampled only from inputs in its own group
(LogicLayer's connectivity sampler is unchanged — we just feed it a smaller
input slice). The GroupSum at the end of the network then mixes across all
groups when summing per-class outputs, which is the "recombination" the paper
describes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from difflogic import LogicLayer


class GroupedLogicLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_groups: int,
        device: str = "cuda",
        grad_factor: float = 1.0,
    ):
        super().__init__()
        if in_dim % num_groups != 0:
            raise ValueError(
                f"in_dim ({in_dim}) must be divisible by num_groups ({num_groups})."
            )
        if out_dim % num_groups != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_groups ({num_groups})."
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        self.in_per_group = in_dim // num_groups
        self.out_per_group = out_dim // num_groups

        self.groups = nn.ModuleList([
            LogicLayer(
                self.in_per_group,
                self.out_per_group,
                device=device,
                grad_factor=grad_factor,
            )
            for _ in range(num_groups)
        ])

    def extra_repr(self) -> str:
        return (
            f"{self.in_dim}, {self.out_dim}, num_groups={self.num_groups} "
            f"({self.in_per_group} → {self.out_per_group} per group)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim). Split into num_groups slices along the last dim
        # and run each LogicLayer on its own slice. Outputs are concatenated in
        # the original group order.
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected last dim {self.in_dim}, got {x.shape[-1]} (shape {tuple(x.shape)})"
            )
        chunks = x.chunk(self.num_groups, dim=-1)
        outs = [layer(chunk) for layer, chunk in zip(self.groups, chunks)]
        return torch.cat(outs, dim=-1)
