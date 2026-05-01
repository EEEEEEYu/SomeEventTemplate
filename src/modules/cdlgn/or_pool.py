"""Logical OR pooling (CDLGN, paper §3.1).

For Boolean activations, OR over a pooling window equals `max` (since
`a ∨ b = max(a, b)` on `{0,1}`). The paper relaxes this with the maximum
t-norm during training, which is also `max` on `[0,1]`-valued activations.

So both train- and eval-mode forward are exactly `F.max_pool2d`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrPool2d(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bool:
            # F.max_pool2d does not support bool; convert and convert back.
            return F.max_pool2d(
                x.to(torch.float32),
                kernel_size=self.kernel_size,
                stride=self.stride,
            ).to(torch.bool)
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"
