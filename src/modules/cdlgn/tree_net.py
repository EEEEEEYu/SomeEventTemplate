"""LogicTreeNet — full CDLGN backbone for CIFAR-10 (paper §A.1.1).

Architecture (for width parameter `k`):

    Input  : (B, n_bits*3, 32, 32)
    Block 1: ConvLogicLayer(in=n_bits*3, out=k,    K=5, padding=2, d=tree_depth)
             OrPool2d(2, 2)                                      → (B, k,    16, 16)
    Block 2: ConvLogicLayer(in=k,        out=4k,   K=3, padding=1, d=tree_depth)
             OrPool2d(2, 2)                                      → (B, 4k,    8,  8)
    Block 3: ConvLogicLayer(in=4k,       out=16k,  K=3, padding=1, d=tree_depth)
             OrPool2d(2, 2)                                      → (B, 16k,   4,  4)
    Block 4: ConvLogicLayer(in=16k,      out=32k,  K=3, padding=1, d=tree_depth)
             OrPool2d(2, 2)                                      → (B, 32k,   2,  2)
    Flatten                                                       → (B, 128k)
    Dense  : LogicLayer(128k → 1280k)
             LogicLayer(1280k → 640k)
             LogicLayer(640k  → 320k)
    GroupSum(k_groups=10, tau)                                    → (B, 10)

Width parameter selects S/M/B/L/G:
    S → k=32 (n_bits=3)
    M → k=256 (n_bits=3)
    B → k=512 (n_bits=31)
    L → k=1024 (n_bits=31)
    G → k=2560 (n_bits=31)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from difflogic import LogicLayer, GroupSum
from src.modules.cdlgn.conv_logic import ConvLogicLayer
from src.modules.cdlgn.or_pool import OrPool2d
from src.modules.cdlgn.init import residual_init_
from src.modules.cdlgn.grouped_logic import GroupedLogicLayer


class LogicTreeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        k: int = 32,
        tree_depth: int = 3,
        channel_groups: int = 1,
        dense_channel_groups: int = 1,
        num_classes: int = 10,
        tau: float = 20.0,
        residual_init_z3: float = 5.0,
        grad_factor: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        if num_classes != 10:
            # The dense head's final width is hard-coded to 320k for k_groups=10
            # per paper §A.1.1. Other class counts can be supported by adjusting
            # the final dense layer's out_dim to be a multiple of num_classes.
            raise NotImplementedError(
                f"LogicTreeNet currently assumes num_classes=10 (CIFAR-10/MNIST); "
                f"got {num_classes}."
            )

        self.in_channels = in_channels
        self.k = k
        self.num_classes = num_classes

        # 4 conv blocks (CIFAR-10 has 4; MNIST uses 3 with 5×5 first kernel).
        # Per paper §A.1.1, all CIFAR-10 conv blocks use 3×3 RF with padding=1.
        self.block1_conv = ConvLogicLayer(
            in_channels=in_channels, out_channels=k, kernel_size=3,
            tree_depth=tree_depth, stride=1, padding=1,
            channel_groups=1,                                       # block 1 sees raw input
            grad_factor=grad_factor,
        )
        self.block1_pool = OrPool2d(2, 2)

        self.block2_conv = ConvLogicLayer(
            in_channels=k, out_channels=4 * k, kernel_size=3,
            tree_depth=tree_depth, stride=1, padding=1,
            channel_groups=channel_groups, grad_factor=grad_factor,
        )
        self.block2_pool = OrPool2d(2, 2)

        self.block3_conv = ConvLogicLayer(
            in_channels=4 * k, out_channels=16 * k, kernel_size=3,
            tree_depth=tree_depth, stride=1, padding=1,
            channel_groups=channel_groups, grad_factor=grad_factor,
        )
        self.block3_pool = OrPool2d(2, 2)

        self.block4_conv = ConvLogicLayer(
            in_channels=16 * k, out_channels=32 * k, kernel_size=3,
            tree_depth=tree_depth, stride=1, padding=1,
            channel_groups=channel_groups, grad_factor=grad_factor,
        )
        self.block4_pool = OrPool2d(2, 2)

        # Spatial trace (CIFAR-10 input 32x32; conv blocks preserve spatial via
        # padding; each OR-pool halves):
        #   32 → block1 32 → pool1 16 → block2 16 → pool2 8 → block3 8 →
        #   pool3 4 → block4 4 → pool4 2.
        # Flatten the (B, 32k, 2, 2) tensor → 32k * 2 * 2 = 128k. Published
        # dense head: 128k → 1280k → 640k → 320k → GroupSum(10).
        flat_dim = 32 * k * 2 * 2                                   # = 128k
        self.flatten = nn.Flatten()

        def _dense(in_dim: int, out_dim: int, grouped: bool) -> nn.Module:
            """Plain LogicLayer or GroupedLogicLayer per paper §A.3."""
            if not grouped or dense_channel_groups == 1:
                return LogicLayer(in_dim=in_dim, out_dim=out_dim,
                                  device=device, grad_factor=grad_factor)
            return GroupedLogicLayer(
                in_dim=in_dim, out_dim=out_dim,
                num_groups=dense_channel_groups,
                device=device, grad_factor=grad_factor,
            )

        # Per paper §A.3, the k/8 group split is "only recombined at the stage
        # of output gates after accumulation" — i.e., the last dense layer is
        # ungrouped so it can mix across all groups before the GroupSum reads
        # contiguous per-class slices. Without this, the GroupSum sums only
        # within a single group and accuracy collapses (S regressed -12 pp).
        self.dense1 = _dense(flat_dim,           10 * flat_dim,         grouped=True)
        self.dense2 = _dense(10 * flat_dim,      5  * flat_dim,         grouped=True)
        self.dense3 = _dense(5  * flat_dim,      int(2.5 * flat_dim),   grouped=False)
        # 2.5 * 128k = 320k. flat_dim=128k is always divisible by 4 since 128 is.
        self.group_sum = GroupSum(k=num_classes, tau=tau, device=device)

        residual_init_(self, z3=residual_init_z3)

        # Layer groups for diagnostics (encoder = conv blocks, decoder = dense head).
        self.layer_groups = {
            "encoder": ["block1_conv", "block2_conv", "block3_conv", "block4_conv"],
            "decoder": ["dense1", "dense2", "dense3"],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1_conv(x)
        x = self.block1_pool(x)                                     # 32x32 → 16x16
        x = self.block2_conv(x)
        x = self.block2_pool(x)                                     # 16x16 → 8x8
        x = self.block3_conv(x)
        x = self.block3_pool(x)                                     #  8x8 → 4x4
        x = self.block4_conv(x)
        x = self.block4_pool(x)                                     #  4x4 → 2x2
        x = self.flatten(x)                                         # → 128k
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.group_sum(x)
        return x
