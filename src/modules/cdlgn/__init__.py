"""Convolutional Differentiable Logic Gate Networks (Petersen et al. 2024).

See `docs/cdlgn_paper.md` for the architecture summary; this module provides
the primitives the paper defines:

- `ConvLogicLayer`: per-output-channel logic-gate tree applied at every
  spatial location with fixed-at-init local-RF connectivity.
- `OrPool2d`: logical OR pooling (relaxed via max t-norm during training).
- `residual_init_`: bias the per-gate softmax toward the canonical-A op
  (op index 3, "A" passthrough) to keep deep stacks trainable.
- `LogicTreeNet`: the full CIFAR-10 backbone (4 conv blocks + dense head).
"""

from src.modules.cdlgn.conv_logic import ConvLogicLayer
from src.modules.cdlgn.or_pool import OrPool2d
from src.modules.cdlgn.init import residual_init_
from src.modules.cdlgn.grouped_logic import GroupedLogicLayer
from src.modules.cdlgn.tree_net import LogicTreeNet

__all__ = [
    "ConvLogicLayer",
    "OrPool2d",
    "residual_init_",
    "GroupedLogicLayer",
    "LogicTreeNet",
]
