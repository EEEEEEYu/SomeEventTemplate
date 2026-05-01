"""Backbone factories for the torchlogix-based pipeline.

Three families:
  - `cifar10_classifier_backbone(size)` — wraps torchlogix's
    `ClgnCifar10{Small,Medium,Large}` (binarization + 4 conv blocks + dense head
    + GroupSum). Returns the full `nn.Module` directly.
  - `gesture_classifier_backbone(...)` — DVS-Gesture-shaped 32×32 input with
    pre-encoded TBR boolean planes (no internal binarization). Logic conv
    stack + dense head + GroupSum, tuned for our gesture data shape.
  - `flow_backbone(...)` — fully convolutional logic stack, no GroupSum.
    Stub for Phase 2; populated when MVSEC arrives.

All factories return an `nn.Module`; the LightningModule wrappers
(`TorchlogixClassifier`, `TorchlogixFlow`) attach the loss / metrics / opt.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from torchlogix.layers import (
    GroupSum, LogicConv2d, LogicDense, OrPooling2d,
)
from torchlogix.models.conv import (
    ClgnCifar10Small, ClgnCifar10Medium, ClgnCifar10Large,
)


# ---------------------------------------------------------------------------
# CIFAR-10 classifier backbone (paper-spec via torchlogix's reference class)
# ---------------------------------------------------------------------------

_CIFAR10_REFERENCE = {
    "S": ClgnCifar10Small,
    "M": ClgnCifar10Medium,
    "L": ClgnCifar10Large,
}


def _build_thermometer_thresholds(n_input_bits: int, num_channels: int = 3) -> torch.Tensor:
    """Per-channel thresholds at `(i+1)/(n+1)` for i in [0, n_input_bits).
    Shape: `(num_channels, n_input_bits)`.
    """
    return torch.tensor([
        [(i + 1) / (n_input_bits + 1) for i in range(n_input_bits)]
        for _ in range(num_channels)
    ])


def cifar10_classifier_backbone(
    size: str = "M",
    grad_factor: float = 1.0,
    parametrization: str = "raw",
) -> nn.Module:
    """Returns a paper-spec `ClgnCifar10*` model: float (B, 3, 32, 32) input,
    integer-class logits (B, 10) output. Internal `FixedBinarization` applies
    the thermometer thresholds, so feed raw [0, 1] images."""
    if size not in _CIFAR10_REFERENCE:
        raise KeyError(f"size={size!r} not in {list(_CIFAR10_REFERENCE)}.")
    ref_cls = _CIFAR10_REFERENCE[size]
    thresholds = _build_thermometer_thresholds(
        n_input_bits=ref_cls.n_input_bits, num_channels=3,
    )
    return ref_cls(
        thresholds=thresholds,
        binarization="fixed",
        binarization_kwargs={},
        connections_kwargs={},
        grad_factor=grad_factor,
        parametrization=parametrization,
    )


# ---------------------------------------------------------------------------
# DVS-Gesture classifier backbone (no internal binarization)
# ---------------------------------------------------------------------------

def gesture_classifier_backbone(
    in_channels: int,
    spatial: int = 32,
    k: int = 32,
    num_classes: int = 11,
    tau: float = 20.0,
    channel_group_size: int = 2,
    grad_factor: float = 1.0,
    parametrization: str = "raw",
) -> nn.Module:
    """Logic-conv classifier for DVS-Gesture-shaped inputs.

    Input is expected to be a *pre-encoded* boolean (or {0,1}-float) tensor of
    shape `(B, in_channels, spatial, spatial)`. For our TBR DataModule with
    M=128 bins, in_channels = 2*128 = 256.

    Architecture mirrors `ClgnCifar10` but skips the internal binarization
    module since TBR already produces boolean activations.

    For 32×32 input the spatial trace is:
        32 → conv1 32 → pool1 16 → conv2 16 → pool2 8 → conv3 8 →
        pool3 4 → conv4 4 → pool4 2 → flatten 128k → dense head.
    """
    assert spatial == 32, (
        f"gesture backbone is wired for 32×32 input; got {spatial}. "
        "Adjust this factory if you change the TBR downsample target."
    )
    cgs = {"channel_group_size": channel_group_size}
    llkw = dict(grad_factor=grad_factor, parametrization=parametrization)

    # Final dense out_dim must be divisible by num_classes for GroupSum to
    # split it cleanly. Paper uses 320*k which is fine for CIFAR-10 (num_classes=10)
    # but breaks for DVS-Gesture (num_classes=11). Round up to nearest multiple.
    dense_out = ((320 * k + num_classes - 1) // num_classes) * num_classes

    layers = [
        LogicConv2d(
            in_dim=32, channels=in_channels, num_kernels=k,
            tree_depth=3, receptive_field_size=3, padding=1,
            connections_kwargs={"channel_group_size": 1},      # block 1 sees raw input
            **llkw,
        ),
        OrPooling2d(kernel_size=2, stride=2),                  # → k×16×16

        LogicConv2d(
            in_dim=16, channels=k, num_kernels=4 * k,
            tree_depth=3, receptive_field_size=3, padding=1,
            connections_kwargs=cgs, **llkw,
        ),
        OrPooling2d(kernel_size=2, stride=2),                  # → 4k×8×8

        LogicConv2d(
            in_dim=8, channels=4 * k, num_kernels=16 * k,
            tree_depth=3, receptive_field_size=3, padding=1,
            connections_kwargs=cgs, **llkw,
        ),
        OrPooling2d(kernel_size=2, stride=2),                  # → 16k×4×4

        LogicConv2d(
            in_dim=4, channels=16 * k, num_kernels=32 * k,
            tree_depth=3, receptive_field_size=3, padding=1,
            connections_kwargs=cgs, **llkw,
        ),
        OrPooling2d(kernel_size=2, stride=2),                  # → 32k×2×2

        nn.Flatten(),                                          # → 128k
        LogicDense(in_dim=128 * k, out_dim=1280 * k, **llkw),
        LogicDense(in_dim=1280 * k, out_dim=640 * k, **llkw),
        LogicDense(in_dim=640 * k, out_dim=dense_out, **llkw),
        GroupSum(k=num_classes, tau=tau),
    ]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Flow backbone — Phase 2 stub
# ---------------------------------------------------------------------------

def flow_backbone(*args, **kwargs) -> nn.Module:
    """Phase-2 flow-estimation backbone — fully convolutional logic stack
    followed by a small float head producing dense per-pixel 2D flow.

    NOT YET IMPLEMENTED. Populate when MVSEC data lands; design will share the
    early conv stack with `gesture_classifier_backbone` and replace the
    GroupSum with a dense float-head regressing flow vectors."""
    raise NotImplementedError(
        "flow_backbone is a Phase-2 stub. Populate when MVSEC is uploaded "
        "and the input encoding (TBR vs voxel grid vs event count) is decided."
    )
