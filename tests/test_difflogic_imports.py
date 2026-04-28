"""Stage 0 task 3: assert difflogic and its CUDA extension import cleanly and
that a forward pass works on a random tensor. Skips on the login node."""

from __future__ import annotations

import torch

from .conftest import needs_difflogic, needs_difflogic_cuda, needs_cuda


@needs_difflogic_cuda
def test_difflogic_cuda_imports():
    import difflogic_cuda  # noqa: F401


@needs_difflogic
def test_difflogic_high_level_imports():
    from difflogic import LogicLayer, GroupSum, CompiledLogicNet  # noqa: F401


@needs_difflogic
@needs_cuda
def test_difflogic_forward_pass_smoke():
    from difflogic import LogicLayer, GroupSum

    in_dim = 64
    num_classes = 10
    hidden = 320          # GroupSum requires hidden % num_classes == 0

    net = torch.nn.Sequential(
        LogicLayer(in_dim, hidden),
        LogicLayer(hidden, hidden),
        GroupSum(k=num_classes, tau=10.0),
    ).cuda()

    x = (torch.rand(8, in_dim, device="cuda") > 0.5).float()
    y = net(x)
    assert y.shape == (8, num_classes)
    assert torch.isfinite(y).all()
