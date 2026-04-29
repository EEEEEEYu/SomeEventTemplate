"""P1 task 1 — verify the discretization toggle is genuinely engaging.

STATUS.md (2026-04-28 decisions log) claims "Lightning's auto `model.eval()`
already flips difflogic into the discretized branch." This test pins that
claim down: in `eval()`, `LogicLayer.forward` must dispatch through the
one-hot argmax path (difflogic/difflogic.py:106 / :128), producing
binary-valued output that's deterministic across repeated forward calls.
In `train()`, the softmax-relaxed path produces real-valued output in
[0, 1].

Without this test, a future refactor that breaks the toggle would manifest
as a Stage 1 parity drop with no obvious cause — the loss curve would look
fine but the discretized eval number would silently regress.
"""

from __future__ import annotations

import torch

from .conftest import needs_cuda, needs_difflogic, needs_difflogic_cuda


@needs_difflogic
@needs_difflogic_cuda
@needs_cuda
def test_eval_mode_outputs_are_binary_and_deterministic():
    from difflogic import LogicLayer

    torch.manual_seed(0)
    layer = LogicLayer(in_dim=64, out_dim=128, device="cuda").cuda()
    x = (torch.rand(16, 64, device="cuda") > 0.5).float()

    layer.eval()
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)

    assert torch.equal(y1, y2), "eval-mode forward must be deterministic across calls"
    unique_vals = torch.unique(y1)
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), \
        f"eval-mode output must be binary (one-hot argmax path); got {unique_vals.tolist()}"


@needs_difflogic
@needs_difflogic_cuda
@needs_cuda
def test_train_mode_outputs_are_relaxed_real_valued():
    from difflogic import LogicLayer

    torch.manual_seed(0)
    layer = LogicLayer(in_dim=64, out_dim=128, device="cuda").cuda()
    x = (torch.rand(16, 64, device="cuda") > 0.5).float()

    layer.train()
    y = layer(x)

    assert y.dtype in (torch.float32, torch.float16, torch.bfloat16), y.dtype
    assert (y >= 0).all() and (y <= 1).all(), "softmax-relaxed output should be in [0, 1]"
    unique_vals = torch.unique(y)
    assert unique_vals.numel() > 2, \
        "train-mode output should be real-valued (softmax-relaxed), not just {0, 1}"


@needs_difflogic
@needs_difflogic_cuda
@needs_cuda
def test_toggle_round_trip_preserves_eval_output():
    """eval -> train -> eval must give the same eval-mode output (the weights
    don't change between toggles, so the argmax-branch output is invariant)."""
    from difflogic import LogicLayer

    torch.manual_seed(0)
    layer = LogicLayer(in_dim=64, out_dim=128, device="cuda").cuda()
    x = (torch.rand(16, 64, device="cuda") > 0.5).float()

    layer.eval()
    with torch.no_grad():
        y_before = layer(x)

    layer.train()
    _ = layer(x)  # exercise the train branch

    layer.eval()
    with torch.no_grad():
        y_after = layer(x)

    assert torch.equal(y_before, y_after), \
        "eval-mode output must be invariant across train/eval toggles when weights are unchanged"
