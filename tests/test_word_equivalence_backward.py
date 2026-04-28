"""Stage 3 task 3: WordLogicLayer relaxed forward + backward equivalence.

With soft (non-one-hot) operator distributions and continuous inputs in [0, 1],
output and gradients w.r.t. operator logits must match difflogic's path within
numerical tolerance. This catches relaxation bugs (e.g. subtly wrong T-norm)
that pure forward-on-hard-inputs misses.

Scaffold — bodies will be filled when Stage 3's WordLogicLayer lands.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from .conftest import needs_difflogic, needs_cuda


def _has_word_logic() -> bool:
    return importlib.util.find_spec("src.modules.word_logic") is not None


needs_word_logic = pytest.mark.skipif(
    not _has_word_logic(),
    reason="requires src/modules/word_logic.py (proposal §Stage 3 task 1)",
)


@needs_difflogic
@needs_cuda
@needs_word_logic
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_word_layer_backward_matches_difflogic(seed):
    from difflogic import LogicLayer
    from src.modules.word_logic import WordLogicLayer

    torch.manual_seed(seed)
    in_features = 32
    out_features = 16
    batch = 4
    W = 8

    word_layer = WordLogicLayer(
        in_features=in_features, out_features=out_features, word_width=W,
        connections="random", seed=seed,
    ).cuda().train()
    scalar_layer = LogicLayer(in_features, out_features).cuda().train()
    # TODO Stage 3: synchronize operator logits + connectivity between layers.

    x = torch.rand(batch, in_features, W, device="cuda", requires_grad=False)

    word_out = word_layer(x)                                       # [B, out, W]
    scalar_out = torch.stack([scalar_layer(x[..., w]) for w in range(W)], dim=-1)

    assert torch.allclose(word_out, scalar_out, atol=1e-5), \
        "Relaxed forward outputs differ between word and scalar paths."

    target = torch.zeros_like(word_out)
    word_loss = torch.nn.functional.mse_loss(word_out, target)
    scalar_loss = torch.nn.functional.mse_loss(scalar_out, target)

    word_grad = torch.autograd.grad(word_loss, word_layer.weights, retain_graph=False)[0]
    scalar_grad = torch.autograd.grad(scalar_loss, scalar_layer.weights, retain_graph=False)[0]

    assert torch.allclose(word_grad, scalar_grad, atol=1e-4), \
        "Operator-logit gradients differ between word and scalar paths."
