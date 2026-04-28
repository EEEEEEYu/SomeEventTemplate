"""Stage 3 task 2: WordLogicLayer forward equivalence with difflogic.LogicLayer.

Strict-rigor test (proposal §Stage 3): with fixed connectivity and pre-discretized
operator choices, WordLogicLayer on packed [B, N, W] input must produce bit-for-bit
identical output to LogicLayer run W times on each bit-slice.

This file is a scaffold — concrete `WordLogicLayer` does not exist yet (Stage 3).
The body is in place so the test gains teeth the moment Stage 3 lands.
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
@pytest.mark.parametrize("W", [1, 8, 32])
def test_word_layer_forward_matches_difflogic(seed, W):
    from difflogic import LogicLayer
    from src.modules.word_logic import WordLogicLayer

    torch.manual_seed(seed)
    in_features = 64
    out_features = 32
    batch = 4

    # Same RNG path — must match Stage 3 connectivity-init audit (proposal §Stage 3 task 6).
    word_layer = WordLogicLayer(
        in_features=in_features, out_features=out_features, word_width=W,
        connections="random", seed=seed,
    ).cuda().eval()
    scalar_layer = LogicLayer(in_features, out_features).cuda().eval()
    # TODO Stage 3: copy operator choices and connectivity from word_layer → scalar_layer
    # so the comparison is meaningful. Until then, this assertion will fail and that
    # is intended — it forces the Stage 3 implementer to wire up the parity path.

    x_bits = (torch.rand(batch, in_features, W, device="cuda") > 0.5).float()

    word_out = word_layer.discretized_forward(x_bits)              # [B, out_features, W]

    scalar_outs = []
    for w in range(W):
        scalar_outs.append(scalar_layer(x_bits[..., w]))           # [B, out_features]
    scalar_out = torch.stack(scalar_outs, dim=-1)                  # [B, out_features, W]

    assert word_out.shape == scalar_out.shape
    assert torch.equal(word_out.bool(), scalar_out.bool()), \
        "Word-level discretized forward must match scalar replication bit-for-bit."
