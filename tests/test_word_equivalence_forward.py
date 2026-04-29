"""Phase P §P3 / proposal §Stage 3 task 2 — WordLogicLayer forward parity.

The load-bearing claim of v2 Stage 3:
    WordLogicLayer(in, out, M=1) initialised under the same RNG state as
    difflogic.LogicLayer(in, out) produces bit-for-bit identical outputs in
    eval() mode.

Proposal v2 §Stage 3 line 219 / task 2 line 233 / line 243:
    "the only meaningful matched-accuracy test (`N=1` reproducing Stage 1's
    97.36%) lives in the Stage 4 verification block."

What 'bit-for-bit identical' requires (proposal §Stage 3 task 6 — connectivity-
init audit):
    WordLogicLayer must replicate `difflogic.LogicLayer.__init__`'s RNG calls
    in the *same order*: first `torch.randn(out, 16)` for the weights, then
    the connectivity-init randperms. Otherwise the two layers diverge on
    connectivity / op-choice and parity fails on initialisation, not on the
    forward pass — and we've debugged the wrong thing.

We assert that connectivity + weights are identical at init, then that forward
outputs match across `M ∈ {1, 8, 32}` × 3 seeds.

(For `M > 1`, parity means: WordLogicLayer's output at bit-slice `m` equals
LogicLayer's output on the input's bit-slice `m`. Same weights, same indices,
applied bitwise.)
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from .conftest import needs_cuda, needs_difflogic, needs_difflogic_cuda


def _has_word_logic() -> bool:
    return importlib.util.find_spec("src.modules.word_logic") is not None


needs_word_logic = pytest.mark.skipif(
    not _has_word_logic(),
    reason="requires src/modules/word_logic.py (Phase P §P3)",
)


def _make_paired_layers(in_dim: int, out_dim: int, M: int, seed: int):
    """Build a `WordLogicLayer` and a `difflogic.LogicLayer` whose weights and
    connectivity are bit-identical by sharing an RNG path.
    """
    from difflogic import LogicLayer
    from src.modules.word_logic import WordLogicLayer

    torch.manual_seed(seed)
    word = WordLogicLayer(
        in_dim=in_dim, out_dim=out_dim, M=M,
        device="cuda", grad_factor=1.0, connections="random",
    ).cuda().eval()

    torch.manual_seed(seed)
    scalar = LogicLayer(in_dim=in_dim, out_dim=out_dim, device="cuda").cuda().eval()

    return word, scalar


@needs_difflogic
@needs_difflogic_cuda
@needs_cuda
@needs_word_logic
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_connectivity_and_weights_match_difflogic_under_shared_seed(seed):
    """Connectivity-init audit: with shared RNG state, WordLogicLayer and
    LogicLayer must produce identical weights and indices. Without this,
    the forward parity test below would fail on init differences, not bugs."""
    word, scalar = _make_paired_layers(in_dim=64, out_dim=128, M=32, seed=seed)

    assert torch.equal(word.weights, scalar.weights), \
        "WordLogicLayer and LogicLayer must produce identical weight tensors under " \
        "shared seed (proposal §Stage 3 connectivity-init audit). RNG-call order in " \
        "WordLogicLayer.__init__ has likely diverged from LogicLayer.__init__."
    assert torch.equal(word.indices_a, scalar.indices[0]), \
        "WordLogicLayer.indices_a must match LogicLayer.indices[0] under shared seed."
    assert torch.equal(word.indices_b, scalar.indices[1]), \
        "WordLogicLayer.indices_b must match LogicLayer.indices[1] under shared seed."


@needs_difflogic
@needs_difflogic_cuda
@needs_cuda
@needs_word_logic
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("M", [1, 8, 32])
def test_word_layer_forward_matches_difflogic_per_bit_slice(seed, M):
    """For each bit-slice m, WordLogicLayer's output bit-slice m must equal
    LogicLayer's output on the input's bit-slice m. The whole point of the
    word substrate is that it's M parallel scalar networks — this test pins
    that property down."""
    in_dim, out_dim, batch = 64, 128, 4
    word, scalar = _make_paired_layers(in_dim, out_dim, M, seed)

    x_bits = (torch.rand(batch, in_dim, M, device="cuda") > 0.5).float()

    with torch.no_grad():
        word_out = word(x_bits)  # [B, out_dim, M]
        scalar_outs = []
        for m in range(M):
            scalar_outs.append(scalar(x_bits[..., m]))  # [B, out_dim]
        scalar_out = torch.stack(scalar_outs, dim=-1)  # [B, out_dim, M]

    assert word_out.shape == scalar_out.shape, (word_out.shape, scalar_out.shape)
    assert torch.equal(word_out, scalar_out), \
        "WordLogicLayer must equal LogicLayer applied bit-by-bit (eval mode, " \
        "shared init). At M=1 this is the load-bearing parity anchor for v2 Stage 3."


@needs_difflogic
@needs_difflogic_cuda
@needs_cuda
@needs_word_logic
@pytest.mark.parametrize("seed", [0, 42])
def test_word_layer_M1_squeeze_matches_difflogic_2d(seed):
    """The most important degenerate case: M=1, output squeezed to 2D, equals
    LogicLayer's output on a 2D input. Anchors the streaming `N=1` parity
    check that lives in Stage 4 task 6."""
    in_dim, out_dim, batch = 64, 128, 8
    word, scalar = _make_paired_layers(in_dim, out_dim, M=1, seed=seed)

    x_2d = (torch.rand(batch, in_dim, device="cuda") > 0.5).float()
    x_3d = x_2d.unsqueeze(-1)  # [B, in_dim, 1]

    with torch.no_grad():
        word_out = word(x_3d).squeeze(-1)  # [B, out_dim]
        scalar_out = scalar(x_2d)          # [B, out_dim]

    assert torch.equal(word_out, scalar_out)
