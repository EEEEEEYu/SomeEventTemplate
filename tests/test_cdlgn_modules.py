"""CDLGN primitive correctness — CPU only.

Covers: ConvLogicLayer connectivity invariance, eval-mode discretization,
OrPool2d semantics, residual_init_ effects, and receptive-field locality.
"""

import pytest
import torch
import torch.nn.functional as F

from src.modules.cdlgn import ConvLogicLayer, OrPool2d, residual_init_, GroupedLogicLayer
from src.modules.cdlgn.init import CANONICAL_A_OP_INDEX
from src.modules.cdlgn.conv_logic import _apply_bin_op_s
from difflogic.functional import bin_op, bin_op_s


# ---------------------------------------------------------------------------
# OrPool2d
# ---------------------------------------------------------------------------

def test_orpool_boolean_input_equals_or():
    pool = OrPool2d(kernel_size=2, stride=2)
    x = torch.tensor([
        [[[0, 1, 0, 0],
          [0, 0, 0, 1],
          [1, 0, 1, 0],
          [0, 0, 0, 0]]]
    ], dtype=torch.bool)
    out = pool(x)
    expected = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.bool)
    assert torch.equal(out, expected)


def test_orpool_real_valued_equals_max():
    pool = OrPool2d(kernel_size=2, stride=2)
    x = torch.rand(2, 4, 8, 8)
    expected = F.max_pool2d(x, 2, 2)
    assert torch.allclose(pool(x), expected)


# ---------------------------------------------------------------------------
# ConvLogicLayer — basics
# ---------------------------------------------------------------------------

def test_conv_logic_output_shape():
    layer = ConvLogicLayer(in_channels=4, out_channels=8, kernel_size=3,
                           tree_depth=3, padding=1)
    x = torch.rand(2, 4, 8, 8)
    out = layer(x)
    assert out.shape == (2, 8, 8, 8)


def test_conv_logic_output_shape_with_stride():
    layer = ConvLogicLayer(in_channels=3, out_channels=8, kernel_size=3,
                           tree_depth=3, stride=2, padding=1)
    x = torch.rand(1, 3, 8, 8)
    out = layer(x)
    assert out.shape == (1, 8, 4, 4)


def test_conv_logic_with_5x5_kernel_padding():
    layer = ConvLogicLayer(in_channels=9, out_channels=32, kernel_size=5,
                           tree_depth=3, padding=2)
    x = torch.rand(2, 9, 32, 32)
    out = layer(x)
    assert out.shape == (2, 32, 32, 32)


# ---------------------------------------------------------------------------
# Connectivity invariance — leaves are buffers, not parameters
# ---------------------------------------------------------------------------

def test_leaves_are_not_learnable():
    layer = ConvLogicLayer(in_channels=4, out_channels=8, kernel_size=3,
                           tree_depth=3, padding=1)
    # Only `weights` is a Parameter; leaves and leaves_local are buffers.
    param_names = {name for name, _ in layer.named_parameters()}
    assert param_names == {"weights"}
    buffer_names = {name for name, _ in layer.named_buffers()}
    assert "leaves" in buffer_names


def test_leaves_unchanged_after_optimizer_step():
    torch.manual_seed(0)
    layer = ConvLogicLayer(in_channels=4, out_channels=8, kernel_size=3,
                           tree_depth=3, padding=1)
    leaves_before = layer.leaves.clone()

    x = torch.rand(2, 4, 8, 8)
    target = torch.zeros(2, 8, 8, 8)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    opt.zero_grad()
    loss = (layer(x) - target).pow(2).mean()
    loss.backward()
    opt.step()

    assert torch.equal(layer.leaves, leaves_before)


# ---------------------------------------------------------------------------
# Discretization at eval
# ---------------------------------------------------------------------------

def test_eval_outputs_boolean_on_boolean_input():
    layer = ConvLogicLayer(in_channels=4, out_channels=8, kernel_size=3,
                           tree_depth=3, padding=1)
    layer.eval()
    x = (torch.rand(2, 4, 8, 8) > 0.5).to(torch.bool)
    with torch.no_grad():
        out = layer(x)
    # On Boolean inputs at eval, every gate is one-hot so output is in {0, 1}.
    unique_vals = torch.unique(out)
    for v in unique_vals.tolist():
        assert v in (0.0, 1.0), f"unexpected eval-mode output value {v}"


def test_train_outputs_real_valued():
    layer = ConvLogicLayer(in_channels=4, out_channels=8, kernel_size=3,
                           tree_depth=3, padding=1)
    layer.train()
    x = (torch.rand(2, 4, 8, 8) > 0.5).to(torch.float32)
    out = layer(x)
    # In train mode the softmax is non-degenerate and outputs are non-Boolean.
    assert (out > 0).any() and (out < 1).any()


# ---------------------------------------------------------------------------
# Residual init
# ---------------------------------------------------------------------------

def test_residual_init_sets_canonical_a_logit():
    layer = ConvLogicLayer(in_channels=4, out_channels=8, kernel_size=3,
                           tree_depth=3, padding=1)
    residual_init_(layer, z3=5.0)
    softmax = torch.softmax(layer.weights, dim=-1)
    a_mass = softmax[..., CANONICAL_A_OP_INDEX]
    assert (a_mass > 0.85).all(), f"A-op softmax mass {a_mass.min():.3f} below 0.85"
    other_mass = softmax.sum(-1) - a_mass
    assert (other_mass < 0.15).all()


def test_residual_init_makes_layer_approximate_passthrough_a():
    """After residual_init_, the train-mode forward should be close to evaluating
    the canonical-A op at every gate — i.e. the output of each tree equals the
    leaf at position 0 of that tree (the `a` input of the root gate's leaf
    pair, which after recursive A-passthroughs just propagates leaf 0)."""
    torch.manual_seed(0)
    layer = ConvLogicLayer(in_channels=4, out_channels=4, kernel_size=3,
                           tree_depth=3, padding=1)
    residual_init_(layer, z3=8.0)            # z3=8 → ~0.997 mass on A
    layer.eval()                             # use exact one-hot for the assertion
    x = (torch.rand(1, 4, 4, 4) > 0.5).to(torch.float32)
    with torch.no_grad():
        out = layer(x)

    # Compute the expected output by manually applying canonical-A at every gate.
    # In tree-forward, level 0 has 2^d leaves; canonical-A means each gate
    # outputs its `a` input. After d levels of "always pick a", the surviving
    # leaf is leaves[oc, 0].
    patches = F.unfold(x, kernel_size=3, padding=1, stride=1)            # (1, 4*9, L)
    L = patches.shape[-1]
    # leaves[oc, 0] gives the first-leaf flat index
    leaf0_idx = layer.leaves[:, 0]                                       # (out_channels,)
    expected = patches[0, leaf0_idx, :].view(layer.out_channels, 4, 4)
    expected = expected.unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Receptive field locality
# ---------------------------------------------------------------------------

def test_changing_pixel_only_affects_outputs_in_its_rf():
    """Toggle one input pixel and verify that only output positions whose RF
    contains that pixel can change. With kernel=3, padding=1, stride=1 the RF of
    output (i, j) is the 3x3 patch centered on input (i, j). Changing input
    (4, 4) can only affect outputs at (3..5, 3..5)."""
    torch.manual_seed(0)
    layer = ConvLogicLayer(in_channels=2, out_channels=4, kernel_size=3,
                           tree_depth=2, padding=1)
    layer.eval()
    x = (torch.rand(1, 2, 8, 8) > 0.5).to(torch.float32)
    with torch.no_grad():
        out_a = layer(x)
        x_perturbed = x.clone()
        # Flip both channels at (4, 4) to maximize chance of triggering change
        x_perturbed[0, :, 4, 4] = 1.0 - x[0, :, 4, 4]
        out_b = layer(x_perturbed)

    diff = (out_a - out_b).abs() > 1e-6
    # Outside the (3..5, 3..5) RF of (4, 4), there must be no change.
    diff[:, :, 3:6, 3:6] = False
    assert not diff.any(), (
        f"output changed outside the RF of input (4,4); positions: "
        f"{diff.nonzero(as_tuple=False).tolist()}"
    )


# ---------------------------------------------------------------------------
# Channel groups
# ---------------------------------------------------------------------------

def test_apply_bin_op_s_matches_difflogic_bin_op_s():
    """The bilinear-form `_apply_bin_op_s` rewrite (W0 + W1·a + W2·b + W3·ab)
    must produce the same values as `difflogic.functional.bin_op_s` (the
    explicit Σᵢ wᵢ · binᵢ(a,b) loop). Tolerance reflects FP32 reduction order."""
    torch.manual_seed(0)
    B, oc, n_pairs, L = 2, 4, 3, 5
    a = torch.rand(B, oc, n_pairs, L)
    b = torch.rand(B, oc, n_pairs, L)
    # Random softmax weights of the shape `_apply_bin_op_s` expects.
    raw = torch.randn(oc, n_pairs, 16)
    w_softmax = torch.softmax(raw, dim=-1)
    w_broadcast = w_softmax.view(1, oc, n_pairs, 1, 16)

    out_bilinear = _apply_bin_op_s(a, b, w_broadcast)
    out_reference = bin_op_s(a, b, w_broadcast.expand(B, oc, n_pairs, L, 16))
    assert torch.allclose(out_bilinear, out_reference, atol=1e-5, rtol=1e-5), (
        f"bilinear-form _apply_bin_op_s diverges from difflogic.bin_op_s; "
        f"max diff = {(out_bilinear - out_reference).abs().max().item():.2e}"
    )


def test_apply_bin_op_s_one_hot_picks_single_op():
    """With one-hot weights at op=3 (canonical-A passthrough), the output must
    equal `a` exactly. This is the eval-time discretization regime."""
    torch.manual_seed(0)
    B, oc, n_pairs, L = 2, 4, 3, 5
    a = torch.rand(B, oc, n_pairs, L)
    b = torch.rand(B, oc, n_pairs, L)
    w = torch.zeros(1, oc, n_pairs, 1, 16)
    w[..., CANONICAL_A_OP_INDEX] = 1.0
    out = _apply_bin_op_s(a, b, w)
    assert torch.allclose(out, a, atol=1e-6)

    # Op 5 (= "B") should give exactly `b`.
    w.zero_()
    w[..., 5] = 1.0
    assert torch.allclose(_apply_bin_op_s(a, b, w), b, atol=1e-6)

    # Op 6 (= "xor", a + b - 2ab on relaxed inputs).
    w.zero_()
    w[..., 6] = 1.0
    assert torch.allclose(_apply_bin_op_s(a, b, w), a + b - 2 * a * b, atol=1e-6)


# ---------------------------------------------------------------------------
# GroupedLogicLayer
# ---------------------------------------------------------------------------

def test_grouped_logic_output_shape():
    layer = GroupedLogicLayer(in_dim=64, out_dim=128, num_groups=4, device="cpu")
    x = torch.rand(2, 64)
    out = layer(x)
    assert out.shape == (2, 128)


def test_grouped_logic_isolation_between_groups():
    """An input bit in group g must only be able to affect output bits in
    group g (and never bits in other groups). Verify by perturbing one input
    and observing which output positions change."""
    torch.manual_seed(0)
    layer = GroupedLogicLayer(in_dim=64, out_dim=128, num_groups=4, device="cpu")
    layer.eval()
    x = (torch.rand(1, 64) > 0.5).to(torch.float32)
    with torch.no_grad():
        out_a = layer(x)
        # Perturb input position 5 (in group 0: positions 0..15).
        x_perturbed = x.clone()
        x_perturbed[0, 5] = 1.0 - x[0, 5]
        out_b = layer(x_perturbed)

    diff = (out_a - out_b).abs() > 1e-6
    # Only output positions 0..31 (group 0: 32 outputs per group) may differ.
    assert not diff[0, 32:].any(), (
        f"output changed outside group 0; diffed positions: "
        f"{diff[0, 32:].nonzero(as_tuple=False).tolist()}"
    )


def test_grouped_logic_residual_init_propagates_to_subgroups():
    """`residual_init_` walks `model.modules()` and should hit every nested
    LogicLayer inside a GroupedLogicLayer."""
    layer = GroupedLogicLayer(in_dim=32, out_dim=64, num_groups=4, device="cpu")
    residual_init_(layer, z3=5.0)
    for sub_layer in layer.groups:
        softmax = torch.softmax(sub_layer.weights, dim=-1)
        a_mass = softmax[..., CANONICAL_A_OP_INDEX]
        assert (a_mass > 0.85).all(), f"group sub-layer A-op mass too low: {a_mass.min()}"


def test_grouped_logic_in_dim_indivisible_raises():
    with pytest.raises(ValueError, match="in_dim"):
        GroupedLogicLayer(in_dim=63, out_dim=128, num_groups=4, device="cpu")


def test_channel_groups_indices_stay_within_group():
    layer = ConvLogicLayer(in_channels=16, out_channels=32, kernel_size=3,
                           tree_depth=3, padding=1, channel_groups=4)
    rf_per_group = (16 // 4) * 3 * 3
    out_per_group = 32 // 4
    for oc in range(32):
        g = oc // out_per_group
        lo = g * rf_per_group
        hi = (g + 1) * rf_per_group
        leaves = layer.leaves[oc]
        assert (leaves >= lo).all() and (leaves < hi).all(), (
            f"oc={oc} (group {g}) has leaves {leaves.tolist()} outside [{lo}, {hi})"
        )
