"""Phase P §P4c — `ShiftedWordLogicLayer` discretized correctness + parity.

Three claims under test:
  1. **Hand-verified op + shift.** With a single neuron locked to (op_idx, s),
     the layer's discretized forward equals the reference `op(a, rotr(b, s))`
     applied bitwise.
  2. **`shift=0` parity.** A `ShiftedWordLogicLayer` with all neurons forced
     to shift=0 (and identical weights+indices) reproduces a `WordLogicLayer`
     bit-for-bit. This is the parity anchor for the shifted layer (proposal
     v2.1 §P4c task 3 in the Phase P plan).
  3. **`M=1` collapse.** At M=1 there's only one possible shift (0) and the
     shifted layer is functionally identical to `WordLogicLayer` regardless
     of shift_weights values.

Reference op table mirrors `difflogic.functional.bin_op` (line 26 of
difflogic/difflogic/functional.py). We reproduce a few entries here for the
hand-checked cases — keeping the test self-contained.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from .conftest import needs_cuda, needs_difflogic, needs_difflogic_cuda


def _has_shifted_word_logic() -> bool:
    return importlib.util.find_spec("src.modules.shifted_word_logic") is not None


needs_shifted_word_logic = pytest.mark.skipif(
    not _has_shifted_word_logic(),
    reason="requires src/modules/shifted_word_logic.py (Phase P §P4c)",
)


def _reference_binary_op(a: torch.Tensor, b: torch.Tensor, op_idx: int) -> torch.Tensor:
    """Hand-coded {0, 1} reference for a few of difflogic's 16 ops. We only
    need the ops we test against — coverage of the full vocabulary is the
    job of the WordLogicLayer parity test."""
    if op_idx == 1:    # A AND B
        return a * b
    if op_idx == 6:    # A XOR B
        return (a + b) % 2
    if op_idx == 7:    # A OR B
        return ((a + b) > 0).to(a.dtype)
    if op_idx == 9:    # NOT(A XOR B) = A == B
        return ((a + b + 1) % 2)
    raise ValueError(f"reference for op_idx={op_idx} not implemented in this test")


def _force_neuron(layer, op_idx: int, shift: int) -> None:
    """Lock a single-neuron layer to a chosen (op, shift) pair via one-hot
    weights + shift_weights. After this, layer.eval() picks (op_idx, shift)."""
    with torch.no_grad():
        layer.weights.zero_()
        layer.weights[:, op_idx] = 10.0
        layer.shift_weights.zero_()
        layer.shift_weights[:, shift] = 10.0


@needs_shifted_word_logic
@pytest.mark.parametrize("op_idx", [1, 6, 7, 9])
@pytest.mark.parametrize("shift", [0, 1, 4, 7])
def test_shifted_layer_matches_reference_op_then_shift(op_idx, shift):
    """A single-neuron `ShiftedWordLogicLayer` locked to (op, s) on a 2-input
    feed should equal `op(a, rotr(b, s))` bit-for-bit."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    M = 8
    layer = ShiftedWordLogicLayer(in_dim=2, out_dim=1, M=M, device="cpu",
                                  connections="random").eval()
    # Force connectivity to the trivial pair (a from input 0, b from input 1).
    with torch.no_grad():
        layer.indices_a.zero_()
        layer.indices_b.fill_(1)
    _force_neuron(layer, op_idx, shift)

    torch.manual_seed(0)
    x = (torch.rand(4, 2, M) > 0.5).float()
    a = x[:, 0, :]   # [B, M]
    b = x[:, 1, :]   # [B, M]

    expected = _reference_binary_op(a, torch.roll(b, shifts=shift, dims=-1), op_idx)
    expected = expected.unsqueeze(1)  # [B, 1, M]

    with torch.no_grad():
        actual = layer(x)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), \
        f"op={op_idx} shift={shift}: layer output disagrees with op(a, rotr(b, s)) reference"


@needs_shifted_word_logic
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("shift_lut", [None, (0, 1, 2, 4, 8)])
def test_shift_zero_matches_word_logic_layer_bit_for_bit(seed, shift_lut):
    """When all shifts are forced to 0 and weights+indices are shared,
    `ShiftedWordLogicLayer` reproduces `WordLogicLayer` bit-for-bit. This is
    the load-bearing parity anchor for the shifted layer.

    Parametrised over `shift_lut`:
      - `None` → full M-alphabet (the original behavior)
      - `(0, 1, 2, 4, 8)` → reduced LUT; LUT index 0 maps to shift=0, so the
        invariant must hold here too.
    """
    from src.modules.word_logic import WordLogicLayer
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    in_dim, out_dim, M = 32, 64, 16

    torch.manual_seed(seed)
    base = WordLogicLayer(in_dim=in_dim, out_dim=out_dim, M=M, device="cpu").eval()

    torch.manual_seed(seed)
    shifted = ShiftedWordLogicLayer(in_dim=in_dim, out_dim=out_dim, M=M, device="cpu",
                                    shift_lut=shift_lut).eval()
    # Copy weights + indices; force all shifts to 0 (LUT index 0 == shift=0).
    with torch.no_grad():
        shifted.weights.copy_(base.weights)
        shifted.indices_a.copy_(base.indices_a)
        shifted.indices_b.copy_(base.indices_b)
    shifted.force_zero_shift_()

    x = (torch.rand(8, in_dim, M) > 0.5).float()
    with torch.no_grad():
        y_base = base(x)
        y_shifted = shifted(x)

    assert torch.equal(y_base, y_shifted), \
        f"ShiftedWordLogicLayer(shift_lut={shift_lut}) with LUT index 0 forced must " \
        f"match WordLogicLayer."


@needs_shifted_word_logic
def test_M1_layer_is_invariant_to_shift_weights():
    """At M=1 the only possible shift is 0; `shift_weights` are a no-op.
    The forward output should not depend on `shift_weights` values."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    layer = ShiftedWordLogicLayer(in_dim=8, out_dim=4, M=1, device="cpu").eval()
    x = (torch.rand(2, 8, 1) > 0.5).float()
    with torch.no_grad():
        y0 = layer(x)
        with torch.no_grad():
            layer.shift_weights.fill_(7.7)
        y1 = layer(x)
    assert torch.equal(y0, y1)


@needs_shifted_word_logic
def test_train_mode_returns_real_valued_relaxed_output():
    """Soft path: training-mode output should be in [0, 1] and not just {0, 1}."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    layer = ShiftedWordLogicLayer(in_dim=16, out_dim=32, M=8, device="cpu").train()
    x = (torch.rand(2, 16, 8) > 0.5).float()
    y = layer(x)
    assert (y >= 0).all() and (y <= 1).all()
    assert torch.unique(y).numel() > 2, "soft-path output should be real-valued"


# ---------------------------------------------------------------------------
# shift_lut path coverage (Phase P optimization plan §Tier B)


@needs_shifted_word_logic
def test_lut_indices_map_correctly():
    """A neuron forced to LUT index 1 (which maps to shift=4 in the LUT below)
    must produce `op(a, rotr(b, 4))` — verifies the LUT lookup wiring."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    M = 8
    layer = ShiftedWordLogicLayer(in_dim=2, out_dim=1, M=M, device="cpu",
                                  shift_lut=(0, 4)).eval()
    # Trivial connectivity: a from input 0, b from input 1.
    with torch.no_grad():
        layer.indices_a.zero_()
        layer.indices_b.fill_(1)
        # Lock op to XOR (op_idx=6).
        layer.weights.zero_()
        layer.weights[:, 6] = 10.0
        # Lock LUT index to 1 → shift_lut[1] == 4.
        layer.shift_weights.zero_()
        layer.shift_weights[:, 1] = 10.0

    torch.manual_seed(0)
    x = (torch.rand(4, 2, M) > 0.5).float()
    a = x[:, 0, :]
    b = x[:, 1, :]
    expected = ((a + torch.roll(b, shifts=4, dims=-1)) % 2).unsqueeze(1)  # XOR after rotr by 4

    with torch.no_grad():
        actual = layer(x)
    assert torch.equal(actual, expected), \
        "LUT index 1 should map to shift_lut[1]=4 and produce op(a, rotr(b, 4))."


@needs_shifted_word_logic
def test_lut_assertion_first_entry_must_be_zero():
    """The shift=0 ⇔ WordLogicLayer parity invariant requires LUT index 0 to
    map to shift=0. Constructing with a non-zero first entry must raise so
    the invariant is enforceable downstream."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    with pytest.raises(ValueError, match="shift_lut\\[0\\] must be 0"):
        ShiftedWordLogicLayer(in_dim=8, out_dim=4, M=8, device="cpu",
                              shift_lut=(1, 2, 4))


@needs_shifted_word_logic
def test_lut_assertion_entries_in_range():
    """LUT entries must be in [0, M)."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    # Out of range (M=8 means valid shifts are 0..7).
    with pytest.raises(ValueError, match="must be in"):
        ShiftedWordLogicLayer(in_dim=8, out_dim=4, M=8, device="cpu",
                              shift_lut=(0, 8))


@needs_shifted_word_logic
def test_lut_shift_weights_shape_and_K_attribute():
    """shift_weights and the K attribute reflect the LUT length, not M."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    lut = (0, 1, 2, 4, 8, 16, 32, 64, 127)
    layer = ShiftedWordLogicLayer(in_dim=64, out_dim=128, M=128, device="cpu",
                                  shift_lut=lut).eval()
    assert layer.K == len(lut)
    assert layer.shift_weights.shape == (128, len(lut))
    assert layer._shift_gather_idx.shape == (len(lut), 128)
    # Without a LUT the shapes scale with M.
    layer_full = ShiftedWordLogicLayer(in_dim=64, out_dim=128, M=128, device="cpu").eval()
    assert layer_full.K == 128
    assert layer_full.shift_weights.shape == (128, 128)


@needs_shifted_word_logic
@pytest.mark.parametrize("op_idx,shift", [(1, 0), (6, 1), (7, 2), (9, 4)])
def test_shifted_layer_with_lut_matches_reference(op_idx, shift):
    """Same hand-verified op×shift reference as the M-alphabet test, but with
    the LUT path. The LUT contains the tested shift in the right index."""
    from src.modules.shifted_word_logic import ShiftedWordLogicLayer

    M = 8
    lut = (0, 1, 2, 4)  # contains the tested shifts (0, 1, 2, 4)
    layer = ShiftedWordLogicLayer(in_dim=2, out_dim=1, M=M, device="cpu",
                                  shift_lut=lut).eval()
    with torch.no_grad():
        layer.indices_a.zero_()
        layer.indices_b.fill_(1)
        layer.weights.zero_()
        layer.weights[:, op_idx] = 10.0
        layer.shift_weights.zero_()
        layer.shift_weights[:, lut.index(shift)] = 10.0

    torch.manual_seed(0)
    x = (torch.rand(4, 2, M) > 0.5).float()
    a = x[:, 0, :]
    b = x[:, 1, :]
    expected = _reference_binary_op(a, torch.roll(b, shifts=shift, dims=-1), op_idx).unsqueeze(1)

    with torch.no_grad():
        actual = layer(x)
    assert torch.equal(actual, expected), \
        f"LUT path: op={op_idx} shift={shift} (LUT index {lut.index(shift)}): mismatch"
