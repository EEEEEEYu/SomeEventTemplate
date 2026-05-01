"""Thermometer encoding correctness (CDLGN, Appendix A.1.1).

Anchored to the difflogic reference's threshold convention:
    bit_i(x) = 1 iff x > (i+1)/(n_bits+1)
"""

import torch

from src.data.thermometer import thermometer_encode


def test_shape_3channel_3bits():
    x = torch.zeros(3, 32, 32)
    out = thermometer_encode(x, n_bits=3)
    assert out.shape == (9, 32, 32)
    assert out.dtype == torch.bool


def test_shape_3channel_31bits():
    x = torch.zeros(3, 32, 32)
    out = thermometer_encode(x, n_bits=31)
    assert out.shape == (93, 32, 32)


def test_batched_input():
    x = torch.zeros(8, 3, 32, 32)
    out = thermometer_encode(x, n_bits=3)
    assert out.shape == (8, 9, 32, 32)


def test_zero_input_all_zeros():
    """A pixel at exactly 0 should produce all-zero thermometer bits (since
    0 > (i+1)/(n+1) is false for every i)."""
    x = torch.zeros(3, 4, 4)
    out = thermometer_encode(x, n_bits=3)
    assert not out.any()


def test_one_input_all_ones():
    """A pixel at exactly 1 should produce all-one thermometer bits (since
    1 > (i+1)/(n+1) is true for every i in [0..n-1])."""
    x = torch.ones(3, 4, 4)
    out = thermometer_encode(x, n_bits=3)
    assert out.all()


def test_threshold_values_3bits():
    """With n_bits=3 thresholds are 0.25, 0.5, 0.75. Build a single-pixel
    image with controlled values and check each bit independently."""
    # Stack values along channel axis so we can read off each independently.
    x = torch.tensor([[[0.10]], [[0.30]], [[0.60]]])           # (3, 1, 1)
    out = thermometer_encode(x, n_bits=3)                      # (9, 1, 1)
    out = out.squeeze(-1).squeeze(-1)                          # (9,)
    # Channel 0 (value 0.10): all bits 0 (0.10 > 0.25? no)
    assert out[0:3].tolist() == [False, False, False]
    # Channel 1 (value 0.30): bit 0 only (0.30 > 0.25 yes; > 0.5 no)
    assert out[3:6].tolist() == [True, False, False]
    # Channel 2 (value 0.60): bits 0,1 (0.60 > 0.25 yes; > 0.5 yes; > 0.75 no)
    assert out[6:9].tolist() == [True, True, False]


def test_matches_difflogic_reference_3bits():
    """Bit-for-bit match against the difflogic reference's lambda for
    `cifar-10-3-thresholds`: torch.cat([(x > (i+1)/4).float() for i in range(3)], dim=0).

    Note: the reference concatenates *along the existing channel axis* with
    `dim=0`, which produces grouping `[ch0_bit0, ch1_bit0, ch2_bit0,
    ch0_bit1, ...]`. Our function groups `[ch0_bit0, ch0_bit1, ch0_bit2,
    ch1_bit0, ...]` — same total content, different interleave. Compare as
    sets of (bit, channel) per pixel to verify equivalence.
    """
    torch.manual_seed(0)
    x = torch.rand(3, 8, 8)
    ours = thermometer_encode(x, n_bits=3)                         # (9, 8, 8) ours
    ref = torch.cat([(x > (i + 1) / 4) for i in range(3)], dim=0)  # (9, 8, 8) ref

    # Both encodings have the same multiset of (bit_i, channel_c) planes.
    # Sort each as a (9, 8*8) tensor row-wise and compare.
    ours_planes = ours.reshape(9, -1).bool()
    ref_planes = ref.reshape(9, -1).bool()
    # Convert each plane to a tuple-key for set comparison
    ours_keys = sorted(tuple(p.tolist()) for p in ours_planes)
    ref_keys = sorted(tuple(p.tolist()) for p in ref_planes)
    assert ours_keys == ref_keys


def test_threshold_values_31bits():
    """With n_bits=31 thresholds are (i+1)/32 for i in 0..30. Test boundaries."""
    x = torch.tensor([[[0.5]]])                                # (1, 1, 1)
    out = thermometer_encode(x, n_bits=31)                     # (31, 1, 1)
    out = out.squeeze(-1).squeeze(-1)                          # (31,)
    # 0.5 > (i+1)/32 holds iff i+1 < 16, i.e. i in 0..14 → 15 ones
    assert out[:15].all()
    assert not out[15:].any()
