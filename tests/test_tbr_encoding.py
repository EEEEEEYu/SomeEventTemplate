"""Unit tests for src/data/tbr.py — proposal §Stage 2 task 2 acceptance test."""

from __future__ import annotations

import torch

from src.data.tbr import encode_tbr


def _ev(x, y, t, p):
    """Build an events tensor of shape [N, 4] from parallel lists."""
    return torch.tensor(list(zip(x, y, t, p)), dtype=torch.long)


def test_empty_events_returns_all_zero():
    events = torch.zeros((0, 4), dtype=torch.long)
    out = encode_tbr(events, num_bins=8, bin_duration_us=1000, sensor_size=(4, 4, 2))
    assert out.shape == (2, 8, 4, 4)
    assert out.dtype == torch.bool
    assert not out.any()


def test_single_event_lands_in_expected_cell():
    # One event: (x=2, y=1, t=2500us, p=1), t0=0, bin_duration=1000us
    # → bin_idx = 2500 // 1000 = 2
    events = _ev([2], [1], [2500], [1])
    out = encode_tbr(events, num_bins=4, bin_duration_us=1000, sensor_size=(4, 4, 2), t0=0)
    expected = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    expected[1, 2, 1, 2] = True
    assert torch.equal(out, expected)


def test_multiple_events_same_bin_collapse_to_one():
    events = _ev([0, 0, 0], [0, 0, 0], [0, 500, 999], [0, 0, 0])
    out = encode_tbr(events, num_bins=4, bin_duration_us=1000, sensor_size=(4, 4, 2), t0=0)
    # All three fall in bin 0, polarity 0, pixel (0, 0); should collapse to a single True.
    assert out[0, 0, 0, 0].item() is True
    assert out.sum().item() == 1


def test_events_after_window_are_ignored():
    # bin_duration_us=1000, num_bins=4 → window covers t ∈ [0, 4000)
    events = _ev([0, 0], [0, 0], [3999, 4000], [0, 0])
    out = encode_tbr(events, num_bins=4, bin_duration_us=1000, sensor_size=(4, 4, 2), t0=0)
    assert out[0, 3, 0, 0].item() is True   # t=3999 → bin 3
    assert out.sum().item() == 1            # t=4000 dropped


def test_events_before_t0_are_ignored():
    events = _ev([0, 0], [0, 0], [-100, 0], [0, 0])
    out = encode_tbr(events, num_bins=4, bin_duration_us=1000, sensor_size=(4, 4, 2), t0=0)
    assert out[0, 0, 0, 0].item() is True   # t=0
    assert out.sum().item() == 1


def test_out_of_bounds_pixels_are_ignored():
    H, W, P = 4, 4, 2
    events = _ev([5, 0, -1], [0, 5, 0], [0, 0, 0], [0, 0, 0])
    out = encode_tbr(events, num_bins=4, bin_duration_us=1000, sensor_size=(H, W, P), t0=0)
    assert not out.any()


def test_polarity_separation():
    events = _ev([1, 1], [2, 2], [0, 0], [0, 1])
    out = encode_tbr(events, num_bins=4, bin_duration_us=1000, sensor_size=(4, 4, 2), t0=0)
    assert out[0, 0, 2, 1].item() is True
    assert out[1, 0, 2, 1].item() is True
    assert out.sum().item() == 2


def test_t0_inferred_when_omitted_uses_min_t():
    # If t0 is None we should snap the earliest event to bin 0.
    events = _ev([0, 1], [0, 1], [5000, 5500], [0, 1])
    out = encode_tbr(events, num_bins=4, bin_duration_us=1000, sensor_size=(4, 4, 2))
    # Earliest event at t=5000 sets t0 → bin 0; second at t=5500 → still bin 0.
    assert out[0, 0, 0, 0].item() is True
    assert out[1, 0, 1, 1].item() is True
    assert out.sum().item() == 2


def test_full_coverage_one_event_per_bin():
    # Place one event in each of num_bins=8 bins at distinct pixels.
    num_bins = 8
    bin_duration = 1000
    xs = list(range(num_bins))
    ys = [0] * num_bins
    ts = [b * bin_duration + 17 for b in range(num_bins)]   # 17us into each bin
    ps = [b % 2 for b in range(num_bins)]
    events = _ev(xs, ys, ts, ps)
    out = encode_tbr(events, num_bins=num_bins, bin_duration_us=bin_duration,
                     sensor_size=(4, num_bins, 2), t0=0)
    assert out.sum().item() == num_bins
    for b in range(num_bins):
        assert out[ps[b], b, ys[b], xs[b]].item() is True


def test_input_shape_validation():
    bad = torch.zeros((3, 5), dtype=torch.long)
    try:
        encode_tbr(bad, sensor_size=(4, 4, 2))
    except ValueError:
        return
    raise AssertionError("expected ValueError for [N, 5] input")
