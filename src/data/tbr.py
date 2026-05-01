"""Time-Binned Representation (TBR) encoder for event-camera streams.

Spec (proposal §Stage 2 task 2):
  - Output: [P, num_bins, H, W] boolean tensor.
  - Each cell is True iff at least one event of polarity p at pixel (y, x) falls
    in time bin ⌊(t - t0) / bin_duration_us⌋ ∈ [0, num_bins).
  - Vectorized: no Python loop over events.

Input convention: a 2D float/int tensor of shape [N, 4] with columns (x, y, t, p),
  - x ∈ [0, W), y ∈ [0, H)
  - t in microseconds (any monotonic int unit works as long as bin_duration_us
    is in the same unit)
  - p ∈ [0, P) — callers using tonic's {-1, +1} polarity convention must remap
    to {0, 1} before calling this function.

This minimal API keeps the encoder callable from unit tests without depending
on tonic. The DataModules adapt tonic structured arrays into [N, 4] tensors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def encode_tbr(
    events: torch.Tensor,
    num_bins: int = 32,
    bin_duration_us: int = 1000,
    sensor_size: Tuple[int, int, int] = (34, 34, 2),
    t0: Optional[int] = None,
) -> torch.Tensor:
    if events.dim() != 2 or events.shape[1] != 4:
        raise ValueError(f"events must have shape [N, 4]; got {tuple(events.shape)}")
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive; got {num_bins}")
    if bin_duration_us <= 0:
        raise ValueError(f"bin_duration_us must be positive; got {bin_duration_us}")

    H, W, P = sensor_size
    device = events.device
    out = torch.zeros(P, num_bins, H, W, dtype=torch.bool, device=device)
    if events.shape[0] == 0:
        return out

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2].long()
    p = events[:, 3].long()

    if t0 is None:
        t0_val = int(t.min().item())
    else:
        t0_val = int(t0)

    bin_idx = (t - t0_val) // int(bin_duration_us)
    mask = (
        (bin_idx >= 0) & (bin_idx < num_bins)
        & (x >= 0) & (x < W)
        & (y >= 0) & (y < H)
        & (p >= 0) & (p < P)
    )
    if not bool(mask.any()):
        return out

    x = x[mask]; y = y[mask]; bin_idx = bin_idx[mask]; p = p[mask]
    out[p, bin_idx, y, x] = True
    return out
