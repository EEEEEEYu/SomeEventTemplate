"""Thermometer encoding for CIFAR-10 (Petersen et al. 2024, Appendix A.1.1).

Mirrors the threshold convention `(i+1)/(n_bits+1)` used by the published
difflogic reference (`difflogic/experiments/main.py`, see
`cifar-10-3-thresholds` and `cifar-10-31-thresholds`):

    bit_i(x) = 1 iff x > (i+1) / (n_bits + 1)        for i in {0, .., n_bits-1}

Boundary behaviour: a pixel exactly at 0 yields all zeros; a pixel at 1 yields
all ones. This matches the difflogic reference and is what the CDLGN gate-count
totals assume (encoding gates are counted in the published numbers per paper §6).
"""

from __future__ import annotations

import torch


def thermometer_encode(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Per-channel thermometer encoding.

    Args:
        x:        shape `(C, H, W)` or `(B, C, H, W)`, values in `[0, 1]`.
        n_bits:   thresholds per channel. CDLGN uses 3 (S/M) or 31 (B/L/G).

    Returns:
        Boolean tensor with `C` replaced by `n_bits * C`. The bits for a single
        channel are contiguous in the channel axis: `[ch0_bit0, ch0_bit1, ...,
        ch0_bit{n-1}, ch1_bit0, ...]`. This grouping matches the difflogic
        reference's `torch.cat([... for i in range(n_bits)], dim=channel)`,
        because we apply the same outer loop and concatenation order.
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1, got {n_bits}")
    if x.dim() not in (3, 4):
        raise ValueError(f"x must be (C,H,W) or (B,C,H,W); got shape {tuple(x.shape)}")
    thresholds = torch.tensor(
        [(i + 1) / (n_bits + 1) for i in range(n_bits)],
        dtype=x.dtype,
        device=x.device,
    )
    # Broadcast thresholds against channel axis so each channel produces n_bits planes.
    # x: (..., C, H, W) → (..., C, 1, H, W) > thresholds[None, :, None, None] yields
    # (..., C, n_bits, H, W); reshape collapses C and n_bits into the channel axis.
    expanded = x.unsqueeze(-3)                       # (..., C, 1,      H, W)
    th = thresholds.view(1, n_bits, 1, 1)            # (   1, n_bits, 1, 1)
    bits = expanded > th                             # (..., C, n_bits, H, W)
    *batch, c, n, h, w = bits.shape
    return bits.reshape(*batch, c * n, h, w)
