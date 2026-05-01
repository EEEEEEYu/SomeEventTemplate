"""LightningModule for event-camera optical-flow estimation.

**Phase-2 stub** per proposal v3 §Phase 2. Populated once MVSEC (or DSEC-Flow)
is uploaded and the input encoding is decided (TBR vs voxel grid vs event
count image).

Design (from proposal):
    - Hybrid Boolean-then-float: a torchlogix logic-conv backbone produces
      Boolean spatial features densely per-pixel; a small float head converts
      each pixel's feature vector to a 2D (u, v) flow vector.
    - Loss: L1 or L2 on flow vectors (with valid-pixel masking from the GT
      flow's mask channel).
    - Metric: End-Point Error (EPE) — `mean(sqrt((u_pred - u_gt)^2 +
      (v_pred - v_gt)^2))` over valid pixels.

This file currently raises if instantiated. Implementation plan when data
arrives is in `STATUS.md` ("Phase 2 readiness")."""

from __future__ import annotations

import lightning.pytorch as pl


class TorchlogixFlow(pl.LightningModule):
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "TorchlogixFlow is a Phase-2 stub. Populate when MVSEC data is "
            "uploaded; see STATUS.md → 'Phase 2 readiness'."
        )
