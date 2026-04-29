"""Feature buffer (proposal v2 §Stage 4 task 1 / proposal v2.1 §Tier 2).

The `[N, M]` shift register that holds the most recent `N` per-slice feature
words emitted by the streaming encoder. Each `step(f_t)` call shifts the
buffer down by one row and writes `f_t` into row 0 (the newest slot).

`tbptt_k` controls the gradient depth through time, per proposal v2.1:

  - `tbptt_k = None`         → **full BPTT** (Tier 1) — no detachment, all N
                                 rows carry gradient.
  - `tbptt_k = k` (k < N)    → **truncated BPTT** (Tier 2) — gradient flows
                                 through the most recent `k` rows; rows
                                 `[k:N]` are detached on each step.
  - `tbptt_k = 1`            → v2-original "detach all but row 0" — kept as
                                 an ablation knob; not the default.

The class is deliberately small. Future variations (gated memory, attention
over slices, exponential-moving-average summaries, ring-buffer with valid-
mask for variable T) should subclass or compose `FeatureBuffer` rather than
flag-bloat the constructor.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FeatureBuffer(nn.Module):
    """`[B, N, M]` shift register with config-tunable gradient depth.

    Owns a transient state (`self._state`) that lives across forward calls
    within one training/validation example and is reset between examples
    via `reset(...)`.

    State is *not* a `nn.Parameter` — it doesn't persist across optimizer
    steps and isn't checkpointed. It is stored as a plain attribute so it
    automatically follows the LightningModule's device when the parent
    moves with `.to(device)` (the parent's `_apply` walks attributes for
    `Tensor` instances).
    """

    def __init__(self, N: int, M: int, tbptt_k: Optional[int] = None) -> None:
        super().__init__()
        if N < 1:
            raise ValueError(f"N must be >= 1; got {N}")
        if M < 1:
            raise ValueError(f"M must be >= 1; got {M}")
        if tbptt_k is not None and tbptt_k < 1:
            raise ValueError(f"tbptt_k must be None or >= 1; got {tbptt_k}")

        self.N = N
        self.M = M
        self.tbptt_k = tbptt_k
        self._state: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # state management

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """Zero the buffer at the start of a new sample. Called once per
        sample by the streaming model's training_step / validation_step."""
        self._state = torch.zeros(batch_size, self.N, self.M, device=device, dtype=dtype)

    @property
    def state(self) -> torch.Tensor:
        """Read-only view of the current buffer (`[B, N, M]`).

        Raises if `reset()` hasn't been called — buffers are stateful and
        need explicit initialization per sample.
        """
        if self._state is None:
            raise RuntimeError("FeatureBuffer.reset(...) must be called before state access")
        return self._state

    # ------------------------------------------------------------------
    # core update

    def _apply_tbptt_detach(self, buffer: torch.Tensor) -> torch.Tensor:
        """Apply the tbptt_k detachment policy *to the rolled buffer before
        row 0 is overwritten*. Rows that are detached lose their grad path
        on this step — encoder gradients still flow through row 0, but only
        the most recent `tbptt_k` rows carry decoder grads onward.

        - `tbptt_k = None`: no detachment (Tier 1 full BPTT)
        - `tbptt_k = N`: equivalent to `None` (every row carries grad)
        - `tbptt_k = k < N`: rows `[k:N]` detached
        - `tbptt_k = 1`: only row 0 will eventually carry grad (v2 original)
        """
        if self.tbptt_k is None or self.tbptt_k >= self.N:
            return buffer
        # Rows [tbptt_k:N] become grad-detached. Use a single index expression
        # to avoid breaking autograd on the kept rows.
        kept = buffer[:, : self.tbptt_k, :]
        detached = buffer[:, self.tbptt_k :, :].detach()
        return torch.cat([kept, detached], dim=1)

    def step(self, f_t: torch.Tensor) -> torch.Tensor:
        """Shift the buffer down by one row, write `f_t` into row 0, return
        the new buffer.

        Args:
            f_t: `[B, M]` — the encoder output for slice `t`. Carries gradient
                back to the encoder.
        Returns:
            `[B, N, M]` — the updated buffer. Row 0 is `f_t`; row 1 is the
            previous row 0; etc.
        """
        if self._state is None:
            raise RuntimeError("FeatureBuffer.reset(...) must be called before step()")
        if f_t.dim() != 2 or f_t.shape[1] != self.M:
            raise ValueError(
                f"step() expects f_t of shape [B, {self.M}]; got {tuple(f_t.shape)}"
            )
        if f_t.shape[0] != self._state.shape[0]:
            raise ValueError(
                f"f_t batch size {f_t.shape[0]} != buffer batch size {self._state.shape[0]}"
            )

        # Shift down: row i ← row i-1. Use roll then overwrite row 0 with f_t.
        # The roll itself preserves grad for all rows (it's a view); detachment
        # is applied separately so we can express tbptt_k cleanly.
        rolled = torch.roll(self._state, shifts=1, dims=1)
        rolled = self._apply_tbptt_detach(rolled)
        # In-place row-0 write would break autograd on `rolled`. Use a clone +
        # functional update instead — autograd treats this as a fresh tensor
        # whose row 0 depends on `f_t` and rows 1..N-1 depend on the rolled
        # (and possibly detached) historical values.
        new_state = rolled.clone()
        new_state[:, 0, :] = f_t
        self._state = new_state
        return self._state
