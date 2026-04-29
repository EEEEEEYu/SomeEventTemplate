"""Streaming classifier — proposal v2 §Stage 4 architecture.

```
TBR slice [B, 2, H, W]
    │
    ▼
ENCODER (logic-gate): per-slice WordLogicLayer stack
    │  emits f_t ∈ {0, 1}^M per slice
    ▼
BUFFER [B, N, M] shift register  (proposal v2 §Stage 4 task 1)
    │  row 0 = newest f_t (grad-attached); rows 1..N-1 detached/attached
    │  per `tbptt_k` (proposal v2.1 Tier 1/2/3)
    ▼
CROSS-SLICE LAYER: pluggable family from `cross_slice_ops` registry
    │  default `difflogic16` (each output bit picks i ∈ [0, N) and
    │  one of 16 binary ops, paired with row 0)
    ▼
DECODER: stack of WordLogicLayer
    │
    ▼
GroupSum -> logits
```

The class composes Buffer + CrossSliceOpFamily + WordLogicLayer + GroupSum
modules, none of which it owns by inheritance. Future variations (different
buffer policies, alternative cross-slice families, encoder/decoder depth
sweeps) come from swapping arguments — no model rewrite required.

Per-step training loop (proposal §Stage 4 task 3):

    buffer.reset(B, device, dtype)
    loss = 0
    for t in range(T):
        f_t = encoder(x[:, t])              # grad-attached
        B_buf = buffer.step(f_t)            # row 0 attached; older rows
                                            # follow `tbptt_k` policy
        z = cross_slice(B_buf)              # [B, hidden, M]
        z = decoder(z)
        logits_t = readout(z)
        if t >= warmup_steps:
            loss += CE(logits_t, y)
    loss /= max(T - warmup_steps, 1)

`warmup_steps` defaults to `N` (don't backprop until the buffer is
populated). `loss_at_every_step=False` reduces this to "loss at the final
step only" — useful for ablations.

`N=1` collapses the architecture to a per-frame model: the buffer is a
single row, the cross-slice family's slice softmax has one degenerate
choice, and the layer is functionally a `WordLogicLayer` reading row 0.
This is the parity anchor against Stage 1.
"""

from __future__ import annotations

from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import multiclass_accuracy

from src.modules.buffer import FeatureBuffer
from src.modules.cross_slice_ops import get_family_class
from src.modules.logic_blocks import GroupSum
from src.modules.word_logic import WordLogicLayer
from src.utils.diagnostics import freeze_decoder_for_warmup


class StreamingClassifier(pl.LightningModule):
    """Per-slice encoder + buffer + cross-slice op + word-logic decoder + GroupSum."""

    def __init__(
        self,
        # input shape (per slice)
        in_features: int,                  # 2 * H * W per slice
        M: int = 32,                       # bits per slice / encoder output width
        # encoder
        encoder_hidden_dim: int = 4070,
        encoder_num_layers: int = 1,       # WordLogicLayers in the encoder
        # buffer
        N: int = 32,                       # slice history depth
        tbptt_k: Optional[int] = None,     # None = full BPTT (Tier 1)
        # cross-slice + decoder
        cross_slice_family: str = "difflogic16",
        decoder_hidden_dim: int = 4070,
        decoder_num_layers: int = 3,       # WordLogicLayers AFTER cross-slice
        num_classes: int = 11,
        tau: float = 500.0,                # match Stage 1 / Tier 0's softmax-temperature regime
        # training
        lr: float = 0.01,
        grad_factor: float = 1.0,
        connections: str = "random",
        warmup_steps: Optional[int] = None,  # None = N (don't loss-accum until buffer full)
        loss_at_every_step: bool = True,
        encoder_warmup_epochs: int = 0,    # P2 hook
        # plumbing
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if (decoder_hidden_dim * M) % num_classes != 0:
            raise ValueError(
                f"decoder_hidden_dim * M ({decoder_hidden_dim * M}) must be divisible "
                f"by num_classes ({num_classes}); GroupSum splits into k equal groups. "
                f"Pick decoder_hidden_dim such that decoder_hidden_dim * M %% num_classes == 0."
            )

        # ------------------------------------------------------------------
        # encoder: stack of WordLogicLayer(M=1) — i.e. a difflogic-style
        # scalar logic-gate net. Each of the `M` output neurons emits one
        # bit of the slice's M-bit feature word.
        #
        # Input to layer 0: per-slice TBR flattened to `[B, 2*H*W, 1]`.
        # Output of last layer: `[B, M, 1]` → squeeze last axis → `[B, M]`,
        # the buffer's row-0 write target.
        #
        # Intermediate hidden width is `encoder_hidden_dim` (interior layers
        # have in_dim=out_dim=encoder_hidden_dim); the last encoder layer
        # is a "head" that projects from `encoder_hidden_dim` to `M`.
        # ------------------------------------------------------------------
        encoder_layers: List[nn.Module] = []
        prev = in_features
        for li in range(encoder_num_layers):
            out = M if li == encoder_num_layers - 1 else encoder_hidden_dim
            encoder_layers.append(WordLogicLayer(
                in_dim=prev, out_dim=out, M=1,
                grad_factor=grad_factor, connections=connections, device=device,
            ))
            prev = out
        self.encoder = nn.Sequential(*encoder_layers)

        # ------------------------------------------------------------------
        # buffer
        # ------------------------------------------------------------------
        self.buffer = FeatureBuffer(N=N, M=M, tbptt_k=tbptt_k)

        # ------------------------------------------------------------------
        # cross-slice layer (the streaming-architecture-specific piece)
        # ------------------------------------------------------------------
        family_cls = get_family_class(cross_slice_family)
        self.cross_slice = family_cls(
            N=N, M=M, out_dim=decoder_hidden_dim,
            connections=connections, device=device,
        )

        # ------------------------------------------------------------------
        # decoder: WordLogicLayer stack after the cross-slice op
        # ------------------------------------------------------------------
        decoder_layers: List[nn.Module] = []
        prev = decoder_hidden_dim
        for _ in range(decoder_num_layers):
            decoder_layers.append(WordLogicLayer(
                in_dim=prev, out_dim=decoder_hidden_dim, M=M,
                grad_factor=grad_factor, connections=connections, device=device,
            ))
            prev = decoder_hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)

        self.readout = GroupSum(k=num_classes, tau=tau)
        self.loss_fn = nn.CrossEntropyLoss()

        # P2 layer-groups for diagnostics + warmup. Encoder = encoder layers;
        # decoder = cross_slice + word decoder; readout separate.
        encoder_prefixes = [f"encoder.{i}." for i in range(encoder_num_layers)]
        decoder_prefixes = ["cross_slice."] + [f"decoder.{i}." for i in range(decoder_num_layers)]
        self.layer_groups = {
            "encoder": encoder_prefixes,
            "decoder": decoder_prefixes,
            "readout": ["readout."],
        }
        self.encoder_warmup_epochs = encoder_warmup_epochs

        # config knobs the training loop reads
        self._N = N
        self._warmup_steps = warmup_steps if warmup_steps is not None else N
        self._loss_at_every_step = loss_at_every_step

    # ----------------------------------------------------------------------
    # input-shaping helpers

    @staticmethod
    def _slice_to_word_input(x_slice: torch.Tensor, M: int) -> torch.Tensor:
        """Reshape a per-slice TBR `[B, 2, H, W]` boolean tensor into the
        `[B, in_features, 1]` shape the M=1 encoder wants. The encoder is a
        scalar (M=1) logic-gate stack; the cross-slice / decoder layers above
        the buffer are M-wide. `M` is unused here, kept for API symmetry."""
        del M  # unused; kept to mirror the call site's intent
        if x_slice.dim() != 4:
            raise ValueError(f"expected [B, 2, H, W]; got {tuple(x_slice.shape)}")
        b, p, h, w = x_slice.shape
        return x_slice.reshape(b, p * h * w, 1)

    # ----------------------------------------------------------------------
    # one-step forward (used by both training and eval)

    def forward_step(self, x_slice: torch.Tensor) -> torch.Tensor:
        """One per-slice forward: encoder → buffer.step → cross-slice → decoder → readout.

        Args:
            x_slice: `[B, 2, H, W]` float in {0, 1} (post-`.round()` if in eval).
        Returns:
            logits: `[B, num_classes]`.
        """
        x = self._slice_to_word_input(x_slice, self.hparams.M)        # [B, in_features, 1]
        f = self.encoder(x)                                            # [B, M, 1]
        f_t = f.squeeze(-1)                                            # [B, M]
        buf = self.buffer.step(f_t)                                    # [B, N, M]
        z = self.cross_slice(buf)                                      # [B, decoder_hidden_dim, M]
        z = self.decoder(z)                                            # [B, decoder_hidden_dim, M]
        b = z.shape[0]
        z_flat = z.reshape(b, -1)                                      # [B, decoder_hidden_dim * M]
        return self.readout(z_flat)                                    # [B, num_classes]

    # ----------------------------------------------------------------------
    # streaming step over T slices

    def _streaming_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch                                                   # x: [B, T, 2, H, W]
        if stage != "train":
            x = x.round()
        if x.dim() != 5:
            raise ValueError(f"expected [B, T, 2, H, W]; got {tuple(x.shape)}")
        B, T = x.shape[0], x.shape[1]

        self.buffer.reset(batch_size=B, device=x.device, dtype=x.dtype)
        warmup = min(self._warmup_steps, T)

        loss_acc: Optional[torch.Tensor] = None
        loss_count = 0
        last_logits: Optional[torch.Tensor] = None

        if self._loss_at_every_step:
            for t in range(T):
                logits_t = self.forward_step(x[:, t])
                last_logits = logits_t
                if t >= warmup:
                    li = self.loss_fn(logits_t, y)
                    loss_acc = li if loss_acc is None else loss_acc + li
                    loss_count += 1
        else:
            # Run T steps; only the final step contributes to loss.
            for t in range(T):
                logits_t = self.forward_step(x[:, t])
                last_logits = logits_t
            loss_acc = self.loss_fn(last_logits, y)
            loss_count = 1

        loss = loss_acc / max(loss_count, 1)
        assert last_logits is not None
        acc = multiclass_accuracy(
            last_logits, y, num_classes=self.hparams.num_classes,
            average="micro", top_k=1,
        )
        bs = x.shape[0]
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        return loss

    # ----------------------------------------------------------------------
    # Lightning hooks

    def on_train_epoch_start(self):
        # Encoder warm-up (proposal v2.1 §Always-on additions task 2).
        freeze_decoder_for_warmup(self, current_epoch=self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self._streaming_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._streaming_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._streaming_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ----------------------------------------------------------------------
    # diagnostics

    def slice_choice_distribution(self) -> torch.Tensor:
        """Histogram of `argmax(idx_logits)` across the cross-slice layer's
        neurons — answers "do downstream neurons read from a *variety* of
        slice rows, or collapse to row 0?"
        """
        if not hasattr(self.cross_slice, "idx_logits"):
            raise RuntimeError(
                f"cross-slice family {self.hparams.cross_slice_family!r} has no "
                f"`idx_logits` parameter; can't extract a slice histogram."
            )
        return torch.bincount(
            self.cross_slice.idx_logits.argmax(-1).cpu(),
            minlength=self.cross_slice.N,
        )
