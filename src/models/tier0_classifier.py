"""Tier 0 LightningModule (Phase P §P4d / proposal v2.1 §Tier 0).

The decisive experiment for the streaming pivot. Single fused TBR tensor per
sample, M=128 word width, cross-bit shifts in the first decoder layer for
within-word temporal coupling, full gradient flow, no buffer.

If this clears ≥80% on DVS-Gesture with shifts contributing meaningfully
(P(shift ≠ 0) ≥ 0.3 in the shifted layer's argmax distribution), Stages 3–4
collapse to follow-up work. Otherwise resume Stage 2 with the always-on
diagnostic infrastructure enabled.

Architecture:
    [B, 2, num_bins, H, W]
       │  reshape: treat (polarity, y, x) as features, num_bins as the M-axis
       ▼
    [B, in_features, M]                       in_features = 2 * H * W
       │  encoder_layers × WordLogicLayer (M=128)
       ▼
    [B, hidden_dim, M]
       │  ShiftedWordLogicLayer (M=128)       # the Tier 0 expressivity hook
       ▼
    [B, hidden_dim, M]
       │  decoder_layers × WordLogicLayer (M=128)
       ▼
    [B, hidden_dim, M]
       │  flatten last two dims
       ▼
    [B, hidden_dim * M]
       │  GroupSum(k=num_classes, tau)
       ▼
    [B, num_classes]

`hidden_dim * M` must be divisible by num_classes — validated in __init__.

Training: full BPTT, lr=0.01 (same as Stage 1's discovered necessity for
difflogic's relaxation). Gradient-flow logging via the P2 callback is
config-gated; turn on via `DIAGNOSTICS.gradient_norm_logger=true`.

Discretization: standard difflogic — `model.eval()` flips every
`{Word,ShiftedWord}LogicLayer` into the one-hot argmax branch.
"""

from __future__ import annotations

from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import multiclass_accuracy

from src.modules.logic_blocks import GroupSum
from src.modules.word_logic import WordLogicLayer
from src.modules.shifted_word_logic import ShiftedWordLogicLayer
from src.utils.diagnostics import freeze_decoder_for_warmup


class Tier0Classifier(pl.LightningModule):
    """Wide-word LGN classifier with one cross-bit-shift layer."""

    def __init__(
        self,
        in_features: int,                  # 2 * H * W (e.g., 2048 for 32×32)
        M: int = 128,
        hidden_dim: int = 4000,
        num_encoder_layers: int = 1,       # WordLogicLayers before the shift
        num_decoder_layers: int = 3,       # WordLogicLayers after the shift
        num_classes: int = 11,
        tau: float = 10.0,
        lr: float = 0.01,
        grad_factor: float = 1.0,
        connections: str = "random",
        device: str = "cuda",              # passed through to layer constructors
        encoder_warmup_epochs: int = 0,    # P2 hook; 0 = disabled (Stage 1 default)
        shift_lut: Optional[List[int]] = None,  # see ShiftedWordLogicLayer; e.g.
                                                # [0,1,2,4,8,16,32,64,127] for ~14×
                                                # speedup on the soft-shift bottleneck
    ):
        super().__init__()
        self.save_hyperparameters()

        if (hidden_dim * M) % num_classes != 0:
            raise ValueError(
                f"hidden_dim * M ({hidden_dim * M}) must be divisible by num_classes "
                f"({num_classes}); GroupSum splits the last layer's outputs into k equal "
                f"groups. Pick hidden_dim such that hidden_dim * M % num_classes == 0."
            )

        layers: List[nn.Module] = []

        prev = in_features
        for _ in range(num_encoder_layers):
            layers.append(WordLogicLayer(in_dim=prev, out_dim=hidden_dim, M=M,
                                         grad_factor=grad_factor, connections=connections,
                                         device=device))
            prev = hidden_dim

        # The Tier 0 expressivity hook (proposal v2.1 §Tier 0).
        layers.append(ShiftedWordLogicLayer(in_dim=prev, out_dim=hidden_dim, M=M,
                                            grad_factor=grad_factor, connections=connections,
                                            device=device, shift_lut=shift_lut))
        prev = hidden_dim

        for _ in range(num_decoder_layers):
            layers.append(WordLogicLayer(in_dim=prev, out_dim=hidden_dim, M=M,
                                         grad_factor=grad_factor, connections=connections,
                                         device=device))
            prev = hidden_dim

        self.body = nn.Sequential(*layers)
        self.readout = GroupSum(k=num_classes, tau=tau)
        self.loss_fn = nn.CrossEntropyLoss()
        self.encoder_warmup_epochs = encoder_warmup_epochs

        # Layer groups for diagnostics + warm-up. The shift layer is grouped
        # with the decoder per proposal v2.1 §"Always-on additions"; encoder
        # = the WordLogicLayers preceding the shift, decoder = the shift layer
        # plus the WordLogicLayers following.
        encoder_prefixes = [f"body.{i}." for i in range(num_encoder_layers)]
        decoder_start = num_encoder_layers
        decoder_count = 1 + num_decoder_layers  # shift + post layers
        decoder_prefixes = [f"body.{decoder_start + i}." for i in range(decoder_count)]
        self.layer_groups = {
            "encoder": encoder_prefixes,
            "decoder": decoder_prefixes,
            "readout": ["readout."],
        }

    # ----------------------------------------------------------------------
    # forward / step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, num_bins=M, H, W] from the DataModule. Reshape so M is the
        # last (word) axis and (polarity, y, x) is the in_features axis.
        # [B, 2, M, H, W] -> permute -> [B, 2, H, W, M] -> reshape -> [B, 2*H*W, M]
        if x.dim() != 5:
            raise ValueError(f"Tier0Classifier expects [B, 2, M, H, W] input; got {tuple(x.shape)}")
        b, polarities, m, h, w = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous().reshape(b, polarities * h * w, m)
        if x.shape[1] != self.hparams.in_features:
            raise ValueError(
                f"flattened in_features {x.shape[1]} != configured {self.hparams.in_features}"
            )
        z = self.body(x)                          # [B, hidden_dim, M]
        z_flat = z.reshape(b, -1)                 # [B, hidden_dim * M]
        return self.readout(z_flat)               # [B, num_classes]

    def _shared_step(self, batch, stage: str):
        x, y = batch
        if stage != "train":
            x = x.round()  # discretized eval per Petersen's reference
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = multiclass_accuracy(logits, y, num_classes=self.hparams.num_classes,
                                  average="micro", top_k=1)
        bs = x.shape[0]
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        return loss

    def on_train_epoch_start(self):
        # Encoder warm-up (proposal v2.1 §"Always-on additions" task 2). For
        # the first `encoder_warmup_epochs` epochs only the encoder + readout
        # train; the shifted layer + decoder word layers are frozen. Lets the
        # encoder establish reasonable representations before the heavier
        # decoder grad signal dominates optimization. Layer groups are set in
        # __init__ — encoder = body[0..N_enc), decoder = body[N_enc..].
        freeze_decoder_for_warmup(self, current_epoch=self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ----------------------------------------------------------------------
    # diagnostics

    def shift_distribution(self) -> torch.Tensor:
        """Returns the histogram of `argmax(shift_weights)` across the shifted
        layer's neurons. Used for the operator-choice diagnostic in Phase P
        §P4d task 4 — if >90% of neurons collapse to LUT index 0 (≡ shift=0),
        cross-bit shifts aren't contributing. Length is `K` (= `M` for the
        full alphabet, = `len(shift_lut)` for the reduced LUT).
        """
        for module in self.body:
            if isinstance(module, ShiftedWordLogicLayer):
                hist = torch.bincount(module.shift_weights.argmax(-1).cpu(),
                                      minlength=module.K)
                return hist
        raise RuntimeError("no ShiftedWordLogicLayer found in body")
