"""Stage 1 LightningModule wrapping difflogic LogicLayer + GroupSum.

Goal (proposal §Stage 1): reproduce difflogic's MNIST result within ±0.3% of
our Stage 0 baseline (97.40% → parity window 97.10–97.70%). This is a
correctness gate, not a contribution.

Notes:
- Lightning automatically calls `model.eval()` before validation / test, which
  flips `self.training=False` on every `LogicLayer` — and `LogicLayer.forward`
  reads `self.training` to dispatch to the discretized branch
  (difflogic/difflogic.py:103,111). So no manual `on_validation_epoch_start`
  hook is needed; the proposal's mention of one is defensive.
- Eval-time inputs are `.round()`-ed to {0, 1} before forward, mirroring
  Petersen's `eval()` in difflogic/experiments/main.py:189.
- LR=0.01 is a hard requirement (proposal §Stage 1 task 2): difflogic's
  relaxation needs the higher LR. The default lives here, not in the optimizer
  config, so a config typo can't silently break parity.
"""

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import multiclass_accuracy

from src.modules.logic_blocks import LogicLayer, GroupSum
from src.utils.diagnostics import freeze_decoder_for_warmup


class LogicClassifier(pl.LightningModule):
    def __init__(
        self,
        in_dim: int = 784,
        hidden_dim: int = 8000,
        num_layers: int = 6,
        num_classes: int = 10,
        tau: float = 10.0,
        connections: str = "random",
        lr: float = 0.01,
        grad_factor: float = 1.0,
        encoder_warmup_epochs: int = 0,
        aux_per_slice_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hidden_dim % num_classes != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_classes ({num_classes}); "
                "GroupSum splits the last layer's outputs into k equal groups."
            )

        layers: list[nn.Module] = [nn.Flatten()]
        layers.append(LogicLayer(in_dim, hidden_dim, connections=connections, grad_factor=grad_factor))
        for _ in range(num_layers - 1):
            layers.append(LogicLayer(hidden_dim, hidden_dim, connections=connections, grad_factor=grad_factor))
        layers.append(GroupSum(k=num_classes, tau=tau))

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()
        # Stage 1 has no meaningful encoder/decoder split — `layer_groups` is
        # absent, so `freeze_decoder_for_warmup` and `GradientNormLogger`'s
        # group resolution both fall back to a single "all" group. Stage 4's
        # streaming model will set `self.layer_groups = {"encoder": [...], ...}`.
        self.encoder_warmup_epochs = encoder_warmup_epochs

    def aux_per_slice_loss(self, encoder_features, y):
        """Hook for the v2.1 auxiliary per-slice loss (proposal §"Always-on
        additions" task 1). No-op in Stage 1 (no per-slice encoder output);
        the streaming Stage 4 model overrides this to attach a small classifier
        head to the encoder's pre-buffer output and return its CE against `y`.
        """
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        if stage != "train":
            x = x.round()                     # discretized inputs at eval, per Petersen's eval()
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
        # Encoder warm-up (proposal v2.1 §"Always-on additions" task 2).
        # No-op for Stage 1 since `layer_groups` is unset.
        freeze_decoder_for_warmup(self, current_epoch=self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
