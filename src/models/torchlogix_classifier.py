"""LightningModule for classification tasks on a torchlogix backbone.

Used for both **CIFAR-10** (RGB images via `cifar10_classifier_backbone`)
and **DVS-Gesture** (TBR-encoded event frames via
`gesture_classifier_backbone`). Backbone and num_classes are config-driven so
the same Lightning class trains both tasks.

Discretization at eval is automatic: torchlogix's logic layers honor
`self.training`, so Lightning's `model.eval()` before val/test gives the
discretized accuracy without any extra hooks.
"""

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import multiclass_accuracy

from src.modules.torchlogix_backbones import (
    cifar10_classifier_backbone,
    gesture_classifier_backbone,
)


_BACKBONES = {
    "cifar10": cifar10_classifier_backbone,
    "gesture": gesture_classifier_backbone,
}


class TorchlogixClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "cifar10",
        backbone_kwargs: Optional[dict] = None,
        num_classes: int = 10,
        lr: float = 0.02,
        weight_decay: float = 0.002,
        lr_schedule: Optional[str] = None,
        lr_schedule_T_max: Optional[int] = None,
        lr_schedule_eta_min: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        if backbone not in _BACKBONES:
            raise KeyError(f"backbone={backbone!r} not in {list(_BACKBONES)}.")
        kwargs = dict(backbone_kwargs or {})
        self.net = _BACKBONES[backbone](**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
        # DVS-Gesture TBR comes in as `(B, P, T, H, W)`; the gesture backbone
        # treats `(P, T)` as a flat channel axis. CIFAR-10 inputs are already
        # 4-D and pass through unchanged.
        if x.dim() == 5:
            x = x.flatten(1, 2)
        return self.net(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = multiclass_accuracy(
            logits, y, num_classes=self.hparams.num_classes,
            average="micro", top_k=1,
        )
        bs = x.shape[0]
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched_name = self.hparams.lr_schedule
        if sched_name is None:
            return opt
        if sched_name == "cosine":
            T_max = self.hparams.lr_schedule_T_max or self.trainer.max_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=T_max, eta_min=self.hparams.lr_schedule_eta_min
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        raise ValueError(f"Unknown lr_schedule={sched_name!r}; expected None or 'cosine'.")
