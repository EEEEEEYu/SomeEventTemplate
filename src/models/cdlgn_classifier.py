"""CDLGN classifier — LightningModule wrapping a configurable backbone.

The Phase-1 deliverable per proposal v3. The backbone is selected by name
from a small registry: `"cdlgn"` for the real LogicTreeNet, `"stub_conv"` for
a tiny standard-CNN stand-in used only to validate the data + Lightning glue
before the conv-logic primitives are exercised.

Discretized inference: every `LogicLayer` and `ConvLogicLayer` reads
`self.training` and dispatches to the argmax/one-hot path automatically when
Lightning calls `model.eval()` before val/test. We therefore log the val/test
metrics directly — they are the discretized numbers per proposal §Cross-cutting
rules ("Discretization is what matters").
"""

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_accuracy

from difflogic import LogicLayer

from src.modules.cdlgn import LogicTreeNet


def _ensure_difflogic_tensors_on_device(model: nn.Module) -> None:
    """`difflogic.LogicLayer` keeps `indices`, `given_x_indices_of_y_start` and
    `given_x_indices_of_y` as plain attributes (not registered buffers), so
    Lightning's `.to(device)` does not move them. Under DDP each rank lands
    its parameters on `cuda:<rank>` while these attrs stay on `cuda:0`,
    causing `illegal memory access` in the CUDA kernel.

    Walk every `LogicLayer` and move them to the device of that layer's own
    parameters (which Lightning has already placed correctly).
    """
    for sub in model.modules():
        if not isinstance(sub, LogicLayer):
            continue
        try:
            target = next(sub.parameters()).device
        except StopIteration:
            continue
        if isinstance(sub.indices, tuple):
            sub.indices = tuple(t.to(target) for t in sub.indices)
        if hasattr(sub, "given_x_indices_of_y_start"):
            sub.given_x_indices_of_y_start = sub.given_x_indices_of_y_start.to(target)
        if hasattr(sub, "given_x_indices_of_y"):
            sub.given_x_indices_of_y = sub.given_x_indices_of_y.to(target)


def _build_stub_conv(
    in_channels: int,
    k: int,
    num_classes: int,
    **_unused,
) -> nn.Module:
    """Tiny standard CNN used as a smoke-test backbone before plugging in the
    real CDLGN. Two Conv2d → ReLU → MaxPool blocks, then GAP + linear classifier.
    No logic gates; no discretization. Outputs raw logits of shape (B, num_classes).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(k, 2 * k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2 * k, num_classes),
    )


def _build_cdlgn_backbone(
    in_channels: int,
    k: int,
    tree_depth: int,
    channel_groups: int,
    dense_channel_groups: int,
    num_classes: int,
    tau: float,
    residual_init_z3: float,
    grad_factor: float,
    device: str,
    **_unused,
) -> nn.Module:
    return LogicTreeNet(
        in_channels=in_channels,
        k=k,
        tree_depth=tree_depth,
        channel_groups=channel_groups,
        dense_channel_groups=dense_channel_groups,
        num_classes=num_classes,
        tau=tau,
        residual_init_z3=residual_init_z3,
        grad_factor=grad_factor,
        device=device,
    )


BACKBONES = {
    "stub_conv": _build_stub_conv,
    "cdlgn": _build_cdlgn_backbone,
}


class CDLGNClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "cdlgn",
        in_channels: int = 9,                       # 3 channels × 3-bit thermo by default
        k: int = 256,                               # M (paper Phase-1 baseline)
        tree_depth: int = 3,
        channel_groups: int = 1,
        dense_channel_groups: int = 1,              # paper §A.3: should match channel_groups
        num_classes: int = 10,
        tau: float = 40.0,
        residual_init_z3: float = 5.0,
        grad_factor: float = 1.0,
        lr: float = 0.02,
        weight_decay: float = 0.002,
        lr_schedule: Optional[str] = None,           # None | "cosine"
        lr_schedule_T_max: Optional[int] = None,     # epochs over which to anneal
        lr_schedule_eta_min: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if backbone not in BACKBONES:
            raise KeyError(
                f"backbone={backbone!r} not in {list(BACKBONES)}; register it in BACKBONES."
            )
        device_for_difflogic = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = BACKBONES[backbone](
            in_channels=in_channels,
            k=k,
            tree_depth=tree_depth,
            channel_groups=channel_groups,
            dense_channel_groups=dense_channel_groups,
            num_classes=num_classes,
            tau=tau,
            residual_init_z3=residual_init_z3,
            grad_factor=grad_factor,
            device=device_for_difflogic,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # Expose layer groups for GradientNormLogger when the real CDLGN
        # backbone is used (its LogicTreeNet declares them).
        if hasattr(self.net, "layer_groups"):
            self.layer_groups = {
                # Prefix every group with "net." since the backbone is wrapped.
                k_: ["net." + p for p in v]
                for k_, v in self.net.layer_groups.items()
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Boolean inputs are accepted directly — the CDLGN backbone casts as
        # needed. The stub CNN backbone needs float inputs.
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
        return self.net(x)

    def on_fit_start(self) -> None:
        # See `_ensure_difflogic_tensors_on_device` — DDP needs LogicLayer's
        # unregistered tensors moved to each rank's device.
        _ensure_difflogic_tensors_on_device(self)

    def on_test_start(self) -> None:
        _ensure_difflogic_tensors_on_device(self)

    def on_validation_start(self) -> None:
        _ensure_difflogic_tensors_on_device(self)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = multiclass_accuracy(
            logits, y, num_classes=self.hparams.num_classes, average="micro", top_k=1
        )
        bs = x.shape[0]
        # sync_dist=True for epoch-level metrics so DDP averages across ranks.
        # Step-level train_loss is per-rank (cheap diagnostic); epoch metrics
        # are the contractual numbers we report.
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
