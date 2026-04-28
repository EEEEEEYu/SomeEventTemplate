"""Trainer callback assembly. Reads callback flags from AppConfig and returns a list."""

from __future__ import annotations

from typing import List

import lightning.pytorch.callbacks as plc
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.table import Table

from src.utils.config import AppConfig


class MultiRowRichProgressBar(RichProgressBar):
    """Rich progress bar that wraps logged metrics across multiple rows so wide
    metric tables don't truncate. Drops the noisy `v_num` field."""

    def __init__(self, metrics_per_row: int = 5, refresh_rate: int = 1, leave: bool = False):
        super().__init__(theme=RichProgressBarTheme())
        self.metrics_per_row = metrics_per_row

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        return metrics

    def _render_metrics_table(self, metrics: dict) -> Table:
        table = Table(show_header=False, box=None, expand=True)
        keys = list(metrics.keys())
        for i in range(0, len(keys), self.metrics_per_row):
            row = []
            for k in keys[i: i + self.metrics_per_row]:
                v = metrics[k]
                row.append(f"{k}: {v:.5f}" if isinstance(v, (int, float)) else f"{k}: {v}")
            table.add_row(*row)
        return table

    def render(self, *args, **kwargs):
        renderables = super().render(*args, **kwargs)
        metrics = self.get_metrics(self._trainer, self._trainer.lightning_module)
        if renderables and metrics:
            renderables[-1] = self._render_metrics_table(metrics)
        return renderables


def load_callbacks(cfg: AppConfig) -> List[plc.Callback]:
    callbacks: List[plc.Callback] = [
        MultiRowRichProgressBar(refresh_rate=1, leave=False, metrics_per_row=5),
        plc.LearningRateMonitor(logging_interval="epoch"),
    ]

    es = cfg.SCHEDULER.early_stopping
    if es.enabled:
        callbacks.append(plc.EarlyStopping(
            monitor=es.monitor, mode=es.mode, patience=es.patience, min_delta=es.min_delta,
        ))

    ckpt = cfg.CHECKPOINT
    if ckpt.enabled:
        callbacks.append(plc.ModelCheckpoint(
            every_n_epochs=ckpt.every_n_epochs,
            monitor=ckpt.monitor, mode=ckpt.mode,
            filename=ckpt.filename,
            save_top_k=ckpt.save_top_k, save_last=ckpt.save_last,
        ))

    ga = cfg.OPTIMIZER.gradient_accumulation
    if ga.enabled:
        callbacks.append(plc.GradientAccumulationScheduler(scheduling=ga.scheduling))

    swa = cfg.OPTIMIZER.stochastic_weight_averaging
    if swa.enabled:
        callbacks.append(plc.StochasticWeightAveraging(swa_lrs=swa.swa_lrs))

    return callbacks
