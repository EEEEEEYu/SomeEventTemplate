"""Trainer callback assembly. Reads callback flags from AppConfig and returns a list."""

from __future__ import annotations

import os
import time
from typing import List, Optional

import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc

from src.utils.config import AppConfig
from src.utils.diagnostics import GradientNormLogger


class PlainTextProgress(plc.Callback):
    """Newline-per-update training progress that survives `tail -f`.

    The default `RichProgressBar` redraws in place, which is invisible to
    log tailers and `tmux capture-pane`. This callback prints a single line
    per `print_every_n_steps` steps and a one-line summary at every epoch
    end. Rank-0 only.
    """

    def __init__(self, print_every_n_steps: int = 50):
        super().__init__()
        self.print_every_n_steps = print_every_n_steps
        self._epoch_start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None

    @staticmethod
    def _is_rank_zero() -> bool:
        return int(os.environ.get("LOCAL_RANK", 0)) == 0 and int(os.environ.get("RANK", 0)) == 0

    def on_train_epoch_start(self, trainer, pl_module):
        if not self._is_rank_zero():
            return
        self._epoch_start_time = time.time()
        self._last_step_time = self._epoch_start_time
        total_steps = trainer.num_training_batches
        print(
            f"[epoch {trainer.current_epoch:3d} start] steps={total_steps} "
            f"global_step={trainer.global_step}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._is_rank_zero():
            return
        if (batch_idx + 1) % self.print_every_n_steps != 0:
            return
        now = time.time()
        dt = now - (self._last_step_time or now)
        self._last_step_time = now
        steps_per_sec = self.print_every_n_steps / dt if dt > 0 else 0.0

        loss = outputs["loss"].item() if isinstance(outputs, dict) and "loss" in outputs else float("nan")
        total = trainer.num_training_batches
        print(
            f"[epoch {trainer.current_epoch:3d}] step {batch_idx + 1:5d}/{total:5d} "
            f"({(batch_idx + 1) / total * 100:5.1f}%)  loss={loss:.4f}  "
            f"{steps_per_sec:5.2f} it/s",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._is_rank_zero():
            return
        dt = time.time() - (self._epoch_start_time or time.time())
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                   if hasattr(v, "item") or isinstance(v, (int, float))}
        # Compact one-line summary.
        keys = ["train_loss_epoch", "train_acc", "val_loss", "val_acc"]
        parts = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]
        print(
            f"[epoch {trainer.current_epoch:3d} end  ] elapsed={dt:6.1f}s  "
            + "  ".join(parts),
            flush=True,
        )

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._is_rank_zero():
            return
        m = trainer.callback_metrics
        ta = m.get("test_acc")
        tl = m.get("test_loss")
        ta_v = ta.item() if hasattr(ta, "item") else ta
        tl_v = tl.item() if hasattr(tl, "item") else tl
        print(f"[test] test_acc={ta_v}  test_loss={tl_v}", flush=True)


def load_callbacks(cfg: AppConfig) -> List[plc.Callback]:
    callbacks: List[plc.Callback] = [
        PlainTextProgress(print_every_n_steps=50),
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

    diag = cfg.DIAGNOSTICS
    if diag.gradient_norm_logger:
        callbacks.append(GradientNormLogger(
            log_every_n_steps=diag.gradient_norm_log_every_n_steps,
        ))

    return callbacks
