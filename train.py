"""Training entry point.

Usage:
    python train.py --config configs/exp/01_mnist_lightning.yaml
    python train.py --config configs/base.yaml --dry-run

Each experiment lives in a single yaml under configs/exp/. The DATA.name and
MODEL.name fields in that yaml index the explicit registries below to pick a
LightningDataModule and LightningModule. Add a new entry to the registry when
landing a new model or dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Dict, Type

import lightning.pytorch as pl
from lightning.pytorch import Trainer, LightningDataModule, LightningModule
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils.config import AppConfig, load_config, config_to_dict
from src.utils.callbacks import load_callbacks
from src.utils.resume import get_resume_info
from src.utils.seeding import seed_all


from src.data.cifar10_dm import CIFAR10DataModule
from src.data.dvsgesture_dm import DVSGestureDataModule
from src.models.torchlogix_classifier import TorchlogixClassifier
from src.models.torchlogix_flow import TorchlogixFlow

# Registries — task-specific Lightning modules and datamodules. Add a new
# entry when landing a new task or dataset.
DATAMODULES: Dict[str, Type[LightningDataModule]] = {
    "cifar10": CIFAR10DataModule,
    "dvsgesture": DVSGestureDataModule,
}

MODELS: Dict[str, Type[LightningModule]] = {
    "torchlogix_classifier": TorchlogixClassifier,    # CIFAR-10 / DVS-Gesture
    "torchlogix_flow": TorchlogixFlow,                # MVSEC (Phase 2 stub)
}


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return "unknown"


def _build_trainer_kwargs(cfg: AppConfig, logger: TensorBoardLogger, callbacks: list):
    kwargs = dict(
        max_epochs=cfg.TRAINING.max_epochs,
        deterministic=cfg.TRAINING.deterministic,
        inference_mode=cfg.TRAINING.inference_mode,
        num_sanity_val_steps=cfg.TRAINING.num_sanity_val_steps,
        logger=logger,
        callbacks=callbacks,
        accelerator=cfg.DISTRIBUTED.accelerator,
        devices=cfg.DISTRIBUTED.devices,
        num_nodes=cfg.DISTRIBUTED.num_nodes,
        strategy=cfg.DISTRIBUTED.strategy,
        # Disable Lightning's default progress bar; PlainTextProgress in
        # src/utils/callbacks.py handles training progress with newlines that
        # survive log tailing.
        enable_progress_bar=False,
    )
    gc = cfg.OPTIMIZER.gradient_clip
    if gc.enabled:
        kwargs["gradient_clip_val"] = gc.gradient_clip_val
        kwargs["gradient_clip_algorithm"] = gc.gradient_clip_algorithm
    return {k: v for k, v in kwargs.items() if v is not None}


def _write_manifest(run_dir: str, cfg: AppConfig, runtime: dict):
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        "config": config_to_dict(cfg),
        "runtime": runtime,
        "git_commit": _git_commit_hash(),
        "seed": cfg.TRAINING.seed,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def main(cfg: AppConfig, runtime: dict, dry_run: bool):
    seed_all(cfg.TRAINING.seed, deterministic=cfg.TRAINING.deterministic)
    pl.seed_everything(cfg.TRAINING.seed, workers=True)

    # Activate TF32 / tensor cores for fp32 matmul before any CUDA tensor is
    # constructed. Lightning warns about this on Ampere+/L40S; here it's
    # config-controlled instead of the global default.
    import torch as _torch
    _torch.set_float32_matmul_precision(cfg.TRAINING.matmul_precision)

    info = get_resume_info(cfg, runtime)
    mode, run_name, version, ckpt_path = info["mode"], info["run_name"], info["version"], info["ckpt_path"]

    # DDP rank-0 propagates the run_name via env var so subprocess ranks reuse
    # it (each rank otherwise calls datetime.now() independently and lands in
    # a different timestamped dir, fragmenting tfevents). LOCAL_RANK is set by
    # Lightning before importing user code on subprocess ranks.
    rank0_runname_env = "TRAIN_PY_RUN_NAME"
    if os.environ.get("LOCAL_RANK") is None:
        # parent / single-process: use freshly-resolved run_name; export for kids.
        os.environ[rank0_runname_env] = run_name
    elif rank0_runname_env in os.environ:
        run_name = os.environ[rank0_runname_env]

    logger = TensorBoardLogger(save_dir=".", name=run_name, version=version if mode == "resume" else None)
    callbacks = load_callbacks(cfg)
    trainer_kwargs = _build_trainer_kwargs(cfg, logger, callbacks)
    trainer = Trainer(**trainer_kwargs)

    if dry_run:
        print(f"[dry-run] Trainer constructed. mode={mode}, run_name={run_name}, version={version}")
        print(f"[dry-run] DATA.name={cfg.DATA.name!r}  MODEL.name={cfg.MODEL.name!r}")
        print(f"[dry-run] Registered datamodules: {list(DATAMODULES)}")
        print(f"[dry-run] Registered models:      {list(MODELS)}")
        return

    if cfg.DATA.name not in DATAMODULES:
        raise KeyError(
            f"DATA.name={cfg.DATA.name!r} is not registered. "
            f"Known: {list(DATAMODULES)}. Register it in train.py:DATAMODULES."
        )
    if cfg.MODEL.name not in MODELS:
        raise KeyError(
            f"MODEL.name={cfg.MODEL.name!r} is not registered. "
            f"Known: {list(MODELS)}. Register it in train.py:MODELS."
        )

    dm = DATAMODULES[cfg.DATA.name](**cfg.DATA.args, dataloader_cfg=cfg.DATA.dataloader)
    if mode == "warmstart" and ckpt_path:
        model = MODELS[cfg.MODEL.name].load_from_checkpoint(
            ckpt_path,
            strict=bool(runtime.get("strict_state_dict", True)),
            map_location=runtime.get("map_location"),
            **cfg.MODEL.args,
        )
        ckpt_for_fit = None
    else:
        model = MODELS[cfg.MODEL.name](**cfg.MODEL.args)
        ckpt_for_fit = ckpt_path if mode == "resume" else None

    # `torch.compile` after construction (config-gated). For Tier 0 this fuses
    # the 16-op `bin_op_s` loop into a single CUDA kernel.
    # NB: `mode='reduce-overhead'` uses CUDAGraphs which caches tensor pointers
    # across iterations and crashes on our pattern (the LightningModule's
    # training_step receives outputs that the next iteration's CUDAGraph will
    # overwrite — `RuntimeError: accessing tensor output of CUDAGraphs that
    # has been overwritten`). The default mode keeps the inductor fusion of
    # `bin_op_s`'s 16-op loop without CUDAGraphs, which is the speedup we
    # actually wanted.
    if cfg.TRAINING.compile_model:
        model = _torch.compile(model)

    _write_manifest(os.path.join(run_name, version or "version_pending"), cfg, runtime)

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_for_fit)
    # Always evaluate on the actual test set after fit. Stage gates target test
    # accuracy, not the train-split val accuracy.
    trainer.test(model=model, datamodule=dm, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to experiment yaml.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build config + trainer, print summary, exit (no training).")
    parser.add_argument("--resume_from_last_checkpoint", action="store_true")
    parser.add_argument("--load_manual_checkpoint", default=None, type=str)
    parser.add_argument("--weights_only", action="store_true")
    parser.add_argument("--strict_state_dict", action="store_true")
    parser.add_argument("--no_strict_state_dict", dest="strict_state_dict", action="store_false")
    parser.add_argument("--map_location", default=None, type=str)
    parser.set_defaults(strict_state_dict=True)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"No config file found at {args.config}")

    cfg = load_config(args.config)
    runtime = dict(
        load_manual_checkpoint=args.load_manual_checkpoint,
        resume_from_last_checkpoint=args.resume_from_last_checkpoint,
        weights_only=args.weights_only,
        strict_state_dict=args.strict_state_dict,
        map_location=args.map_location,
    )
    main(cfg, runtime, args.dry_run)
