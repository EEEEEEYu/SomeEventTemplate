"""Decide how to start a training run: scratch, resume, or warm-start."""

from __future__ import annotations

import datetime
import os
import re
from typing import Any, Dict

from src.utils.config import AppConfig

_VERSION_RE = re.compile(r"^version_(\d+)$")


def _pick_latest_ckpt(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith(".ckpt"):
            continue
        if fn == "last.ckpt" or fn.startswith("latest") or "epoch=" in fn:
            full = os.path.join(ckpt_dir, fn)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            candidates.append((mtime, full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _find_latest_run_dir(root_dir: str):
    if not os.path.isdir(root_dir):
        return None
    entries = []
    for d in os.listdir(root_dir):
        full = os.path.join(root_dir, d)
        if os.path.isdir(full):
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            entries.append((mtime, full))
    if not entries:
        return None
    entries.sort(reverse=True)
    return entries[0][1]


def _find_latest_version_dir(run_dir: str):
    if not os.path.isdir(run_dir):
        return None
    candidates = []
    for d in os.listdir(run_dir):
        m = _VERSION_RE.match(d)
        if not m:
            continue
        full = os.path.join(run_dir, d)
        if not os.path.isdir(full):
            continue
        idx = int(m.group(1))
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0
        candidates.append((idx, mtime, full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _parse_run_and_version_from_ckpt(ckpt_path: str):
    checkpoints_dir = os.path.dirname(ckpt_path)
    version_dir = os.path.dirname(checkpoints_dir)
    run_dir = os.path.dirname(version_dir)
    version_name = os.path.basename(version_dir)
    run_name_rel = os.path.relpath(run_dir, start=".")
    return run_name_rel, version_name, version_dir, run_dir


def _new_run_name(log_dir_root: str, experiment_name: str) -> str:
    return os.path.join(
        log_dir_root,
        f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{experiment_name}",
    )


def get_resume_info(cfg: AppConfig, runtime: Dict[str, Any]):
    """Return {mode, run_name, version, ckpt_path}.

    mode is one of:
      'scratch'   — new run, new version
      'resume'    — restore full trainer state from a checkpoint
      'warmstart' — load weights only, start a new version under same run
    """
    log_dir_root = cfg.LOGGER.log_dir_root
    experiment_name = cfg.LOGGER.experiment_name

    if not cfg.CHECKPOINT.enabled:
        return dict(mode="scratch", run_name=_new_run_name(log_dir_root, experiment_name),
                    version=None, ckpt_path=None)

    load_manual = runtime.get("load_manual_checkpoint")
    resume_last = runtime.get("resume_from_last_checkpoint")
    weights_only = bool(runtime.get("weights_only", False))

    if load_manual:
        if not os.path.isfile(load_manual):
            raise FileNotFoundError(f"Manual checkpoint not found: {load_manual}")
        run_name, version_name, _, _ = _parse_run_and_version_from_ckpt(load_manual)
        if weights_only:
            return dict(mode="warmstart", run_name=run_name, version=None, ckpt_path=load_manual)
        return dict(mode="resume", run_name=run_name, version=version_name, ckpt_path=load_manual)

    if resume_last:
        latest_run_dir = _find_latest_run_dir(log_dir_root)
        if latest_run_dir is None:
            print(f"Warning: resume_from_last_checkpoint=True but no runs under {log_dir_root}. Starting new training.")
            return dict(mode="scratch", run_name=_new_run_name(log_dir_root, experiment_name),
                        version=None, ckpt_path=None)
        version_dir = _find_latest_version_dir(latest_run_dir)
        if version_dir is None:
            return dict(mode="scratch", run_name=_new_run_name(log_dir_root, experiment_name),
                        version=None, ckpt_path=None)
        ckpt_path = _pick_latest_ckpt(os.path.join(version_dir, "checkpoints"))
        if ckpt_path is None:
            return dict(mode="scratch", run_name=_new_run_name(log_dir_root, experiment_name),
                        version=None, ckpt_path=None)
        run_name_rel = os.path.relpath(os.path.dirname(version_dir), start=".")
        return dict(mode="resume", run_name=run_name_rel,
                    version=os.path.basename(version_dir), ckpt_path=ckpt_path)

    return dict(mode="scratch", run_name=_new_run_name(log_dir_root, experiment_name),
                version=None, ckpt_path=None)
