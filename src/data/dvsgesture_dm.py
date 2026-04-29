"""DVS-Gesture DataModule (Phase P §P4b / proposal v2.1 §Tier 0).

Tier 0 wants a single fused TBR tensor per sample (the v1 framing) — not the
per-slice [T, 2, H, W] format the v2 streaming path needs. So this DataModule
emits `[2, num_bins, H, W]` boolean tensors directly, computed once at first
access and cached to HDF5 to avoid re-encoding every epoch.

Defaults (proposal v2.1 §Tier 0, line 105):
    M = num_bins = 128, bin_duration_us = 16000 (16 ms)
    → 2.048-second sample window
    Downsample 128×128 → 32×32 (proposal §Stage 2 task 5 recommendation)

Samples shorter than the window are zero-padded at the end (no events in
late bins). Samples longer are truncated. Variable sample length is on the
open-risks list (STATUS.md).

Cache layout:
    data/dvsgesture_tbr_M{num_bins}_bin{bin_duration_us}_H{H}xW{W}/
        train.h5         — datasets `data` [N,2,B,H,W] uint8, `labels` [N] int64
        test.h5
        manifest.json    — encoding params for re-encoding/audit
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.tbr import encode_tbr
from src.utils.config import DataloaderConfig


@dataclass
class DVSGestureTBRSpec:
    num_bins: int = 128
    bin_duration_us: int = 16_000      # 16 ms / bin → 2.048 s / sample at num_bins=128
    sensor_h: int = 32                 # downsampled from 128
    sensor_w: int = 32
    polarities: int = 2

    @property
    def sample_window_us(self) -> int:
        return self.num_bins * self.bin_duration_us

    def cache_dir(self, root: str) -> str:
        return os.path.join(
            root,
            f"dvsgesture_tbr_M{self.num_bins}_bin{self.bin_duration_us}_H{self.sensor_h}xW{self.sensor_w}",
        )


def _events_to_tensor(events_struct: np.ndarray, spec: DVSGestureTBRSpec) -> torch.Tensor:
    """Convert tonic's structured event array to TBR tensor [P, num_bins, H, W].

    tonic.datasets.DVSGesture.dtype = [('x', i2), ('y', i2), ('p', ?), ('t', i8)]
    where p is bool (False=OFF/0, True=ON/1) and t is microseconds.
    """
    if events_struct.size == 0:
        return torch.zeros(spec.polarities, spec.num_bins, spec.sensor_h, spec.sensor_w,
                           dtype=torch.bool)

    # Stack into [N, 4] = (x, y, t, p) — encode_tbr's calling convention.
    x = torch.from_numpy(events_struct["x"].astype(np.int64))
    y = torch.from_numpy(events_struct["y"].astype(np.int64))
    t = torch.from_numpy(events_struct["t"].astype(np.int64))
    p = torch.from_numpy(events_struct["p"].astype(np.int64))  # bool → int

    # Downsample 128→32 (factor 4) by integer-divide. Multiple events at the
    # same downsampled pixel collapse to one (TBR's any-event semantics handle this).
    x = x // 4
    y = y // 4

    events = torch.stack([x, y, t, p], dim=1)
    return encode_tbr(
        events,
        num_bins=spec.num_bins,
        bin_duration_us=spec.bin_duration_us,
        sensor_size=(spec.sensor_h, spec.sensor_w, spec.polarities),
        t0=int(t.min().item()),
    )


class _CachedTBRDataset(Dataset):
    """Reads pre-encoded boolean tensors from an HDF5 cache. The cache stores
    bools as uint8 to keep h5py + compression simple; we cast back at __getitem__."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        # Open lazily — we need to be fork-safe with num_workers > 0.
        self._h: Optional[h5py.File] = None
        with h5py.File(h5_path, "r") as f:
            self._length = int(f["labels"].shape[0])

    def _ensure_open(self):
        if self._h is None:
            self._h = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        self._ensure_open()
        x = torch.from_numpy(self._h["data"][idx]).bool()  # [P, num_bins, H, W]
        y = int(self._h["labels"][idx])
        return x.float(), y


def _build_cache(out_path: str, tonic_dataset, spec: DVSGestureTBRSpec) -> None:
    """Encode every sample to TBR and write to HDF5. Boolean as uint8 with
    LZF compression — DVS-Gesture is small enough that this fits comfortably
    on /fs/nexus-scratch."""
    n = len(tonic_dataset)
    shape = (n, spec.polarities, spec.num_bins, spec.sensor_h, spec.sensor_w)
    with h5py.File(out_path, "w") as f:
        data_ds = f.create_dataset(
            "data", shape=shape, dtype=np.uint8,
            compression="lzf", chunks=(1, spec.polarities, spec.num_bins, spec.sensor_h, spec.sensor_w),
        )
        label_ds = f.create_dataset("labels", shape=(n,), dtype=np.int64)
        for i in range(n):
            events, label = tonic_dataset[i]
            tensor = _events_to_tensor(events, spec)
            data_ds[i] = tensor.to(torch.uint8).numpy()
            label_ds[i] = int(label)


class DVSGestureDataModule(pl.LightningDataModule):
    """Single-fused TBR DataModule for Tier 0 (proposal v2.1 §Tier 0, line 104).

    Emits `(x, y)` where x has shape `[2, num_bins, H, W]` (float32 in {0, 1})
    and y is a class index in [0, 11). The first-time setup encodes events to
    TBR and writes to an HDF5 cache; subsequent setups read from cache.

    For the per-slice streaming variant (Stage 2/4 work, post-Tier-0), a
    separate DataModule will emit `[T, 2, H, W]` per sample. They will share
    the same `encode_tbr` foundation in `src/data/tbr.py`.
    """

    def __init__(
        self,
        data_root: str = "data",
        num_bins: int = 128,
        bin_duration_us: int = 16_000,
        sensor_h: int = 32,
        sensor_w: int = 32,
        download: bool = True,
        dataloader_cfg: Optional[DataloaderConfig] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.spec = DVSGestureTBRSpec(
            num_bins=num_bins,
            bin_duration_us=bin_duration_us,
            sensor_h=sensor_h,
            sensor_w=sensor_w,
        )
        self.download = download
        self.dl_cfg = dataloader_cfg or DataloaderConfig()

    @property
    def num_classes(self) -> int:
        return 11

    def _cache_dir(self) -> str:
        return self.spec.cache_dir(self.data_root)

    def prepare_data(self) -> None:
        # Tonic handles download lazily inside the dataset class. Build the
        # cache here so DDP workers don't all race to encode.
        import tonic
        cache_dir = self._cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        manifest_path = os.path.join(cache_dir, "manifest.json")
        train_h5 = os.path.join(cache_dir, "train.h5")
        test_h5 = os.path.join(cache_dir, "test.h5")

        if os.path.exists(train_h5) and os.path.exists(test_h5) and os.path.exists(manifest_path):
            return  # cache hit

        # `tonic.datasets.DVSGesture._check_exists` requires the original
        # tar.gz to be present alongside the extracted .npy tree, but
        # figshare's WAF blocks tonic's downloader on this network. The user
        # downloaded + extracted the data manually; tar.gz isn't around. Patch
        # `_check_exists` to require only the .npy tree (the archive itself is
        # never read at sample-time). Patching the upstream class rather than
        # subclassing keeps tonic's `location_on_system` path
        # (`{save_to}/DVSGesture/`) intact — a subclass would resolve to
        # `{save_to}/<SubclassName>/` and re-trigger the broken download.
        if not getattr(tonic.datasets.DVSGesture, "_p4d_check_exists_patched", False):
            tonic.datasets.DVSGesture._check_exists = (
                lambda self: self._folder_contains_at_least_n_files_of_type(100, ".npy")
            )
            tonic.datasets.DVSGesture._p4d_check_exists_patched = True

        train_set = tonic.datasets.DVSGesture(save_to=self.data_root, train=True)
        test_set = tonic.datasets.DVSGesture(save_to=self.data_root, train=False)

        _build_cache(train_h5, train_set, self.spec)
        _build_cache(test_h5, test_set, self.spec)

        with open(manifest_path, "w") as f:
            json.dump({
                "num_bins": self.spec.num_bins,
                "bin_duration_us": self.spec.bin_duration_us,
                "sensor_h": self.spec.sensor_h,
                "sensor_w": self.spec.sensor_w,
                "polarities": self.spec.polarities,
                "sample_window_us": self.spec.sample_window_us,
                "downsample_factor": 4,
                "source_sensor_size": [128, 128, 2],
                "n_train": len(train_set),
                "n_test": len(test_set),
                "encoder": "src.data.tbr.encode_tbr",
            }, f, indent=2)

    def setup(self, stage: Optional[str] = None) -> None:
        cache_dir = self._cache_dir()
        train_h5 = os.path.join(cache_dir, "train.h5")
        test_h5 = os.path.join(cache_dir, "test.h5")
        full_train = _CachedTBRDataset(train_h5)
        # No held-out val split for Tier 0 — DVS-Gesture's official splits are
        # train/test. Use 10% of train as val deterministically.
        n = len(full_train)
        n_val = max(1, n // 10)
        n_train = n - n_val
        gen = torch.Generator().manual_seed(42)
        from torch.utils.data import random_split
        self.train_set, self.val_set = random_split(full_train, [n_train, n_val], generator=gen)
        self.test_set = _CachedTBRDataset(test_h5)

    def _eval_batch_size(self) -> int:
        return self.dl_cfg.test_batch_size or self.dl_cfg.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.dl_cfg.batch_size,
            num_workers=self.dl_cfg.num_workers,
            shuffle=self.dl_cfg.shuffle_train,
            persistent_workers=self.dl_cfg.persistent_workers and self.dl_cfg.num_workers > 0,
            pin_memory=self.dl_cfg.pin_memory,
            drop_last=self.dl_cfg.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self._eval_batch_size(),
            num_workers=self.dl_cfg.num_workers,
            shuffle=False,
            persistent_workers=self.dl_cfg.persistent_workers and self.dl_cfg.num_workers > 0,
            pin_memory=self.dl_cfg.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self._eval_batch_size(),
            num_workers=self.dl_cfg.num_workers,
            shuffle=False,
            persistent_workers=self.dl_cfg.persistent_workers and self.dl_cfg.num_workers > 0,
            pin_memory=self.dl_cfg.pin_memory,
        )
