"""Per-slice DVS-Gesture DataModule (proposal v2 §Stage 2 reframed).

Emits `[T, 2, H, W]` per sample — `T` per-slice TBR tensors stacked along
the time axis — for the streaming buffer architecture. Reuses the same
HDF5 cache built by `DVSGestureDataModule` (the encoder writes
`[2, num_bins, H, W]`; this class permutes to `[num_bins, 2, H, W]` at
sample time, no re-encoding required).

The streaming `LightningModule.training_step` iterates over the `T` axis:

    for t in range(T):
        f_t = encoder(x[:, t])     # [B, 2, H, W] -> [B, M]
        buffer.step(f_t)           # writes row 0
        ...

`T = num_bins` from the encoder spec. With the default Tier 0 cache
(num_bins=128, bin_duration_us=16000), the streaming model sees 128 slices
per 2.048-second sample.

If you want a different `(num_bins, bin_duration)` for streaming-only runs
without affecting the Tier 0 cache, instantiate this DataModule with new
spec args — it'll write a separate cache directory keyed on the spec.
"""

from __future__ import annotations

import os
from typing import Optional

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Reuse the cache builder + spec dataclass from the fused DataModule. Only
# `__getitem__` differs.
from src.data.dvsgesture_dm import (
    DVSGestureDataModule,
    DVSGestureTBRSpec,
    _build_cache,
)
from src.utils.config import DataloaderConfig


class _CachedTBRPerSliceDataset(Dataset):
    """Same HDF5 cache as `_CachedTBRDataset`, but reshapes the
    `[2, num_bins, H, W]` boolean tensor to `[num_bins, 2, H, W]` so the
    leading axis is time — what a per-step streaming consumer expects.

    SWMR mode + lazy-open means this is fork-safe under DataLoader's
    `num_workers > 0`."""

    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path
        self._h: Optional[h5py.File] = None
        with h5py.File(h5_path, "r") as f:
            self._length = int(f["labels"].shape[0])

    def _ensure_open(self) -> None:
        if self._h is None:
            self._h = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        self._ensure_open()
        # Cache stores [2, num_bins, H, W]; permute to [num_bins, 2, H, W].
        # `np.transpose` returns a view; `from_numpy` on a non-contiguous
        # array works, but we copy via `.contiguous()` to keep downstream
        # `.view()`-style ops simple.
        raw = self._h["data"][idx]                                     # uint8 [2, num_bins, H, W]
        per_slice = np.transpose(raw, (1, 0, 2, 3))                    # [num_bins, 2, H, W]
        x = torch.from_numpy(per_slice.copy()).bool()                  # contiguous bool
        y = int(self._h["labels"][idx])
        return x.float(), y


class DVSGesturePerSliceDataModule(pl.LightningDataModule):
    """Streaming variant of `DVSGestureDataModule`. Same TBR encoding +
    HDF5 cache; per-sample output shape is `[num_bins, 2, H, W]`.

    Cache layout is identical to the fused DataModule (the underlying
    bytes are the same; only the `__getitem__` reshape changes), so if a
    fused-cache directory already exists it'll be reused without re-
    encoding.
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
    ) -> None:
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

    @property
    def T(self) -> int:
        """Number of slices per sample (= num_bins)."""
        return self.spec.num_bins

    def _cache_dir(self) -> str:
        return self.spec.cache_dir(self.data_root)

    def prepare_data(self) -> None:
        """Build the cache if it doesn't exist. Uses the same patched
        tonic loader as `DVSGestureDataModule` (figshare WAF blocks the
        downloader on this machine; the user-extracted `.npy` tree under
        `data/DVSGesture/` is the source of truth)."""
        import json
        import tonic

        cache_dir = self._cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        manifest_path = os.path.join(cache_dir, "manifest.json")
        train_h5 = os.path.join(cache_dir, "train.h5")
        test_h5 = os.path.join(cache_dir, "test.h5")
        if os.path.exists(train_h5) and os.path.exists(test_h5) and os.path.exists(manifest_path):
            return

        # Same monkey-patch as the fused DataModule: bypass the broken
        # tar.gz download check, require only the extracted .npy tree.
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
        full_train = _CachedTBRPerSliceDataset(train_h5)
        # Same 90/10 split as the fused DataModule for direct comparability.
        n = len(full_train)
        n_val = max(1, n // 10)
        n_train = n - n_val
        gen = torch.Generator().manual_seed(42)
        self.train_set, self.val_set = random_split(full_train, [n_train, n_val], generator=gen)
        self.test_set = _CachedTBRPerSliceDataset(test_h5)

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
