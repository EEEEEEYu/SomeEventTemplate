"""CIFAR-10 LightningDataModule for CDLGN reproduction (proposal v3 §Phase 1).

Choices anchored to Petersen et al. 2024:
- 45,000 / 5,000 train/val split from the 50,000 official train set; the 10,000
  test set is held out (Appendix A.2). Split is reproducible under a fixed seed.
- Per-pixel thermometer encoding is applied inside the datamodule's transform
  so the model receives `(B, n_bits*3, 32, 32)` Boolean tensors directly. This
  keeps the model code task-agnostic (S/M/B/L/G all share the same forward).
- No data augmentation: the paper does not document any, and adding standard
  CIFAR-10 augmentation would invalidate the reproduction comparison.
- No normalization: thermometer encoding operates on `ToTensor()`'s [0,1]
  output; mean/std normalization would shift the threshold semantics.
- Bool dtype is preserved into the dataloader. Logic layers cast to float
  internally; storing bools halves memory bandwidth on the way to the GPU.
"""

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.data.thermometer import thermometer_encode
from src.utils.config import DataloaderConfig


class _ThermometerTransform:
    """Compose ToTensor() with thermometer_encode for the torchvision pipeline."""

    def __init__(self, n_bits: int):
        self.n_bits = n_bits
        self._to_tensor = transforms.ToTensor()

    def __call__(self, img):
        x = self._to_tensor(img)              # (3, 32, 32) float in [0, 1]
        return thermometer_encode(x, self.n_bits)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str = "data",
        n_bits: int = 3,
        val_size: int = 5_000,
        split_seed: int = 42,
        download: bool = True,
        dataloader_cfg: Optional[DataloaderConfig] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.n_bits = n_bits
        self.val_size = val_size
        self.split_seed = split_seed
        self.download = download
        self.dl_cfg = dataloader_cfg or DataloaderConfig()
        self.transform = _ThermometerTransform(n_bits)

    def prepare_data(self):
        if self.download:
            datasets.CIFAR10(self.data_root, train=True, download=True)
            datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        full_train = datasets.CIFAR10(self.data_root, train=True, transform=self.transform)
        train_size = len(full_train) - self.val_size
        gen = torch.Generator().manual_seed(self.split_seed)
        self.train_set, self.val_set = random_split(
            full_train, [train_size, self.val_size], generator=gen
        )
        self.test_set = datasets.CIFAR10(self.data_root, train=False, transform=self.transform)

    def _eval_batch_size(self):
        return self.dl_cfg.test_batch_size or self.dl_cfg.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.dl_cfg.batch_size,
            num_workers=self.dl_cfg.num_workers,
            shuffle=self.dl_cfg.shuffle_train,
            persistent_workers=self.dl_cfg.persistent_workers and self.dl_cfg.num_workers > 0,
            pin_memory=self.dl_cfg.pin_memory,
            drop_last=self.dl_cfg.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self._eval_batch_size(),
            num_workers=self.dl_cfg.num_workers,
            shuffle=False,
            persistent_workers=self.dl_cfg.persistent_workers and self.dl_cfg.num_workers > 0,
            pin_memory=self.dl_cfg.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self._eval_batch_size(),
            num_workers=self.dl_cfg.num_workers,
            shuffle=False,
            persistent_workers=self.dl_cfg.persistent_workers and self.dl_cfg.num_workers > 0,
            pin_memory=self.dl_cfg.pin_memory,
        )
