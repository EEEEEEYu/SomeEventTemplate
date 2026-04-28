"""MNIST DataModule for Stage 1 (proposal §Stage 1 task 3).

Key choices, all of which mirror difflogic/experiments/main.py to maximize
parity with our 97.40% Stage 0 baseline:

- 50k train / 10k val / 10k test split. The 60k MNIST train set is split
  50k/10k with a fixed seed; the 10k MNIST test set is held out as test.
- Inputs are float in [0, 1] (`torchvision.transforms.ToTensor`), NOT
  binarized in the dataloader. The proposal's "binarize at 0.5" wording is
  realized in the LightningModule by `.round()`-ing inputs before forward in
  eval mode — matching Petersen's `eval()` function which uses `x.round()`.
  Dataloader-side binarization would diverge from the reference and risk
  failing the ±0.3% parity gate.
- Flattening happens in the model (via `nn.Flatten`), not here.
"""

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils.config import DataloaderConfig


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str = "data",
        val_size: int = 10_000,
        split_seed: int = 42,
        download: bool = True,
        dataloader_cfg: Optional[DataloaderConfig] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.val_size = val_size
        self.split_seed = split_seed
        self.download = download
        self.dl_cfg = dataloader_cfg or DataloaderConfig()
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        if self.download:
            datasets.MNIST(self.data_root, train=True, download=True)
            datasets.MNIST(self.data_root, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        full_train = datasets.MNIST(self.data_root, train=True, transform=self.transform)
        train_size = len(full_train) - self.val_size
        gen = torch.Generator().manual_seed(self.split_seed)
        self.train_set, self.val_set = random_split(full_train, [train_size, self.val_size], generator=gen)
        self.test_set = datasets.MNIST(self.data_root, train=False, transform=self.transform)

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
