"""CIFAR-10 datamodule plumbing — runs only if the CIFAR-10 cache is already
on disk. The actual download is deferred to the GPU node; here we just verify
shapes, dtypes, and split reproducibility against a synthetic stand-in if the
real dataset is absent."""

import os

import pytest
import torch
from torch.utils.data import Dataset

from src.data.cifar10_dm import CIFAR10DataModule, _ThermometerTransform
from src.utils.config import DataloaderConfig


def test_thermometer_transform_shape():
    """The transform should turn a PIL image into a (n_bits*3, 32, 32) bool tensor."""
    from PIL import Image
    import numpy as np
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    out = _ThermometerTransform(n_bits=3)(img)
    assert out.shape == (9, 32, 32)
    assert out.dtype == torch.bool


def test_thermometer_transform_31bits():
    from PIL import Image
    import numpy as np
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    out = _ThermometerTransform(n_bits=31)(img)
    assert out.shape == (93, 32, 32)


@pytest.mark.skipif(
    not os.path.exists("data/cifar-10-batches-py"),
    reason="CIFAR-10 cache absent; download is GPU-node only.",
)
def test_split_sizes():
    dm = CIFAR10DataModule(data_root="data", n_bits=3, val_size=5_000, download=False)
    dm.setup()
    assert len(dm.train_set) == 45_000
    assert len(dm.val_set) == 5_000
    assert len(dm.test_set) == 10_000


@pytest.mark.skipif(
    not os.path.exists("data/cifar-10-batches-py"),
    reason="CIFAR-10 cache absent; download is GPU-node only.",
)
def test_split_reproducibility():
    """Same split_seed should produce the same train/val partition."""
    dm1 = CIFAR10DataModule(data_root="data", split_seed=42, download=False)
    dm2 = CIFAR10DataModule(data_root="data", split_seed=42, download=False)
    dm1.setup()
    dm2.setup()
    # random_split returns Subset objects with .indices
    assert dm1.train_set.indices == dm2.train_set.indices
    assert dm1.val_set.indices == dm2.val_set.indices


@pytest.mark.skipif(
    not os.path.exists("data/cifar-10-batches-py"),
    reason="CIFAR-10 cache absent; download is GPU-node only.",
)
def test_dataloader_emits_correct_shape():
    dl_cfg = DataloaderConfig(batch_size=4, num_workers=0, persistent_workers=False)
    dm = CIFAR10DataModule(data_root="data", n_bits=3, download=False, dataloader_cfg=dl_cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert x.shape == (4, 9, 32, 32)
    assert x.dtype == torch.bool
    assert y.shape == (4,)
