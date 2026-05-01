"""Unit tests for the per-layer-group gradient-norm logger.

Runs on CPU; no GPU / difflogic dependency.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn

from src.utils.diagnostics import GradientNormLogger


class _ToyEncoderDecoder(pl.LightningModule):
    """2-encoder-layer + 2-decoder-layer toy MLP with declared layer groups."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        self.readout = nn.Linear(4, 4)
        self.layer_groups = {
            "encoder": ["encoder."],
            "decoder": ["decoder."],
            "readout": ["readout."],
        }

    def forward(self, x):
        return self.readout(self.decoder(self.encoder(x)))


def test_gradnorm_groups_match_layer_groups_attribute():
    model = _ToyEncoderDecoder()
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()

    logger = GradientNormLogger(log_every_n_steps=1)
    norms = logger.get_param_group_norms(model)

    assert set(norms.keys()) == {"encoder", "decoder", "readout"}
    for group, value in norms.items():
        assert value is not None, f"group {group} should have non-None grad norm after backward()"
        assert value > 0, f"group {group} grad norm should be > 0; got {value}"


def test_gradnorm_falls_back_to_single_all_group_when_no_layer_groups():
    """A bare model without `layer_groups` should still be reportable under a
    single 'all' bucket — the diagnostic isn't useful here but the callback
    must not crash."""
    model = pl.LightningModule()
    model.body = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    y = torch.randint(0, 4, (2,))
    loss = nn.functional.cross_entropy(model.body(x), y)
    loss.backward()

    logger = GradientNormLogger(log_every_n_steps=1)
    norms = logger.get_param_group_norms(model)
    assert list(norms.keys()) == ["all"]
    assert norms["all"] is not None and norms["all"] > 0


def test_gradnorm_excludes_frozen_params():
    """A parameter with `requires_grad=False` should be excluded from group norms."""
    model = _ToyEncoderDecoder()
    for p in model.decoder.parameters():
        p.requires_grad = False

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()

    logger = GradientNormLogger(log_every_n_steps=1)
    norms = logger.get_param_group_norms(model)
    assert "decoder" not in norms or norms["decoder"] is None
    assert norms.get("encoder") is not None and norms["encoder"] > 0
