"""P2 task 5 — unit tests for the always-on diagnostic infrastructure
(proposal v2.1 §"Always-on additions").

Two pieces under test:
  1. GradientNormLogger — per-layer-group L2 grad norms; group resolution via
     `pl_module.layer_groups`; "all" fallback when absent.
  2. freeze_decoder_for_warmup — encoder warm-up freezing during the first
     `encoder_warmup_epochs` epochs.

These run on CPU. No GPU / difflogic dependency — the diagnostics layer is
purely PyTorch.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn

from src.utils.diagnostics import (
    GradientNormLogger,
    freeze_decoder_for_warmup,
)


class _ToyEncoderDecoder(pl.LightningModule):
    """2-encoder-layer + 2-decoder-layer toy MLP with declared layer groups."""

    def __init__(self, encoder_warmup_epochs: int = 0):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        self.readout = nn.Linear(4, 4)
        self.layer_groups = {
            "encoder": ["encoder."],
            "decoder": ["decoder."],
            "readout": ["readout."],
        }
        self.encoder_warmup_epochs = encoder_warmup_epochs

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
    """A bare model (Stage 1) without `layer_groups` should still be reportable
    under a single 'all' bucket — the diagnostic isn't useful here but the
    callback must not crash."""
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


def test_warmup_freezes_decoder_then_unfreezes_after_warmup_epochs():
    model = _ToyEncoderDecoder(encoder_warmup_epochs=2)

    # Epoch 0 — in warm-up. Decoder + readout? Note: readout is kept unfrozen
    # alongside encoder so the loss head can still learn the trivial mapping.
    changed = freeze_decoder_for_warmup(model, current_epoch=0)
    assert changed
    decoder_params_frozen = all(not p.requires_grad for p in model.decoder.parameters())
    encoder_params_trainable = all(p.requires_grad for p in model.encoder.parameters())
    readout_params_trainable = all(p.requires_grad for p in model.readout.parameters())
    assert decoder_params_frozen, "decoder must be frozen during warm-up"
    assert encoder_params_trainable, "encoder must remain trainable during warm-up"
    assert readout_params_trainable, "readout must remain trainable during warm-up"

    # Epoch 2 — warm-up over. Everything should be trainable.
    changed = freeze_decoder_for_warmup(model, current_epoch=2)
    assert changed
    assert all(p.requires_grad for p in model.parameters())


def test_warmup_is_noop_when_no_encoder_group_declared():
    """Stage 1 case: without an encoder/decoder split, warm-up does nothing."""
    model = pl.LightningModule()
    model.body = nn.Linear(8, 4)
    model.encoder_warmup_epochs = 5  # set but uneffective without `layer_groups`

    changed = freeze_decoder_for_warmup(model, current_epoch=0)
    assert not changed
    assert all(p.requires_grad for p in model.parameters())


def test_warmup_is_noop_when_warmup_epochs_is_zero():
    model = _ToyEncoderDecoder(encoder_warmup_epochs=0)
    changed = freeze_decoder_for_warmup(model, current_epoch=0)
    assert not changed
    assert all(p.requires_grad for p in model.parameters())


def test_gradnorm_excludes_frozen_params_via_requires_grad_false():
    """A parameter with grad=None (because it's frozen / didn't receive grad)
    should not contribute. Backprop through a partially-frozen model should
    still produce a sensible group norm for the live params."""
    model = _ToyEncoderDecoder(encoder_warmup_epochs=1)
    freeze_decoder_for_warmup(model, current_epoch=0)  # freeze decoder

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()

    logger = GradientNormLogger(log_every_n_steps=1)
    norms = logger.get_param_group_norms(model)
    # Decoder is frozen — its params don't appear in the cached groups
    # (requires_grad=False), so we expect no decoder entry at all.
    assert "decoder" not in norms or norms["decoder"] is None
    assert norms.get("encoder") is not None and norms["encoder"] > 0
