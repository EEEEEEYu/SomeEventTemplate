"""Stage 4 task 3: ShiftedWordLogicLayer discretized correctness.

For a few hand-picked (op_idx, shift_b) configurations, check that the
discretized forward matches the reference op(a, SHIFT_b(b)) over packed words.

Scaffold — body lands with Stage 4's ShiftedWordLogicLayer.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from .conftest import needs_difflogic, needs_cuda


def _has_word_logic() -> bool:
    return importlib.util.find_spec("src.modules.word_logic") is not None


needs_word_logic = pytest.mark.skipif(
    not _has_word_logic(),
    reason="requires src/modules/word_logic.py with ShiftedWordLogicLayer (proposal §Stage 4 task 2)",
)


@needs_difflogic
@needs_cuda
@needs_word_logic
@pytest.mark.parametrize("op_idx,shift_b", [(6, 0), (6, 1), (8, 4), (10, 8)])  # XOR, OR, AND × shifts
def test_shifted_word_layer_matches_reference(op_idx, shift_b):
    from src.modules.word_logic import ShiftedWordLogicLayer  # noqa: F401

    # TODO Stage 4: instantiate a 2-input ShiftedWordLogicLayer with a single
    # neuron locked to (op_idx, shift_b), compute its discretized forward on a
    # random binary [B, 2, W] input, and compare against the hand-written
    # reference (apply the binary op after right-shifting input b by shift_b
    # bits within each word).
    pytest.skip("Stage 4 implementation pending — reference body lives in this file's TODO.")
