"""Stage 1: pass-through wrappers around difflogic primitives.

Importing here (rather than from `difflogic` directly throughout the codebase)
gives us a single seam to drop in `WordLogicLayer` (Stage 3 / Phase P §P3)
and `ShiftedWordLogicLayer` (Phase P §P4c) without touching every call site.

Per proposal §"Don't touch difflogic CUDA in Stages 1–4", these are
re-exports — no behavioral changes to the originals.
"""

from __future__ import annotations

import torch  # must precede difflogic import to load libc10 before the .so

from difflogic import LogicLayer, GroupSum

from src.modules.word_logic import WordLogicLayer

__all__ = ["LogicLayer", "GroupSum", "WordLogicLayer"]
