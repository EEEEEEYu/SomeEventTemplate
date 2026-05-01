"""Pass-through re-exports of difflogic's MLP primitives.

Existed in v2.1 as a seam point for the now-removed `WordLogicLayer` /
`ShiftedWordLogicLayer` extensions. In v3 it just re-exports the difflogic
primitives so existing call sites (`logic_classifier.py`) don't break;
new code may import directly from `difflogic` instead.
"""

from __future__ import annotations

import torch  # must precede difflogic import to load libc10 before the .so

from difflogic import LogicLayer, GroupSum

__all__ = ["LogicLayer", "GroupSum"]
