"""Shared pytest helpers — skip markers for tests that need difflogic / CUDA.

The Phase A login-node environment has neither, so these markers let the test
suite stay green while the bodies remain meaningful for Phase B.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch


def _has_difflogic() -> bool:
    return importlib.util.find_spec("difflogic") is not None


def _has_difflogic_cuda() -> bool:
    return importlib.util.find_spec("difflogic_cuda") is not None


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires CUDA")
needs_difflogic = pytest.mark.skipif(not _has_difflogic(),
                                     reason="requires `difflogic` package (proposal §Stage 0)")
needs_difflogic_cuda = pytest.mark.skipif(not _has_difflogic_cuda(),
                                          reason="requires `difflogic_cuda` extension (proposal §Stage 0 task 1)")
