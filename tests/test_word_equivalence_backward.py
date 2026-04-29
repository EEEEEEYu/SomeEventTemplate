"""Backward-pass equivalence test — DROPPED in proposal v2.

> Proposal v2 §Stage 3 task 3 (line 241):
>   "Backward-pass equivalence test — dropped in v2. Without a meaningful
>   scalar reference for the streaming pipeline, this test would only
>   re-validate the same bit operations the forward test already covers.
>   The `M=1` forward equivalence is the parity anchor; deeper gradient
>   checks are deferred until/unless Stage 4 fails the gate and we need
>   to localise."

The original v1 body is preserved in git history (see commit 6e1ef5e
scaffolds and the v2-pivot rewrite). It is left as a `pytest.skip` rather
than deleted so the dropped-test rationale is auditable in the test suite
(per Phase P plan §P3 task 5).

Re-enable this test only if v2 Stage 4 fails its accuracy gate and we
need to disambiguate substrate vs operator bugs.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="Backward-pass equivalence dropped in proposal v2 §Stage 3 task 3. "
           "M=1 forward parity in test_word_equivalence_forward.py is the substrate "
           "anchor; deeper gradient checks are deferred unless Stage 4 fails its gate."
)
def test_word_layer_backward_matches_difflogic():
    pytest.skip("dropped in v2")
