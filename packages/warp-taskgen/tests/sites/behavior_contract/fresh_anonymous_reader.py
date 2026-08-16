"""Assertions for ordinary-reader readback interpretation.

The input is a separately constructed fake observation.  This module verifies
the public interpreter's fail-closed rules; it does not claim live browser or
Fresh Anonymous Reader evidence.
"""

from __future__ import annotations

from warp_taskgen.sites import BoundSite, ReadbackDecision, ReadbackFailure, ReadbackObservation


def assert_fresh_anonymous_reader_behavior(
    bound_site: BoundSite,
    observation: ReadbackObservation,
    *,
    writer_context_observation: ReadbackObservation,
    expected_reason: str,
) -> None:
    """Interpret two separately-built observations without claiming live proof."""

    assert observation is not writer_context_observation
    decision = bound_site.interpret_readback(observation)
    assert isinstance(decision, ReadbackDecision)
    assert decision.verified is True
    assert decision.reason == expected_reason

    rejected = bound_site.interpret_readback(writer_context_observation)
    assert isinstance(rejected, ReadbackDecision)
    assert rejected.verified is False
    assert rejected.reason == "writer_context_reused"

    malformed = bound_site.interpret_readback(
        ReadbackObservation(
            observation.kind,
            observation.identity_tokens,
            {},
            signature=observation.signature,
        )
    )
    assert isinstance(malformed, ReadbackDecision | ReadbackFailure)
    assert not getattr(malformed, "verified", False)
