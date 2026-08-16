"""Separately constructed fake reader observations for the test-only Site."""

from __future__ import annotations

from tests.sites.synthetic_discussion_forum.site import ORIGIN, SITE, THREAD_ID
from warp_taskgen.sites import ReadbackObservation

ACTOR = "participant"
BODY = "A regular participant reply. TRACE-SIGNATURE"
COMMENT_ID = "42"
SIGNATURE = "TRACE-SIGNATURE"


def fresh_anonymous_observation() -> ReadbackObservation:
    """Return static fake data; this is not live Fresh Anonymous Reader proof."""

    return ReadbackObservation(
        "comment_visibility",
        {
            "thread_id": THREAD_ID,
            "comment_id": COMMENT_ID,
            "actor": ACTOR,
        },
        {
            "site": SITE,
            "origin": ORIGIN,
            "parent_kind": "thread",
            "parent_id": THREAD_ID,
            "thread_id": THREAD_ID,
            "resource_kind": "comment",
            "comment_id": COMMENT_ID,
            "actor": ACTOR,
            "body": BODY,
            "signature": SIGNATURE,
            "visible": True,
            "match_count": 1,
        },
        signature=SIGNATURE,
    )


def writer_context_observation() -> ReadbackObservation:
    """Return a deliberately reused writer-context observation for rejection."""

    observation = fresh_anonymous_observation()
    payload = dict(observation.payload)
    payload["reader_context"] = "regular_participant_writer"
    payload["writer_cookie_names"] = ("session",)
    return ReadbackObservation(
        observation.kind,
        observation.identity_tokens,
        payload,
        signature=observation.signature,
    )
