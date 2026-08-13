"""Pure Reddit/Postmill readback interpretation."""

from __future__ import annotations

from collections.abc import Mapping

from warp_taskgen.sites.readback import ReadbackDecision, ReadbackObservation, identity_token_text


class RedditReadbackCapability:
    def interpret_readback(self, observation: ReadbackObservation) -> ReadbackDecision:
        if observation.kind != "comment_visibility":
            return ReadbackDecision(False, "unsupported_readback_kind")
        comment_id = identity_token_text(observation.identity_tokens.get("comment_id"))
        probe = observation.payload
        if comment_id is None or not isinstance(probe, Mapping):
            return ReadbackDecision(False, "missing_comment_identity")
        observed_comment_id = identity_token_text(probe.get("comment_id"))
        if observed_comment_id != comment_id:
            return ReadbackDecision(False, "comment_identity_mismatch")
        first_visible_id = identity_token_text(probe.get("first_visible_comment_id"))
        if first_visible_id != comment_id:
            return ReadbackDecision(False, "different_first_visible_comment")
        if probe.get("match_found_in_comment") is not True:
            return ReadbackDecision(False, "signature_not_in_seed_comment")
        if probe.get("visible_at_entry") is not True:
            return ReadbackDecision(False, "seed_comment_not_visible_at_entry")
        visible_chars = probe.get("visible_nonspace_chars")
        if isinstance(visible_chars, bool) or not isinstance(visible_chars, int):
            return ReadbackDecision(False, "seed_comment_not_painted")
        if visible_chars <= 0:
            return ReadbackDecision(False, "seed_comment_not_painted")
        if probe.get("requires_expand") is not False:
            return ReadbackDecision(False, "seed_comment_requires_expand")
        if probe.get("ok") is not True:
            return ReadbackDecision(False, str(probe.get("reason") or "visibility_unverified"))
        return ReadbackDecision(True, "seed_comment_first_visible_and_painted")


__all__ = ["RedditReadbackCapability"]
