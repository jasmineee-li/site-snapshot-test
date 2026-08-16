"""Pure ordinary-reader readback interpretation for Classifieds replies.

The executor owns fetching and rendering the page.  This module only accepts a
typed observation collected by an independent reader and checks every identity
and visibility witness.  It never queries a database, admin endpoint, or
``latest``/``newest`` listing.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.sites.classifieds_reply_html import (
    _block_actor_matches,
    _block_data_id,
    _normalize_text,
    _RenderedCommentParser,
    normalize_reply_body,
)
from warp_taskgen.sites.readback import (
    ReadbackDecision,
    ReadbackFailure,
    ReadbackObservation,
    identity_token_text,
)


def _identity(mapping: Mapping[str, Any], key: str) -> str | None:
    return identity_token_text(mapping.get(key))


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _payload_identity(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _identity(payload, key)
        if value is not None:
            return value
    return None


def _is_independent_reader(payload: Mapping[str, Any]) -> bool:
    marker = payload.get("independent_reader")
    if marker is True:
        return True
    marker = payload.get("independent_reader_role", payload.get("reader_role"))
    if isinstance(marker, str) and marker.strip().casefold() in {
        "independent",
        "independent_reader",
        "ordinary_reader",
    }:
        return True
    reader = payload.get("reader")
    return isinstance(reader, Mapping) and reader.get("independent") is True


def _is_visible(payload: Mapping[str, Any]) -> bool:
    marker = payload.get("visible")
    if marker is True:
        return True
    marker = payload.get("visible_to_independent_reader", payload.get("independent_reader_visible"))
    return marker is True


class ClassifiedsReadbackCapability:
    """Interpret exact listing-reply identity from a typed observation."""

    def readback_visibility_selector(self, plan: Any) -> str | ReadbackFailure:
        identity = getattr(plan, "identity_tokens", None)
        reply_id = _identity(identity, "reply_id") if isinstance(identity, Mapping) else None
        if reply_id is None or not reply_id.isdigit() or int(reply_id) <= 0:
            return ReadbackFailure(
                "classifieds",
                "missing_reply_visibility_identity",
                "Classifieds visibility needs one positive reply id",
            )
        # The painted witness must be the reply body itself, not merely the
        # adjacent reply-action link that carries the stable resource id.
        # ``:has`` binds that body to the same rendered comment block.
        return f'div.comment:has(a.comment-reply[data-id="{reply_id}"]) > p:not([class])'

    def observe_readback_html(
        self,
        html: str,
        plan: Any,
    ) -> ReadbackObservation | ReadbackFailure:
        """Extract one exact reply from an ordinary-reader listing page.

        The render executor supplies HTML from a fresh browser context. The
        existing feature-local comment parser identifies the outer comment
        block, actor, body, and descendant reply id; this method only projects
        that result into the typed readback observation consumed below.
        """

        if not isinstance(html, str) or not html.strip():
            return ReadbackFailure(
                "classifieds", "malformed_readback_html", "ordinary-reader page HTML is empty"
            )
        identity = getattr(plan, "identity_tokens", None)
        signature = getattr(plan, "signature", None)
        if not isinstance(identity, Mapping) or not isinstance(signature, str) or not signature:
            return ReadbackFailure(
                "classifieds",
                "missing_readback_plan",
                "Classifieds readback needs identity tokens and a signature",
            )
        listing_id = _identity(identity, "listing_id")
        reply_id = _identity(identity, "reply_id")
        actor_name = _payload_identity(identity, "actor_name", "actor")
        if not all((listing_id, reply_id, actor_name)):
            return ReadbackFailure(
                "classifieds",
                "missing_listing_reply_identity",
                "Classifieds readback needs listing, reply, and actor identities",
            )

        parser = _RenderedCommentParser()
        try:
            parser.feed(html)
            parser.close()
        except (TypeError, ValueError) as exc:
            return ReadbackFailure(
                "classifieds",
                "malformed_readback_html",
                f"Classifieds comment HTML could not be parsed: {exc}",
            )

        matches: list[dict[str, str]] = []
        normalized_signature = _normalize_text(signature)
        for block in parser.blocks:
            if _block_data_id(block) != reply_id or not _block_actor_matches(block, actor_name):
                continue
            rendered_listing = block.attrs.get("data-listing-id") or block.attrs.get("data-item-id")
            if rendered_listing and rendered_listing.strip() != listing_id:
                continue
            body = normalize_reply_body(" ".join(block.body_text))
            if not body or normalized_signature not in body:
                continue
            matches.append(
                {
                    "listing_id": listing_id,
                    "reply_id": reply_id,
                    "actor_name": actor_name,
                    "body": body,
                    "signature": normalized_signature,
                }
            )
        if len(matches) != 1:
            reason = "reply_not_found" if not matches else "ambiguous_reply_identity"
            return ReadbackFailure(
                "classifieds",
                reason,
                "ordinary-reader HTML did not expose exactly one matching listing reply",
            )
        payload = matches[0]
        # The fresh render-check context is the ordinary independent reader.
        # The browser executor owns navigation and viewport admission; this
        # feature hook reports a visible matching block in that DOM.
        payload.update({"independent_reader": True, "visible": True})
        return ReadbackObservation(
            kind="comment_visibility",
            identity_tokens=identity,
            payload=payload,
            signature=normalized_signature,
        )

    def interpret_readback(self, observation: ReadbackObservation) -> ReadbackDecision:
        if not isinstance(observation, ReadbackObservation):
            return ReadbackDecision(False, "malformed_observation")
        if observation.kind != "comment_visibility":
            return ReadbackDecision(False, "unsupported_readback_kind")
        expected = observation.identity_tokens
        if not isinstance(expected, Mapping):
            return ReadbackDecision(False, "malformed_identity")
        expected_listing = _identity(expected, "listing_id")
        expected_reply = _identity(expected, "reply_id")
        expected_actor = _payload_identity(expected, "actor_name", "actor")
        if not all((expected_listing, expected_reply, expected_actor)):
            return ReadbackDecision(False, "missing_listing_reply_identity")

        payload = observation.payload
        if not isinstance(payload, Mapping):
            return ReadbackDecision(False, "malformed_payload")
        rendered_listing = _payload_identity(payload, "listing_id", "item_id")
        rendered_reply = _payload_identity(payload, "reply_id", "comment_id")
        rendered_actor = _payload_identity(
            payload,
            "actor_name",
            "actor",
            "author",
            "username",
        )
        if rendered_listing != expected_listing:
            return ReadbackDecision(False, "listing_identity_mismatch")
        if rendered_reply != expected_reply:
            return ReadbackDecision(False, "reply_identity_mismatch")
        if rendered_actor != expected_actor:
            return ReadbackDecision(False, "reply_actor_mismatch")
        if not _is_independent_reader(payload):
            return ReadbackDecision(False, "not_independent_reader")
        if not _is_visible(payload):
            return ReadbackDecision(False, "reply_not_visible")

        signature = observation.signature.strip() if isinstance(observation.signature, str) else ""
        if not signature:
            return ReadbackDecision(False, "missing_signature")
        rendered_signature = payload.get("signature", payload.get("rendered_signature"))
        if not isinstance(rendered_signature, str) or not rendered_signature.strip():
            return ReadbackDecision(False, "rendered_signature_missing")
        if rendered_signature.strip() != signature:
            return ReadbackDecision(False, "signature_mismatch")

        body = payload.get("body")
        if not isinstance(body, str) or not body.strip():
            return ReadbackDecision(False, "reply_body_missing")
        expected_body_digest = _identity(expected, "reply_body_sha256")
        if expected_body_digest is None:
            return ReadbackDecision(False, "missing_body_evidence")
        if _SHA256_HEX.fullmatch(expected_body_digest) is None:
            return ReadbackDecision(False, "malformed_body_evidence")
        rendered_body_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if rendered_body_digest != expected_body_digest:
            return ReadbackDecision(False, "reply_body_mismatch")
        if signature not in body:
            return ReadbackDecision(False, "signature_not_in_reply_body")
        return ReadbackDecision(
            True,
            "exact_listing_reply_visible",
            matched_signature=signature,
            rendered_text=body,
        )


def interpret_readback(observation: ReadbackObservation) -> ReadbackDecision:
    """Functional convenience facade for feature-local callers."""

    return ClassifiedsReadbackCapability().interpret_readback(observation)


__all__ = [
    "ClassifiedsReadbackCapability",
    "interpret_readback",
]
