"""Pure GitLab readback interpretation."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from worldsim.sites.readback import ReadbackDecision, ReadbackObservation, identity_token_text

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def note_html_for_id(payload: Any, note_id: str) -> tuple[bool, str | None]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return False, None
    for discussion in payload:
        if not isinstance(discussion, Mapping):
            continue
        notes = discussion.get("notes")
        if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes)):
            continue
        for note in notes:
            if isinstance(note, Mapping) and str(note.get("id")) == note_id:
                value = note.get("note_html")
                return True, value if isinstance(value, str) else None
    return False, None


def rendered_note_text(note_html: str) -> str:
    """Return the normalized visible text represented by GitLab note HTML."""

    rendered = html.unescape(_HTML_TAG_RE.sub(" ", note_html))
    return re.sub(r"\s+", " ", rendered).strip()


class GitLabReadbackCapability:
    def interpret_readback(self, observation: ReadbackObservation) -> ReadbackDecision:
        if observation.kind == "resource_identity":
            note_id_text = identity_token_text(observation.identity_tokens.get("note_id"))
            if note_id_text is None or not isinstance(observation.payload, str):
                return ReadbackDecision(False, "missing_note_identity")
            try:
                payload = json.loads(observation.payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                return ReadbackDecision(False, "invalid_note_payload")
            found, note_html = note_html_for_id(payload, note_id_text)
            if not found:
                return ReadbackDecision(False, "note_identity_absent")
            rendered = None
            if note_html is not None:
                rendered = rendered_note_text(note_html)
            return ReadbackDecision(
                True,
                "note_identity_present",
                matched_signature=f"note_id={note_id_text}",
                rendered_text=rendered,
            )
        if observation.kind == "resource_signature":
            expected_identity = {
                key: identity_token_text(observation.identity_tokens.get(key))
                for key in ("project_id", "issue_iid")
            }
            if any(value is None for value in expected_identity.values()):
                return ReadbackDecision(False, "missing_resource_identity")
            if not isinstance(observation.payload, Mapping):
                return ReadbackDecision(False, "invalid_resource_payload")
            description = observation.payload.get("normalized_description")
            signature = observation.payload.get("normalized_signature")
            if not isinstance(description, str) or not isinstance(signature, str) or not signature:
                return ReadbackDecision(False, "missing_resource_signature")
            if observation.signature != signature:
                return ReadbackDecision(False, "resource_signature_mismatch")
            for token_key, payload_key in (("project_id", "project_id"), ("issue_iid", "iid")):
                observed = observation.payload.get(payload_key)
                if observed not in (None, ""):
                    observed_text = identity_token_text(observed)
                    if observed_text is None or observed_text != expected_identity[token_key]:
                        return ReadbackDecision(False, "resource_identity_mismatch")
            if signature not in description:
                return ReadbackDecision(False, "resource_signature_absent")
            return ReadbackDecision(True, "resource_signature_present")
        return ReadbackDecision(False, "unsupported_readback_kind")


__all__ = ["GitLabReadbackCapability", "note_html_for_id", "rendered_note_text"]
