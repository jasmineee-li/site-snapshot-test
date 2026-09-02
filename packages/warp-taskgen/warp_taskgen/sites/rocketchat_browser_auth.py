"""Feature-local Rocket.Chat browser-reader authentication preflight."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatContractError
from warp_taskgen.sites.rocketchat_transport import (
    _credentials,
    _declared_reader_user_id,
    _identifier,
    _origin,
)


def _canonical_origin(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value.strip().rstrip("/"))
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _storage_state_reader_user_id(
    instance: Mapping[str, Any], path: str, expected_user_id: str | None
) -> str:
    """Extract the exact Rocket.Chat ``Meteor.userId`` from one state file."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise RocketChatContractError(
            f"Rocket.Chat reader storage_state is unreadable: {exc.__class__.__name__}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RocketChatContractError("Rocket.Chat reader storage_state must be an object")
    origins = payload.get("origins")
    if not isinstance(origins, list):
        raise RocketChatContractError("Rocket.Chat reader storage_state has no origins list")
    expected_origin = _canonical_origin(_origin(instance))
    matching_origins = [
        origin
        for origin in origins
        if isinstance(origin, Mapping)
        and _canonical_origin(origin.get("origin")) == expected_origin
    ]
    if len(matching_origins) != 1:
        raise RocketChatContractError(
            "Rocket.Chat reader storage_state must contain exactly one matching site origin"
        )
    local_storage = matching_origins[0].get("localStorage")
    if not isinstance(local_storage, list):
        raise RocketChatContractError(
            "Rocket.Chat reader storage_state matching origin has no localStorage list"
        )
    entries = [
        entry
        for entry in local_storage
        if isinstance(entry, Mapping) and entry.get("name") == "Meteor.userId"
    ]
    if len(entries) != 1:
        raise RocketChatContractError(
            "Rocket.Chat reader storage_state must contain exactly one Meteor.userId"
        )
    actual = _identifier(entries[0].get("value"), "reader storage user id")
    if expected_user_id is not None and actual != expected_user_id:
        raise RocketChatContractError(
            "Rocket.Chat reader storage user ID does not match reader_auth declaration"
        )
    return actual


@dataclass(frozen=True)
class RocketChatReaderPreflight:
    ok: bool
    reason: str | None = None
    detail: str = ""
    browser_context_kwargs: Mapping[str, Any] = MappingProxyType({})
    reader_user_id: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.reason is not None:
            raise ValueError("successful Rocket.Chat preflight cannot have a reason")
        if not self.ok and not self.reason:
            raise ValueError("failed Rocket.Chat preflight needs a reason")
        object.__setattr__(
            self, "browser_context_kwargs", MappingProxyType(dict(self.browser_context_kwargs))
        )
        if self.reader_user_id is not None:
            object.__setattr__(
                self,
                "reader_user_id",
                _identifier(self.reader_user_id, "reader browser user id"),
            )
        if self.ok and self.reader_user_id is None:
            raise ValueError("successful Rocket.Chat preflight needs reader user identity")

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "reader_auth": "independent_authenticated_ordinary_reader",
            "fresh_context_required": True,
            "writer_context_reuse_forbidden": True,
        }
        if self.reader_user_id is not None:
            metadata["reader_user_id"] = self.reader_user_id
        return metadata


def preflight_rocket_chat_reader(instance: Mapping[str, Any]) -> RocketChatReaderPreflight:
    if not isinstance(instance, Mapping):
        return RocketChatReaderPreflight(
            False, "invalid_instance", "reader instance must be a mapping"
        )
    declared = instance.get("reader_auth")
    if not isinstance(declared, Mapping):
        return RocketChatReaderPreflight(
            False, "missing_reader_auth", "Rocket.Chat requires explicit reader_auth"
        )
    try:
        _credentials(instance, "reader")
    except RocketChatContractError as exc:
        return RocketChatReaderPreflight(False, "reader_credentials_missing", str(exc))
    auth_type = str(declared.get("type") or "credentials").strip().lower()
    try:
        expected_user_id = _declared_reader_user_id(instance)
    except RocketChatContractError as exc:
        return RocketChatReaderPreflight(False, "reader_identity_invalid", str(exc))
    if auth_type == "storage_state":
        state = declared.get("storage_state", declared.get("path"))
        path = state.get("path") if isinstance(state, Mapping) else state
        if not isinstance(path, str) or not path.strip() or not os.path.exists(path):
            return RocketChatReaderPreflight(
                False, "reader_storage_missing", "reader storage_state path is unavailable"
            )
        if expected_user_id is None:
            return RocketChatReaderPreflight(
                False,
                "reader_storage_identity_missing",
                "reader_auth.user_id or expected_reader_user_id is required with storage_state",
            )
        try:
            browser_user_id = _storage_state_reader_user_id(instance, path, expected_user_id)
        except RocketChatContractError as exc:
            return RocketChatReaderPreflight(False, "reader_storage_identity_invalid", str(exc))
        return RocketChatReaderPreflight(
            True,
            browser_context_kwargs={"storage_state": path},
            reader_user_id=browser_user_id,
        )
    if auth_type == "http_headers":
        headers = declared.get("headers")
        if not isinstance(headers, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in headers.items()
        ):
            return RocketChatReaderPreflight(
                False, "reader_headers_invalid", "reader headers must be non-empty strings"
            )
        if not {"X-Auth-Token", "X-User-Id"}.issubset(headers):
            return RocketChatReaderPreflight(
                False,
                "reader_headers_incomplete",
                "reader headers require X-Auth-Token and X-User-Id",
            )
        header_user_id = _identifier(headers["X-User-Id"], "reader header user id")
        if expected_user_id is not None and expected_user_id != header_user_id:
            return RocketChatReaderPreflight(
                False,
                "reader_identity_mismatch",
                "reader_auth user ID does not match X-User-Id",
            )
        return RocketChatReaderPreflight(
            True,
            browser_context_kwargs={"extra_http_headers": dict(headers)},
            reader_user_id=header_user_id,
        )
    if auth_type == "credentials":
        return RocketChatReaderPreflight(
            False,
            "reader_browser_auth_unavailable",
            "REST credentials need reader storage_state or auth headers for browser rendering",
        )
    return RocketChatReaderPreflight(
        False, "reader_auth_unsupported", f"unsupported reader_auth type {auth_type!r}"
    )
