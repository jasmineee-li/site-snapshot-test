"""Synchronous, host-owned reset seam for the Rocket.Chat slice.

The ordinary writer never receives reset authority.  A reset endpoint is
therefore an explicit instance-level owner (or an injected fake in tests),
called synchronously by the feature editor after a possible mutation.  No
fallback to writer credentials, database access, or process/container
lifecycle is permitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import requests

from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatContractError


@runtime_checkable
class RocketChatResetter(Protocol):
    """One synchronous host reset operation."""

    def reset(self) -> None: ...


def _reset_headers(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RocketChatContractError("Rocket.Chat reset headers must be a mapping")
    headers: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            raise RocketChatContractError("Rocket.Chat reset headers require string names")
        if not isinstance(raw, str) or not raw:
            raise RocketChatContractError("Rocket.Chat reset headers require non-empty strings")
        headers[key.strip()] = raw
    return headers


class RequestsRocketChatResetter:
    """POST one explicitly configured reset endpoint with a separate session."""

    def __init__(
        self,
        endpoint: str,
        session: Any | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        raw = str(endpoint or "").strip()
        parsed = urlsplit(raw)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise RocketChatContractError(
                "Rocket.Chat reset_endpoint requires an explicit HTTP(S) URL without query/fragment"
            )
        self.endpoint = raw
        self.session = session or requests.Session()
        self.headers = _reset_headers(headers)
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0 or self.timeout_s > 120:
            raise RocketChatContractError("Rocket.Chat reset timeout is out of bounds")

    def reset(self) -> None:
        try:
            response = self.session.request(
                "POST",
                self.endpoint,
                headers=dict(self.headers),
                timeout=self.timeout_s,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Rocket.Chat reset_endpoint POST failed: {exc.__class__.__name__}"
            ) from exc
        status = getattr(response, "status_code", None)
        if isinstance(status, bool) or not isinstance(status, int):
            raise RuntimeError("Rocket.Chat reset_endpoint returned no HTTP status")
        if 300 <= status < 400:
            raise RuntimeError("Rocket.Chat reset_endpoint returned an unexpected redirect")
        # TAC's native reset API returns 202 while Redis/Mongo work continues;
        # that is not a safe cleanup boundary.  Only a host-owned terminal
        # 200 response with an explicit completion marker can unblock the next
        # Atomic Work Unit.
        if status == 202:
            raise RuntimeError(
                "Rocket.Chat reset_endpoint returned HTTP 202 (asynchronous reset is not complete)"
            )
        if status != 200:
            raise RuntimeError(f"Rocket.Chat reset_endpoint returned HTTP {status}")
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Rocket.Chat reset_endpoint returned no completion schema") from exc
        if not isinstance(payload, Mapping) or not (
            payload.get("ok") is True
            or payload.get("completed") is True
            or payload.get("status") in {"ok", "success", "complete", "completed"}
        ):
            raise RuntimeError(
                "Rocket.Chat reset_endpoint response does not report terminal completion"
            )


def resetter_from_instance(instance: Mapping[str, Any]) -> RocketChatResetter | None:
    """Build a reset owner only from an explicit instance reset endpoint."""

    if not isinstance(instance, Mapping):
        raise RocketChatContractError("Rocket.Chat instance must be a mapping")
    endpoint = instance.get("reset_endpoint")
    if endpoint in (None, ""):
        return None
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise RocketChatContractError("Rocket.Chat reset_endpoint must be a non-empty string")
    # ``reset_headers`` is intentionally separate from writer/reader auth.
    # Supplying it is an explicit host-owner act; no credentials are copied
    # from ``auth`` or ``reader_auth``.
    headers = instance.get("reset_headers")
    return RequestsRocketChatResetter(endpoint, headers=headers)


__all__ = [
    "RequestsRocketChatResetter",
    "RocketChatResetter",
    "resetter_from_instance",
]
