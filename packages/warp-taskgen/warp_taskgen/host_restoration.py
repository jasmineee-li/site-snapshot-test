"""Connection-only consumer for a host-owned benchmark restoration daemon.

The benchmark host owns reset authority and container verification.
Taskgen separately checks a fixed application sample after recreation.  Taskgen
only opens the configured Unix-domain socket, sends a small newline-delimited
JSON request, and accepts a terminal owner response.  This module never sends
browser cookies, benchmark credentials, bearer tokens, or a browser URL as a
target selector.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MAX_MESSAGE_BYTES = 64 * 1024
_DEFAULT_TIMEOUT_S = 300.0
_DEFAULT_TRANSPORT_RETRIES = 2


class HostRestorationError(RuntimeError):
    """Fail-closed error returned by the local restoration consumer."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"host restoration {reason_code}{suffix}")


def _non_empty(value: object, *, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise HostRestorationError("invalid_request", f"{field} is required")
    if len(text) > 512:
        raise HostRestorationError("invalid_request", f"{field} is too long")
    return text


def _uuid_text(value: object, *, field: str) -> str:
    text = _non_empty(value, field=field)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HostRestorationError("invalid_request", f"{field} must be a UUID") from exc
    return str(parsed)


def restoration_scope_id() -> str:
    """Return a fresh UUID for one live restoration lease scope."""

    return str(uuid.uuid4())


def restoration_operation_id() -> str:
    """Return a fresh ID for one intentional restore operation.

    Transport retries happen inside ``HostRestorationClient._request`` with
    the exact same serialized body, so an uncertain response never calls this
    helper a second time.
    """

    return f"warp-reset-{uuid.uuid4()}"


def _restoration_mapping(instance: object) -> Mapping[str, Any] | None:
    if isinstance(instance, Mapping):
        raw = instance.get("restoration")
    else:
        raw = getattr(instance, "restoration", None)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return raw
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise HostRestorationError("invalid_binding", "restoration binding must be an object")


def restoration_binding_for_instance(instance: object) -> tuple[Path, str] | None:
    """Return the configured socket and instance ID, or ``None`` for legacy instances."""

    raw = _restoration_mapping(instance)
    if raw is None:
        return None
    socket_path = Path(_non_empty(raw.get("socket_path"), field="restoration.socket_path"))
    instance_id = _non_empty(raw.get("instance_id"), field="restoration.instance_id")
    return socket_path, instance_id


def _site_url_for_instance(instance: object) -> str:
    if isinstance(instance, Mapping):
        value = instance.get("site_url")
    else:
        value = getattr(instance, "site_url", None)
    return _non_empty(value, field="site_url")


class HostRestorationClient:
    """Small synchronous AF_UNIX client used through ``asyncio.to_thread``."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        instance_id: str,
        site_url: str,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport_retries: int = _DEFAULT_TRANSPORT_RETRIES,
    ) -> None:
        path = Path(_non_empty(socket_path, field="socket_path"))
        if not path.is_absolute():
            raise HostRestorationError("invalid_binding", "socket_path must be absolute")
        if timeout_s <= 0 or timeout_s > 300:
            raise HostRestorationError("invalid_binding", "timeout_s is out of bounds")
        if transport_retries < 0 or transport_retries > 3:
            raise HostRestorationError("invalid_binding", "transport_retries is out of bounds")
        self.socket_path = path
        self.instance_id = _non_empty(instance_id, field="instance_id")
        self.site_url = _non_empty(site_url, field="site_url")
        self.timeout_s = float(timeout_s)
        self.transport_retries = int(transport_retries)

    def _request(self, payload: Mapping[str, object]) -> dict[str, Any]:
        try:
            encoded = (
                json.dumps(dict(payload), separators=(",", ":"), sort_keys=True) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise HostRestorationError(
                "invalid_request", "request is not JSON serializable"
            ) from exc
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise HostRestorationError("invalid_request", "request exceeds 64 KiB")

        last_error: BaseException | None = None
        for attempt in range(self.transport_retries + 1):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                    channel.settimeout(self.timeout_s)
                    channel.connect(str(self.socket_path))
                    channel.sendall(encoded)
                    response = bytearray()
                    while True:
                        chunk = channel.recv(4096)
                        if not chunk:
                            break
                        response.extend(chunk)
                        if len(response) > _MAX_MESSAGE_BYTES:
                            raise HostRestorationError(
                                "invalid_response", "response exceeds 64 KiB"
                            )
                        if b"\n" in chunk:
                            break
                line = bytes(response).split(b"\n", 1)[0].strip()
                if not line:
                    raise OSError("owner returned an empty response")
                raw = json.loads(line.decode("utf-8"))
                if not isinstance(raw, dict):
                    raise HostRestorationError(
                        "invalid_response", "owner response must be an object"
                    )
                if raw.get("status") == "error":
                    reason = raw.get("reason")
                    raise HostRestorationError(
                        str(reason or "owner_rejected"),
                        "owner rejected restoration request",
                    )
                return raw
            except HostRestorationError:
                raise
            except (OSError, TimeoutError, UnicodeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.transport_retries:
                    break
        detail = type(last_error).__name__ if last_error is not None else "unknown"
        raise HostRestorationError("transport_unavailable", detail) from last_error

    def acquire(self, scope_id: str) -> dict[str, Any]:
        scope = _uuid_text(scope_id, field="scope_id")
        response = self._request(
            {
                "action": "acquire",
                "instance_id": self.instance_id,
                "site_url": self.site_url,
                "scope_id": scope,
            }
        )
        if response.get("status") != "acquired":
            raise HostRestorationError("invalid_response", "owner did not acquire a lease")
        if response.get("scope_id") != scope:
            raise HostRestorationError("invalid_response", "owner returned a different scope")
        self._validate_identity(response)
        _non_empty(response.get("lease_token"), field="lease_token")
        return response

    def restore(self, *, scope_id: str, lease_token: str, operation_id: str) -> dict[str, Any]:
        scope = _uuid_text(scope_id, field="scope_id")
        operation = _non_empty(operation_id, field="operation_id")
        response = self._request(
            {
                "action": "restore",
                "instance_id": self.instance_id,
                "site_url": self.site_url,
                "scope_id": scope,
                "lease_token": _non_empty(lease_token, field="lease_token"),
                "operation_id": operation,
            }
        )
        if response.get("status") != "restored":
            raise HostRestorationError("invalid_response", "owner did not restore the instance")
        if response.get("operation_id") != operation:
            raise HostRestorationError("invalid_response", "owner returned a different operation")
        self._validate_identity(response)
        before = _non_empty(response.get("before_container_id"), field="before_container_id")
        after = _non_empty(response.get("after_container_id"), field="after_container_id")
        if before == after:
            raise HostRestorationError("invalid_response", "owner did not recreate the instance")
        image_id = _non_empty(response.get("image_id"), field="image_id")
        if not re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", image_id):
            raise HostRestorationError("invalid_response", "owner image_id is not a SHA-256 digest")
        return response

    def release(self, *, scope_id: str, lease_token: str, operation_id: str) -> dict[str, Any]:
        scope = _uuid_text(scope_id, field="scope_id")
        operation = _non_empty(operation_id, field="operation_id")
        response = self._request(
            {
                "action": "release",
                "instance_id": self.instance_id,
                "site_url": self.site_url,
                "scope_id": scope,
                "lease_token": _non_empty(lease_token, field="lease_token"),
                "operation_id": operation,
            }
        )
        if response.get("status") != "released":
            raise HostRestorationError("invalid_response", "owner did not release the lease")
        if response.get("operation_id") != operation:
            raise HostRestorationError("invalid_response", "owner released a different operation")
        self._validate_identity(response)
        return response

    def _validate_identity(self, response: Mapping[str, Any]) -> None:
        if response.get("instance_id") != self.instance_id:
            raise HostRestorationError("invalid_response", "owner returned a different instance")
        if response.get("site_url") != self.site_url:
            raise HostRestorationError("invalid_response", "owner returned a different site URL")


@dataclass(slots=True)
class HostRestorationScope:
    """One acquired owner lease held across a complete task or matched pair."""

    client: HostRestorationClient
    scope_id: str
    lease_token: str = field(repr=False)
    _last_operation_id: str | None = None
    _released: bool = False

    def matches_instance(self, instance: object) -> bool:
        """Return whether this lease is pinned to the supplied bound instance."""

        try:
            binding = restoration_binding_for_instance(instance)
            return binding == (
                self.client.socket_path,
                self.client.instance_id,
            ) and self.client.site_url == _site_url_for_instance(instance)
        except (HostRestorationError, TypeError, IndexError):
            return False

    def operation_id(self) -> str:
        return restoration_operation_id()

    async def restore(self, operation_id: str) -> dict[str, Any]:
        if self._released:
            raise HostRestorationError("lease_released")
        response = await asyncio.to_thread(
            self.client.restore,
            scope_id=self.scope_id,
            lease_token=self.lease_token,
            operation_id=operation_id,
        )
        self._last_operation_id = operation_id
        return response

    async def release(self, *, operation_id: str | None = None) -> dict[str, Any]:
        if self._released:
            return {"status": "released", "scope_id": self.scope_id}
        final_operation = operation_id or self._last_operation_id
        if final_operation is None:
            raise HostRestorationError("release_without_restore")
        response = await asyncio.to_thread(
            self.client.release,
            scope_id=self.scope_id,
            lease_token=self.lease_token,
            operation_id=final_operation,
        )
        self._released = True
        return response


def _build_client(instance: object) -> HostRestorationClient | None:
    binding = restoration_binding_for_instance(instance)
    if binding is None:
        return None
    return HostRestorationClient(
        binding[0],
        instance_id=binding[1],
        site_url=_site_url_for_instance(instance),
    )


async def acquire_restoration_scope(
    instance: object,
    *,
    scope_id: str,
) -> HostRestorationScope | None:
    """Acquire a configured host lease; return ``None`` for legacy instances."""

    client = _build_client(instance)
    if client is None:
        return None
    scope = _uuid_text(scope_id, field="scope_id")
    response = await asyncio.to_thread(client.acquire, scope)
    return HostRestorationScope(
        client=client,
        scope_id=scope,
        lease_token=str(response["lease_token"]),
    )


def restoration_enabled(instance: object) -> bool:
    """Return whether this instance is managed by the host restoration owner."""

    return restoration_binding_for_instance(instance) is not None


def validate_task_restoration_topology(task: Mapping[str, Any]) -> None:
    """Reject managed multi-instance tasks until a concrete scope group exists."""

    runtime = task.get("_worldsim_runtime")
    if not isinstance(runtime, Mapping):
        return
    bound = runtime.get("bound_instances")
    if not isinstance(bound, Mapping):
        return
    managed = [payload for payload in bound.values() if restoration_enabled(payload)]
    if len(bound) > 1 and managed:
        raise HostRestorationError(
            "unsupported_multi_instance",
            "managed restoration requires one concrete instance scope per task",
        )


__all__ = [
    "HostRestorationClient",
    "HostRestorationError",
    "HostRestorationScope",
    "acquire_restoration_scope",
    "restoration_binding_for_instance",
    "restoration_enabled",
    "restoration_operation_id",
    "restoration_scope_id",
    "validate_task_restoration_topology",
]
