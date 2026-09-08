"""Contract tests for the connection-only host restoration consumer."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.host_restoration import (
    HostRestorationClient,
    HostRestorationError,
    restoration_operation_id,
)


def _short_socket_path() -> Path:
    """Keep macOS AF_UNIX paths below its short pathname limit."""

    return Path("/tmp") / f"warp-owner-{uuid4().hex[:12]}.sock"


def _serve_once(
    socket_path: Path,
    response_factory: Any,
    *,
    request_count: int = 1,
) -> tuple[threading.Thread, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    ready = threading.Event()

    def run() -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(socket_path))
                server.listen()
                ready.set()
                for index in range(request_count):
                    conn, _ = server.accept()
                    with conn:
                        data = b""
                        while b"\n" not in data:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            data += chunk
                        request = json.loads(data.split(b"\n", 1)[0])
                        seen.append(request)
                        response = response_factory(request, index)
                        if response is not None:
                            conn.sendall((json.dumps(response) + "\n").encode())
        finally:
            socket_path.unlink(missing_ok=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    if not ready.wait(timeout=2):
        raise RuntimeError("test owner socket did not become ready")
    return thread, seen


def _client(socket_path: Path) -> HostRestorationClient:
    return HostRestorationClient(
        socket_path,
        instance_id="gitlab-r23",
        site_url="http://gitlab.internal:8929",
        timeout_s=1,
        transport_retries=1,
    )


def test_config_binding_is_optional_and_typed(tmp_path: Path) -> None:
    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="http://gitlab.internal:8929",
        restoration={"socket_path": str(tmp_path / "owner.sock"), "instance_id": "gitlab-r23"},
    )
    assert instance.restoration is not None
    assert instance.restoration.instance_id == "gitlab-r23"
    assert "restoration" not in BenchmarkInstance(
        site_name="gitlab", site_url="http://gitlab.internal:8929"
    ).model_dump(exclude_none=True)


def test_client_accepts_terminal_owner_responses(tmp_path: Path) -> None:
    socket_path = _short_socket_path()

    def response(request: dict[str, Any], _index: int) -> dict[str, Any]:
        common = {"instance_id": request["instance_id"], "site_url": request["site_url"]}
        if request["action"] == "acquire":
            return {
                **common,
                "status": "acquired",
                "scope_id": request["scope_id"],
                "lease_token": "opaque-token",
            }
        if request["action"] == "restore":
            return {
                **common,
                "status": "restored",
                "operation_id": request["operation_id"],
                "before_container_id": "old-container",
                "after_container_id": "new-container",
                "image_id": "sha256:" + "a" * 64,
            }
        return {**common, "status": "released", "operation_id": request["operation_id"]}

    thread, seen = _serve_once(socket_path, response, request_count=3)
    client = _client(socket_path)
    scope_id = "12345678-1234-5678-1234-567812345678"
    acquired = client.acquire(scope_id)
    operation = restoration_operation_id()
    restored = client.restore(scope_id=scope_id, lease_token="opaque-token", operation_id=operation)
    released = client.release(scope_id=scope_id, lease_token="opaque-token", operation_id=operation)
    thread.join(timeout=2)

    assert acquired["status"] == "acquired"
    assert restored["status"] == "restored"
    assert released["status"] == "released"
    assert all("auth" not in request and "cookie" not in request for request in seen)
    assert seen[1]["operation_id"] == seen[2]["operation_id"]


def test_client_retries_same_operation_after_lost_response(tmp_path: Path) -> None:
    socket_path = _short_socket_path()

    def response(request: dict[str, Any], index: int) -> dict[str, Any] | None:
        if index == 0:
            return None
        return {
            "status": "restored",
            "instance_id": request["instance_id"],
            "site_url": request["site_url"],
            "operation_id": request["operation_id"],
            "before_container_id": "old",
            "after_container_id": "new",
            "image_id": "b" * 64,
        }

    thread, seen = _serve_once(socket_path, response, request_count=2)
    client = _client(socket_path)
    operation = restoration_operation_id()
    restored = client.restore(
        scope_id="12345678-1234-5678-1234-567812345678",
        lease_token="opaque-token",
        operation_id=operation,
    )
    thread.join(timeout=2)
    assert restored["status"] == "restored"
    # A lost response is retried with the same operation ID; no new reset is
    # invented for the bounded transport retry.
    assert len(seen) == 2
    assert seen[0]["operation_id"] == seen[1]["operation_id"] == operation


@pytest.mark.parametrize(
    "response",
    [
        {"status": "ok"},
        {"status": "error", "reason": "plain_http_success"},
    ],
)
def test_client_rejects_non_terminal_or_owner_error(
    tmp_path: Path, response: dict[str, Any]
) -> None:
    socket_path = _short_socket_path()
    thread, _ = _serve_once(socket_path, lambda _request, _index: response)
    with pytest.raises(HostRestorationError):
        _client(socket_path).acquire("12345678-1234-5678-1234-567812345678")
    thread.join(timeout=2)


def test_client_rejects_wrong_identity_and_non_recreation(tmp_path: Path) -> None:
    socket_path = _short_socket_path()

    def wrong_identity(request: dict[str, Any], _index: int) -> dict[str, Any]:
        return {
            "status": "restored",
            "instance_id": "other-replica",
            "site_url": request["site_url"],
            "operation_id": request["operation_id"],
            "before_container_id": "same",
            "after_container_id": "same",
            "image_id": "not-a-sha",
        }

    thread, _ = _serve_once(socket_path, wrong_identity)
    with pytest.raises(HostRestorationError, match="different instance"):
        _client(socket_path).restore(
            scope_id="12345678-1234-5678-1234-567812345678",
            lease_token="opaque-token",
            operation_id=restoration_operation_id(),
        )
    thread.join(timeout=2)
