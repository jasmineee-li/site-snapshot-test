from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import os
import secrets
import socket
import stat
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

from .config import load_target, resolved_ports, service_contract, validate_identifier
from .docker import DockerBackend
from .errors import DockerOperationError, DockerTimeoutError, RestorationError
from .protocol import read_request

_OWNER_LOCK_ROOT = Path("/tmp/warp-restoration-locks")


class DockerLike(Protocol):
    def compose_service(self, service: str) -> dict[str, Any]: ...

    def inspect_container(self, name: str) -> dict[str, Any]: ...

    def inspect_image(self, image: str) -> dict[str, Any]: ...

    def recreate(self, service: str) -> None: ...


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise RestorationError("state_write_failed") from exc


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RestorationError("state_corrupt") from exc
    if not isinstance(value, dict):
        raise RestorationError("state_corrupt")
    return value


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalise_mounts(mounts: Any) -> tuple[tuple[str, str, str], ...]:
    """Convert Docker inspect mounts into the resolved Compose shape."""

    if not isinstance(mounts, list):
        raise RestorationError("config_drift")
    result: list[tuple[str, str, str]] = []
    for mount in mounts:
        if not isinstance(mount, dict):
            raise RestorationError("config_drift")
        source = str(mount.get("Source", ""))
        destination = str(mount.get("Destination", ""))
        if not source or not destination:
            raise RestorationError("config_drift")
        read_write = mount.get("RW")
        if isinstance(read_write, bool):
            mode = "rw" if read_write else "ro"
        else:
            raw_mode = str(mount.get("Mode", "rw"))
            mode = "ro" if "ro" in raw_mode.split(",") else "rw"
        result.append((source, destination, mode))
    return tuple(sorted(result, key=lambda item: (item[1], item[0], item[2])))


def _readonly_mount_hashes(
    volumes: tuple[tuple[str, str, str], ...], *, base_dir: Path
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source, destination, mode in volumes:
        if mode != "ro" or destination != "/etc/gitlab/gitlab.rb":
            continue
        path = Path(source)
        if not path.is_absolute():
            path = base_dir / path
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise OSError("config source is not a regular file")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RestorationError("config_drift") from exc
        hashes[destination] = digest
    return hashes


def _actual_port_bindings(container: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    raw = (container.get("NetworkSettings") or {}).get("Ports")
    if not isinstance(raw, dict):
        raise RestorationError("non_loopback_port")
    result: list[tuple[str, int, int]] = []
    for container_port, bindings in raw.items():
        try:
            target = int(str(container_port).split("/", 1)[0])
        except (TypeError, ValueError) as exc:
            raise RestorationError("non_loopback_port") from exc
        if bindings is None:
            # Docker includes EXPOSE-only ports here. An unpublished database
            # port is not a host binding and must remain unpublished.
            continue
        if not isinstance(bindings, list):
            raise RestorationError("non_loopback_port")
        for binding in bindings:
            if not isinstance(binding, dict):
                raise RestorationError("non_loopback_port")
            host = str(binding.get("HostIp", ""))
            try:
                published = int(binding.get("HostPort"))
            except (TypeError, ValueError) as exc:
                raise RestorationError("non_loopback_port") from exc
            result.append((host, published, target))
    return tuple(result)


class RestoreDaemon:
    """One-instance, fail-closed restoration daemon."""

    def __init__(
        self,
        *,
        instances_path: Path,
        compose_path: Path,
        instance_id: str,
        socket_path: Path,
        state_dir: Path,
        docker: DockerLike | None = None,
        readiness_probe: Callable[[], bool] | None = None,
        readiness_timeout: float = 300.0,
        poll_interval: float = 0.25,
    ) -> None:
        self.instances_path = Path(instances_path)
        self.compose_path = Path(compose_path)
        self.instance_id = instance_id
        self.socket_path = Path(socket_path)
        self.state_dir = Path(state_dir)
        self.target = load_target(self.instances_path, instance_id)
        self.docker = docker or DockerBackend(self.compose_path)
        self.readiness_probe = readiness_probe or self._probe_readiness
        if readiness_timeout <= 0 or readiness_timeout > 600:
            raise RestorationError("invalid_readiness_timeout")
        self.readiness_timeout = float(readiness_timeout)
        self.poll_interval = poll_interval
        self.lease_path = self.state_dir / "lease.json"
        self.lock_path = self.state_dir / "daemon.lock"
        self._lock_handle = None
        initial_service = self.docker.compose_service(self.target.service_name)
        self._owner_container_name = validate_identifier(
            initial_service.get("container_name") or self.target.container_name or self.instance_id,
            "config_drift",
        )
        self.owner_lock_path = _OWNER_LOCK_ROOT / f"{self._owner_container_name}.lock"
        self._owner_lock_handle = None
        self._socket: socket.socket | None = None
        self._bound_socket_identity: tuple[int, int] | None = None
        self._ensure_daemon_lock()

    @staticmethod
    def _ensure_private_directory(path: Path, reason: str) -> None:
        try:
            path.lstat()
        except FileNotFoundError:
            try:
                path.mkdir(parents=True, mode=0o700, exist_ok=False)
            except FileExistsError:
                pass
        try:
            info = path.lstat()
        except OSError as exc:
            raise RestorationError(reason) from exc
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise RestorationError(reason)

    @staticmethod
    def _open_lock(path: Path, reason: str):
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise RestorationError(reason) from exc
        handle = os.fdopen(fd, "a+")
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            handle.close()
            raise RestorationError(reason)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RestorationError("daemon_locked") from exc
        return handle

    def _ensure_daemon_lock(self) -> None:
        self._ensure_private_directory(self.state_dir, "state_dir_insecure")
        try:
            self._lock_handle = self._open_lock(self.lock_path, "state_lock_insecure")
            self._ensure_private_directory(_OWNER_LOCK_ROOT, "owner_lock_dir_insecure")
            self._owner_lock_handle = self._open_lock(self.owner_lock_path, "owner_lock_insecure")
        except Exception:
            if self._lock_handle is not None:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
            raise

    def prepare_socket(self) -> socket.socket:
        self._ensure_private_directory(self.socket_path.parent, "socket_dir_insecure")
        try:
            self.socket_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RestorationError("socket_path_exists")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.socket_path))
            info = self.socket_path.lstat()
            self._bound_socket_identity = (info.st_dev, info.st_ino)
            os.chmod(self.socket_path, 0o600)
            server.listen(8)
        except OSError:
            server.close()
            self.close_socket()
            raise
        self._socket = server
        return server

    def close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._bound_socket_identity is not None:
            try:
                info = self.socket_path.lstat()
            except FileNotFoundError:
                info = None
            if info is not None and (info.st_dev, info.st_ino) == self._bound_socket_identity:
                self.socket_path.unlink()
            self._bound_socket_identity = None

    def close(self) -> None:
        self.close_socket()
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None
        if self._owner_lock_handle is not None:
            fcntl.flock(self._owner_lock_handle.fileno(), fcntl.LOCK_UN)
            self._owner_lock_handle.close()
            self._owner_lock_handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def serve_forever(self, *, max_requests: int | None = None) -> None:
        """Serve requests until process shutdown, or a bounded test count."""

        if max_requests is not None and max_requests < 1:
            raise ValueError("max_requests must be positive")
        self.prepare_socket()
        served = 0
        try:
            assert self._socket is not None
            while True:
                conn, _ = self._socket.accept()
                with conn:
                    try:
                        payload = read_request(conn)
                        response = self.handle_request(payload)
                    except RestorationError as exc:
                        response = {"status": "error", "reason": exc.reason}
                    except Exception:
                        response = {"status": "error", "reason": "internal_error"}
                    encoded = (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
                    try:
                        conn.sendall(encoded)
                    except OSError:
                        # A client may time out or disconnect while a restore
                        # is being committed. The operation journal is already
                        # terminal; keep serving and let a retry read it.
                        pass
                served += 1
                if max_requests is not None and served >= max_requests:
                    break
        finally:
            self.close()

    def _probe_readiness(self) -> bool:
        try:
            response = requests.get(self.target.site_url, timeout=5, allow_redirects=False)
            if not (200 <= response.status_code < 400):
                return False
            if self.target.status_url:
                health = requests.get(self.target.status_url, timeout=5, allow_redirects=False)
                return 200 <= health.status_code < 300
            return True
        except requests.RequestException:
            return False

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.readiness_timeout
        while True:
            if self.readiness_probe():
                return
            if time.monotonic() >= deadline:
                raise RestorationError("readiness_timeout")
            time.sleep(self.poll_interval)

    def _service(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        service = self.docker.compose_service(self.target.service_name)
        image_ref = service.get("image") if isinstance(service, dict) else None
        image = self.docker.inspect_image(str(image_ref))
        contract = service_contract(service, image)
        return service, image, contract

    def _container(self, name: str) -> dict[str, Any]:
        return self.docker.inspect_container(name)

    def _validate_instance_shape(
        self,
        service: dict[str, Any],
        contract: dict[str, Any],
        container: dict[str, Any],
    ) -> dict[str, str]:
        configured_ports = resolved_ports(service)
        site_port = urlsplit(self.target.site_url).port
        expected_site_target = {"reddit": 80, "gitlab": 8023}[self.target.site_name]
        site_bindings = [
            published for _, published, target in configured_ports if target == expected_site_target
        ]
        if not site_bindings:
            raise RestorationError("wrong_target")
        if site_port not in site_bindings:
            raise RestorationError("config_drift")
        actual_ports = _actual_port_bindings(container)
        if sorted(actual_ports) != sorted(configured_ports):
            raise RestorationError("config_drift")
        for host, _, _ in actual_ports:
            try:
                if not ipaddress.ip_address(host).is_loopback:
                    raise RestorationError("non_loopback_port")
            except ValueError as exc:
                raise RestorationError("non_loopback_port") from exc
        config = container.get("Config") if isinstance(container.get("Config"), dict) else {}
        actual_labels = config.get("Labels") if isinstance(config.get("Labels"), dict) else {}
        configured_labels = service.get("labels") if isinstance(service.get("labels"), dict) else {}
        for key, value in configured_labels.items():
            if str(actual_labels.get(str(key))) != str(value):
                raise RestorationError("config_drift")
        return _readonly_mount_hashes(contract["volumes"], base_dir=self.compose_path.parent)

    def _preflight(self) -> dict[str, Any]:
        service, _image, contract = self._service()
        configured_name = service.get("container_name")
        if configured_name is None:
            configured_name = self.target.container_name or self.instance_id
        if not isinstance(configured_name, str) or not configured_name:
            raise RestorationError("config_drift")
        if configured_name != self._owner_container_name:
            raise RestorationError("config_drift")
        if self.target.container_name and self.target.container_name != configured_name:
            raise RestorationError("config_drift")
        container = self._container(configured_name)
        if container.get("Image") != contract["image_id"]:
            raise RestorationError("image_drift")
        config = container.get("Config") if isinstance(container.get("Config"), dict) else {}
        if config.get("Image") != contract["image_ref"]:
            raise RestorationError("config_drift")
        expected_mounts = tuple(
            sorted(contract["volumes"], key=lambda item: (item[1], item[0], item[2]))
        )
        if _normalise_mounts(container.get("Mounts", [])) != expected_mounts:
            raise RestorationError("config_drift")
        readonly_hashes = self._validate_instance_shape(service, contract, container)
        return {
            "service_hash": contract["service_hash"],
            "image_ref": contract["image_ref"],
            "image_id": contract["image_id"],
            "container_id": str(container.get("Id", "")),
            "container_name": configured_name,
            "readonly_hashes": readonly_hashes,
        }

    def _lease(self) -> dict[str, Any] | None:
        lease = _read_json(self.lease_path)
        if lease is None:
            return None
        if (
            lease.get("status") not in {"active", "released", "quarantined"}
            or lease.get("instance_id") != self.instance_id
            or lease.get("site_url") != self.target.site_url
            or not isinstance(lease.get("before"), dict)
            or not isinstance(lease.get("token_hash"), str)
            or len(lease["token_hash"]) != 64
        ):
            raise RestorationError("state_corrupt")
        try:
            uuid.UUID(lease.get("scope_id"))
            int(lease["token_hash"], 16)
        except (ValueError, AttributeError, TypeError) as exc:
            raise RestorationError("state_corrupt") from exc
        return lease

    def _request_target(self, payload: dict[str, Any], *, action: str) -> tuple[str, str, str]:
        if not isinstance(payload, dict):
            raise RestorationError("invalid_request")
        expected = {
            "acquire": {"action", "instance_id", "site_url", "scope_id"},
            "restore": {
                "action",
                "instance_id",
                "site_url",
                "scope_id",
                "lease_token",
                "operation_id",
            },
            "release": {
                "action",
                "instance_id",
                "site_url",
                "scope_id",
                "lease_token",
                "operation_id",
            },
        }[action]
        if set(payload) != expected:
            raise RestorationError("invalid_request")
        if payload.get("action") != action or payload.get("instance_id") != self.instance_id:
            raise RestorationError("wrong_target")
        if payload.get("site_url") != self.target.site_url:
            raise RestorationError("wrong_target")
        scope_id = payload.get("scope_id")
        if not isinstance(scope_id, str) or not scope_id or len(scope_id) > 128:
            raise RestorationError("invalid_request")
        try:
            scope_id = str(uuid.UUID(scope_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise RestorationError("invalid_request") from exc
        token = payload.get("lease_token", "")
        operation_id = payload.get("operation_id", "")
        if action != "acquire" and (not isinstance(token, str) or not token):
            raise RestorationError("lease_required")
        if action != "acquire" and (
            not isinstance(operation_id, str) or not operation_id or len(operation_id) > 128
        ):
            raise RestorationError("invalid_request")
        if action != "acquire":
            operation_id = validate_identifier(operation_id, "invalid_request")
        return scope_id, str(token), str(operation_id)

    def _check_lease(self, scope_id: str, token: str) -> dict[str, Any]:
        lease = self._lease()
        if not lease or lease.get("status") != "active":
            if lease and lease.get("status") == "quarantined":
                raise RestorationError("scope_quarantined")
            raise RestorationError("lease_required")
        if lease.get("scope_id") != scope_id or lease.get("token_hash") != _token_hash(token):
            raise RestorationError("lease_required")
        return lease

    def _operation_path(self, operation_id: str) -> Path:
        return self.state_dir / f"operation-{operation_id}.json"

    def _validate_operation_journals(self) -> None:
        """Reject any existing malformed operation record before a lease."""

        try:
            paths = sorted(self.state_dir.glob("operation-*.json"))
        except OSError as exc:
            raise RestorationError("state_corrupt") from exc
        for path in paths:
            record = _read_json(path)
            if record is None:
                continue
            operation_id = path.name.removeprefix("operation-").removesuffix(".json")
            if (
                not validate_identifier(operation_id, "state_corrupt")
                or record.get("operation_id") != operation_id
                or not isinstance(record.get("scope_id"), str)
            ):
                raise RestorationError("state_corrupt")
            try:
                uuid.UUID(record["scope_id"])
            except (ValueError, AttributeError, TypeError) as exc:
                raise RestorationError("state_corrupt") from exc
            status = record.get("status")
            if status == "intent":
                raise RestorationError("state_corrupt")
            if status == "restored" and (
                not isinstance(record.get("result"), dict)
                or not isinstance(record.get("after"), dict)
            ):
                raise RestorationError("state_corrupt")
            if status not in {"restored", "quarantined"}:
                raise RestorationError("state_corrupt")

    def _quarantine_operation(
        self,
        path: Path,
        *,
        intent: dict[str, Any],
        operation_id: str,
        scope_id: str,
        reason: str,
    ) -> None:
        record = _read_json(path)
        if record is None:
            record = dict(intent)
        if record.get("scope_id") not in {None, scope_id}:
            raise RestorationError("state_corrupt")
        record.setdefault("operation_id", operation_id)
        record.setdefault("scope_id", scope_id)
        record["status"] = "quarantined"
        record["reason"] = reason
        _atomic_json(path, record)

    def _reconcile_completed_operation(
        self,
        lease: dict[str, Any],
        operation: dict[str, Any],
        *,
        scope_id: str,
        operation_id: str,
    ) -> dict[str, Any]:
        """Repair the lease after a crash between terminal journal writes."""

        if operation.get("scope_id") != scope_id or operation.get("operation_id") != operation_id:
            raise RestorationError("operation_conflict")
        result = operation.get("result")
        recorded_after = operation.get("after")
        if not isinstance(result, dict) or not isinstance(recorded_after, dict):
            raise RestorationError("operation_quarantined")
        if (
            result.get("operation_id") != operation_id
            or result.get("instance_id") != self.instance_id
            or result.get("site_url") != self.target.site_url
            or result.get("image_id") != recorded_after.get("image_id")
            or result.get("before_container_id") != operation.get("before_container_id")
            or result.get("after_container_id") != recorded_after.get("container_id")
        ):
            raise RestorationError("operation_quarantined")
        current = self._preflight()
        if any(
            current.get(key) != recorded_after.get(key)
            for key in (
                "service_hash",
                "image_ref",
                "image_id",
                "container_id",
                "container_name",
                "readonly_hashes",
            )
        ):
            raise RestorationError("config_drift")
        lease["before"] = current
        lease["operation_id"] = operation_id
        _atomic_json(self.lease_path, lease)
        return result

    def _acquire(self, payload: dict[str, Any]) -> dict[str, Any]:
        scope_id, _, _ = self._request_target(payload, action="acquire")
        lease = self._lease()
        if lease and lease.get("status") == "active":
            raise RestorationError("lease_conflict")
        if lease and lease.get("status") == "quarantined":
            raise RestorationError("scope_quarantined")
        self._validate_operation_journals()
        before = self._preflight()
        token = secrets.token_urlsafe(32)
        _atomic_json(
            self.lease_path,
            {
                "status": "active",
                "scope_id": scope_id,
                "token_hash": _token_hash(token),
                "instance_id": self.instance_id,
                "site_url": self.target.site_url,
                "before": before,
                "operation_id": None,
            },
        )
        return {
            "status": "acquired",
            "scope_id": scope_id,
            "lease_token": token,
            "instance_id": self.instance_id,
            "site_url": self.target.site_url,
        }

    def _restore(self, payload: dict[str, Any]) -> dict[str, Any]:
        scope_id, token, operation_id = self._request_target(payload, action="restore")
        lease = self._check_lease(scope_id, token)
        self._validate_operation_journals()
        operation_path = self._operation_path(operation_id)
        previous = _read_json(operation_path)
        if previous:
            if previous.get("scope_id") != scope_id:
                raise RestorationError("operation_conflict")
            if previous.get("status") == "restored":
                return dict(
                    self._reconcile_completed_operation(
                        lease,
                        previous,
                        scope_id=scope_id,
                        operation_id=operation_id,
                    )
                )
            raise RestorationError("operation_quarantined")
        latest_operation_id = lease.get("operation_id")
        if latest_operation_id not in {None, operation_id}:
            latest = _read_json(self._operation_path(str(latest_operation_id)))
            if (
                not latest
                or latest.get("scope_id") != scope_id
                or latest.get("status") != "restored"
            ):
                raise RestorationError("operation_conflict")
        before = self._preflight()
        recorded = lease.get("before")
        if not isinstance(recorded, dict) or any(
            before.get(key) != recorded.get(key)
            for key in (
                "service_hash",
                "image_ref",
                "image_id",
                "container_id",
                "container_name",
                "readonly_hashes",
            )
        ):
            raise RestorationError("config_drift")
        intent = {
            "status": "intent",
            "operation_id": operation_id,
            "scope_id": scope_id,
            "before_container_id": before["container_id"],
            "service_hash": before["service_hash"],
            "image_id": before["image_id"],
            "container_name": before["container_name"],
            "readonly_hashes": before["readonly_hashes"],
        }
        _atomic_json(operation_path, intent)
        lease["operation_id"] = operation_id
        _atomic_json(self.lease_path, lease)
        try:
            self.docker.recreate(self.target.service_name)
            after = self._preflight()
            if after["container_id"] == before["container_id"]:
                raise RestorationError("container_not_recreated")
            if any(
                after[key] != before[key]
                for key in (
                    "service_hash",
                    "image_ref",
                    "image_id",
                    "container_name",
                    "readonly_hashes",
                )
            ):
                raise RestorationError("config_drift")
            self._wait_ready()
        except DockerTimeoutError as exc:
            self._quarantine_operation(
                operation_path,
                intent=intent,
                operation_id=operation_id,
                scope_id=scope_id,
                reason="timeout_quarantined",
            )
            lease["status"] = "quarantined"
            _atomic_json(self.lease_path, lease)
            raise RestorationError("timeout_quarantined") from exc
        except TimeoutError as exc:
            self._quarantine_operation(
                operation_path,
                intent=intent,
                operation_id=operation_id,
                scope_id=scope_id,
                reason="timeout_quarantined",
            )
            lease["status"] = "quarantined"
            _atomic_json(self.lease_path, lease)
            raise RestorationError("timeout_quarantined") from exc
        except (RestorationError, DockerOperationError) as exc:
            reason = exc.reason if isinstance(exc, RestorationError) else "restore_quarantined"
            self._quarantine_operation(
                operation_path,
                intent=intent,
                operation_id=operation_id,
                scope_id=scope_id,
                reason=reason,
            )
            lease["status"] = "quarantined"
            _atomic_json(self.lease_path, lease)
            if isinstance(exc, RestorationError):
                raise
            raise RestorationError("restore_quarantined") from exc
        result = {
            "status": "restored",
            "operation_id": operation_id,
            "instance_id": self.instance_id,
            "site_url": self.target.site_url,
            "before_container_id": before["container_id"],
            "after_container_id": after["container_id"],
            "image_id": after["image_id"],
        }
        terminal = dict(intent)
        terminal.update({"status": "restored", "after": after, "result": result})
        _atomic_json(operation_path, terminal)
        lease["before"] = after
        lease["operation_id"] = operation_id
        _atomic_json(self.lease_path, lease)
        return result

    def _release(self, payload: dict[str, Any]) -> dict[str, Any]:
        scope_id, token, operation_id = self._request_target(payload, action="release")
        lease = self._check_lease(scope_id, token)
        self._validate_operation_journals()
        operation = _read_json(self._operation_path(operation_id))
        if not operation or operation.get("status") != "restored":
            raise RestorationError("operation_required")
        if operation.get("scope_id") != scope_id:
            raise RestorationError("operation_conflict")
        if lease.get("operation_id") not in {None, operation_id}:
            raise RestorationError("operation_conflict")
        if lease.get("operation_id") != operation_id:
            self._reconcile_completed_operation(
                lease,
                operation,
                scope_id=scope_id,
                operation_id=operation_id,
            )
        lease["status"] = "released"
        _atomic_json(self.lease_path, lease)
        return {
            "status": "released",
            "operation_id": operation_id,
            "instance_id": self.instance_id,
            "site_url": self.target.site_url,
        }

    def handle_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
                raise RestorationError("invalid_request")
            action = payload["action"]
            if action == "acquire":
                return self._acquire(payload)
            if action == "restore":
                return self._restore(payload)
            if action == "release":
                return self._release(payload)
            raise RestorationError("invalid_request")
        except RestorationError as exc:
            return {"status": "error", "reason": exc.reason}
        except Exception:
            return {"status": "error", "reason": "internal_error"}
