from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .errors import RestorationError

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_ID_RE = _DIGEST_RE
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DB_PATH_MARKERS = (
    "/var/lib/postgresql",
    "/var/lib/mysql",
    "/var/lib/mariadb",
    "/var/opt/gitlab/postgresql",
    "/var/opt/gitlab/mysql",
)
_ALLOWED_READONLY_CONFIG_TARGETS = frozenset({"/etc/gitlab/gitlab.rb"})
_SUPPORTED_SITES = frozenset({"reddit", "gitlab"})


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_loopback_host(value: str) -> bool:
    host = value.strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback_url(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RestorationError("wrong_target")
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not _is_loopback_host(parsed.hostname)
    ):
        raise RestorationError(reason)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RestorationError(reason)
    if parsed.path not in {"", "/"}:
        raise RestorationError(reason)
    if parsed.port is None:
        raise RestorationError(reason)
    return value.strip().rstrip("/")


def validate_identifier(value: Any, reason: str = "invalid_request") -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise RestorationError(reason)
    return value


@dataclass(frozen=True)
class InstanceTarget:
    instance_id: str
    site_name: str
    site_url: str
    status_url: str | None
    service_name: str
    container_name: str | None


def resolved_ports(service: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    """Return ``(host_ip, published, target)`` for resolved Compose ports."""

    ports = service.get("ports", []) or []
    if not isinstance(ports, list):
        raise RestorationError("non_loopback_port")
    result: list[tuple[str, int, int]] = []
    for raw in ports:
        if isinstance(raw, dict):
            host = str(raw.get("host_ip", ""))
            published = raw.get("published")
            target = raw.get("target")
        elif isinstance(raw, str):
            pieces = raw.split(":")
            if len(pieces) == 3:
                host, published, target = pieces
            elif len(pieces) == 4 and pieces[0] in {"", "127.0.0.1", "::1"}:
                host, published, target = pieces[0] or "127.0.0.1", pieces[2], pieces[3]
            else:
                raise RestorationError("non_loopback_port")
        else:
            raise RestorationError("non_loopback_port")
        if not host or published is None or target is None:
            raise RestorationError("non_loopback_port")
        try:
            published_int = int(published)
            target_int = int(target)
        except (TypeError, ValueError) as exc:
            raise RestorationError("non_loopback_port") from exc
        if not 1 <= published_int <= 65535 or not 1 <= target_int <= 65535:
            raise RestorationError("non_loopback_port")
        if not _is_loopback_host(host):
            raise RestorationError("non_loopback_port")
        result.append((host, published_int, target_int))
    if not result:
        raise RestorationError("non_loopback_port")
    return tuple(result)


def load_target(instances_path: Path, instance_id: str) -> InstanceTarget:
    """Load one exact loopback instance from generated host-local topology."""

    validate_identifier(instance_id, "invalid_request")
    try:
        document = json.loads(instances_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RestorationError("invalid_instances") from exc
    rows = document.get("instances") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise RestorationError("invalid_instances")
    matching = [
        row for row in rows if isinstance(row, dict) and row.get("replica_name") == instance_id
    ]
    if len(matching) != 1:
        raise RestorationError("wrong_target")
    row = matching[0]
    site_name = row.get("site_name")
    if not isinstance(site_name, str) or not site_name.strip():
        raise RestorationError("invalid_instances")
    site_name = site_name.strip().lower()
    if site_name not in _SUPPORTED_SITES:
        raise RestorationError("unsupported_site")
    site_url = _require_loopback_url(row.get("site_url"), "non_loopback_target")
    status_raw = row.get("status_url")
    status_url = None
    if status_raw is not None:
        status_url = _require_loopback_url(status_raw, "non_loopback_target")
    service_name = row.get("service_name") or instance_id
    if not isinstance(service_name, str) or service_name != instance_id:
        raise RestorationError("config_drift")
    container_name = row.get("container_name")
    if container_name is not None:
        if not isinstance(container_name, str) or not container_name.strip():
            raise RestorationError("invalid_instances")
        container_name = container_name.strip()
    return InstanceTarget(
        instance_id=instance_id,
        site_name=site_name.strip(),
        site_url=site_url,
        status_url=status_url,
        service_name=service_name,
        container_name=container_name,
    )


def image_pin(image: Any) -> tuple[str, str | None, str | None]:
    """Return (kind, expected_repo_digest, expected_image_id)."""

    if not isinstance(image, str) or not image.strip():
        raise RestorationError("mutable_image")
    value = image.strip()
    if _IMAGE_ID_RE.fullmatch(value):
        return "image_id", None, value
    if "@" in value:
        _, digest = value.rsplit("@", 1)
        if _DIGEST_RE.fullmatch(digest):
            return "repo_digest", digest, None
    raise RestorationError("mutable_image")


def _volume_parts(value: Any) -> tuple[str, str, str]:
    if isinstance(value, str):
        pieces = value.split(":")
        if len(pieces) == 2:
            return pieces[0], pieces[1], "rw"
        if len(pieces) == 3:
            return pieces[0], pieces[1], pieces[2] or "rw"
        if len(pieces) == 4 and pieces[0] in {"", "127.0.0.1", "::1"}:
            return pieces[0], pieces[2], pieces[3] or "rw"
    elif isinstance(value, dict):
        source = str(value.get("source", ""))
        target = str(value.get("target", ""))
        mode = "ro" if value.get("read_only") else str(value.get("mode", "rw"))
        return source, target, mode
    raise RestorationError("unexpected_mount")


def validate_volumes(
    service: dict[str, Any], image: dict[str, Any]
) -> tuple[tuple[str, str, str], ...]:
    volumes: list[tuple[str, str, str]] = []
    for raw in service.get("volumes", []) or []:
        source, target, mode = _volume_parts(raw)
        target = target.strip()
        if not target:
            raise RestorationError("unexpected_mount")
        target_lower = target.lower()
        if any(marker in target_lower for marker in _DB_PATH_MARKERS):
            raise RestorationError("unexpected_db_mount")
        if mode not in {"ro", "rw"}:
            raise RestorationError("unexpected_mount")
        if mode != "ro" and target not in _ALLOWED_READONLY_CONFIG_TARGETS:
            raise RestorationError("unexpected_mount")
        volumes.append((source, target, mode))
    config = image.get("Config") if isinstance(image, dict) else None
    image_volumes = config.get("Volumes") if isinstance(config, dict) else None
    if isinstance(image_volumes, dict):
        for target in image_volumes:
            if any(marker in str(target).lower() for marker in _DB_PATH_MARKERS):
                raise RestorationError("unexpected_db_mount")
            raise RestorationError("unexpected_mount")
    return tuple(volumes)


def validate_ports(service: dict[str, Any]) -> None:
    resolved_ports(service)


def service_contract(service: dict[str, Any], image: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(service, dict):
        raise RestorationError("config_drift")
    image_ref = service.get("image")
    image_kind, expected_digest, expected_image_id = image_pin(image_ref)
    network_mode = service.get("network_mode")
    if network_mode is not None:
        network_mode_text = str(network_mode).strip().lower()
        if network_mode_text in {"host", "container"} or network_mode_text.startswith("container:"):
            raise RestorationError("network_mode_unsupported")
    validate_ports(service)
    volumes = validate_volumes(service, image)
    platform = service.get("platform")
    if platform not in {None, "linux/amd64", "linux/x86_64"}:
        raise RestorationError("config_drift")
    architecture = str(image.get("Architecture", ""))
    if architecture and architecture not in {"amd64", "x86_64"}:
        raise RestorationError("config_drift")
    contract = {
        "service_hash": _canonical_sha256(service),
        "image_ref": str(image_ref),
        "image_kind": image_kind,
        "expected_repo_digest": expected_digest,
        "expected_image_id": expected_image_id,
        "image_id": str(image.get("Id", "")),
        "volumes": volumes,
    }
    if expected_image_id and contract["image_id"] != expected_image_id:
        raise RestorationError("image_drift")
    if expected_digest:
        repo_digests = image.get("RepoDigests", [])
        if not isinstance(repo_digests, list) or not any(
            isinstance(item, str) and item.rsplit("@", 1)[-1] == expected_digest
            for item in repo_digests
        ):
            raise RestorationError("image_drift")
    return contract
