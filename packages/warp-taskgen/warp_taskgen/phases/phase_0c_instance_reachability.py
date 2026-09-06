"""Phase 0c: Benchmark Instance URLs, proxy rewriting, and the reachability report.

Owns instance URL sanitizing and the Modal connectivity rules, the instance map,
lookup, and groups the profiling loop iterates, verification-proxy metadata and
redaction values, and the per-site reachability record and its report file.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from warp_taskgen.config import BenchmarkInstance, VerificationProxy
from warp_taskgen.phases.phase_0c_artifacts import (
    reachability_report_path,
    write_text_atomic,
)
from warp_taskgen.placeholders import normalize_site_name

logger = logging.getLogger(__name__)


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text*."""
    write_text_atomic(path, text)


def _reachability_report_path(output_dir: Path) -> Path:
    return reachability_report_path(output_dir)


def _sanitize_instance_site_url(site_url: str, *, site_name: str) -> str:
    """Return a normalized live-verification URL safe to pass into sandboxes."""
    parts = urlsplit(site_url.strip())
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(
            f"site {site_name!r} has unsupported site_url {site_url!r}; expected http(s) base URL"
        )
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            f"site {site_name!r} site_url must not include credentials when used for Phase 0c"
        )
    if parts.query or parts.fragment:
        raise ValueError(
            f"site {site_name!r} site_url must not include query or fragment when used for Phase 0c"
        )
    hostname = parts.hostname
    if not hostname:
        raise ValueError(
            f"site {site_name!r} has unsupported site_url {site_url!r}; expected http(s) base URL"
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"site {site_name!r} has invalid site_url port in {site_url!r}") from exc

    default_port = 80 if scheme == "http" else 443
    normalized_host = hostname.lower()
    host_display = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    normalized_netloc = host_display
    if port is not None and port != default_port:
        normalized_netloc = f"{host_display}:{port}"
    normalized_path = parts.path.rstrip("/")
    return urlunsplit((scheme, normalized_netloc, normalized_path, "", ""))


def _phase_0c_modal_unreachable_host_reason(site_url: str) -> str | None:
    parts = urlsplit(site_url)
    host = parts.hostname
    if not host:
        return "missing host"
    host_lower = host.lower()
    if host_lower in {"localhost"} or host_lower.endswith(".local"):
        return f"non-public hostname {host!r}"
    try:
        address = ipaddress.ip_address(host_lower)
    except ValueError:
        return None
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        return f"non-public IP address {host!r}"
    return None


def _validate_phase_0c_modal_connectivity_urls(
    instance_urls: dict[str, str],
    *,
    instances_path: Path | None = None,
) -> None:
    """Fail early when Modal Phase 0c would receive host-local URLs."""
    invalid = []
    for site_name, site_url in sorted(instance_urls.items()):
        reason = _phase_0c_modal_unreachable_host_reason(site_url)
        if reason:
            invalid.append(f"{site_name}: {site_url} ({reason})")
    if not invalid:
        return
    path_hint = f" from {instances_path}" if instances_path is not None else ""
    raise RuntimeError(
        "Phase 0c live verification runs inside Modal sandboxes and requires "
        f"externally reachable instance URLs{path_hint}. The selected instance "
        "config uses host-local/on-host URLs that Modal cannot reach:\n"
        + "\n".join(f"  - {item}" for item in invalid)
        + "\nUse an externally reachable/proxied instance file for Phase 0c "
        "(for r5, instances.smoke.json), and use instances.scale.json only for "
        "on-host browser phases such as Phase 2c and Phase 4."
    )


def _build_instance_site_url_map(instances: list[BenchmarkInstance] | None) -> dict[str, str]:
    """Return a stable representative site_url for each site for live verification."""
    if not instances:
        return {}

    by_site: dict[str, set[str]] = {}
    for instance in instances:
        normalized_site = normalize_site_name(instance.site_name)
        if not normalized_site:
            continue
        sanitized_url = _sanitize_instance_site_url(instance.site_url, site_name=instance.site_name)
        by_site.setdefault(normalized_site, set()).add(sanitized_url)

    result: dict[str, str] = {}
    for site_name, site_urls in by_site.items():
        selected = sorted(site_urls)[0]
        if len(site_urls) > 1:
            logger.info(
                "Phase 0c: site %r has %d configured instance URLs; using %s for live verification",
                site_name,
                len(site_urls),
                selected,
            )
        result[site_name] = selected
    return result


def _build_instance_lookup(
    instances: list[BenchmarkInstance] | None,
) -> dict[str, BenchmarkInstance]:
    """Map normalized site name to a representative ``BenchmarkInstance``.

    Used by host-side enrichment hooks (e.g. gitlab handle enumeration)
    that need access to the instance's auth config, not just its URL.
    Picks the first instance per site; replicas share auth.
    """
    if not instances:
        return {}
    out: dict[str, BenchmarkInstance] = {}
    for instance in instances:
        normalized_site = normalize_site_name(instance.site_name)
        if not normalized_site or normalized_site in out:
            continue
        out[normalized_site] = instance
    return out


def _build_instance_groups(
    instances: list[BenchmarkInstance] | None,
) -> dict[str, list[BenchmarkInstance]]:
    """Map normalized site name to all configured instances for that site."""
    if not instances:
        return {}
    out: dict[str, list[BenchmarkInstance]] = {}
    for instance in instances:
        normalized_site = normalize_site_name(instance.site_name)
        if not normalized_site:
            continue
        out.setdefault(normalized_site, []).append(instance)
    return out


def _apply_proxy_to_url(site_url: str, port_offset: int, *, scheme: str | None = None) -> str:
    """Rewrite a site URL to use the proxy port (real_port + port_offset)."""
    parts = urlsplit(site_url)
    try:
        port = parts.port
    except ValueError:
        return site_url
    if port is None:
        # No explicit port, cannot apply offset meaningfully.
        return site_url
    proxy_port = port + port_offset
    hostname = parts.hostname or ""
    host_display = f"[{hostname}]" if ":" in hostname else hostname
    proxy_netloc = f"{host_display}:{proxy_port}"
    proxy_scheme = scheme or parts.scheme
    return urlunsplit((proxy_scheme, proxy_netloc, parts.path, parts.query, parts.fragment))


def _verification_proxy_metadata(
    verification_proxy: VerificationProxy | None,
) -> dict[str, Any] | None:
    """Return stable, non-secret proxy metadata for Phase 0c cache fingerprints."""
    if verification_proxy is None:
        return None
    token_digest = hashlib.sha256(verification_proxy.token.encode()).hexdigest()
    return {
        "scheme": verification_proxy.scheme,
        "port_offset": verification_proxy.port_offset,
        "token_sha256": token_digest,
    }


def _phase_0c_redact_values(verification_proxy: VerificationProxy | None) -> tuple[str, ...]:
    if verification_proxy is None:
        return ()
    return (
        verification_proxy.token,
        f"X-Worldsim-Token: {verification_proxy.token}",
    )


def _delivery_channel_verification_status(channel: object) -> tuple[str, str | None]:
    if not isinstance(channel, dict):
        return "malformed_channel", None
    verified = channel.get("verified")
    note = channel.get("verification_notes")
    note_text = note.strip() if isinstance(note, str) and note.strip() else None
    if verified is True:
        return "verified", note_text
    if verified is False:
        return "discrepancy_corrected", note_text
    if verified is None:
        return "unverified", note_text
    return "malformed_verified_field", note_text


def _site_reachability_record(
    *,
    site_name: str,
    site_url: str | None,
    verification_proxy: VerificationProxy | None,
    injection_surface: dict[str, Any] | None,
    cached: bool = False,
) -> dict[str, Any]:
    """Summarize Phase 0c live verification health without gating profiles."""
    proxy_metadata = _verification_proxy_metadata(verification_proxy)
    if not site_url:
        return {
            "site": site_name,
            "status": "no_instance_config",
            "cached": cached,
            "site_url": None,
            "verification_proxy": proxy_metadata,
            "channel_counts": {},
            "notes": ["Phase 0c had no instance URL; profile is code-derived only."],
        }

    surfaces = []
    if isinstance(injection_surface, dict) and isinstance(
        injection_surface.get("injection_surface"), list
    ):
        surfaces = [
            item for item in injection_surface["injection_surface"] if isinstance(item, dict)
        ]

    channel_counts: dict[str, int] = {}
    notes: list[str] = []
    for surface in surfaces:
        channels = surface.get("delivery_channels")
        if not isinstance(channels, list):
            continue
        for channel in channels:
            status, note = _delivery_channel_verification_status(channel)
            channel_counts[status] = channel_counts.get(status, 0) + 1
            if note and len(notes) < 8:
                notes.append(f"{surface.get('id') or 'unknown'}: {note}")

    if not channel_counts:
        status = "no_channels"
    elif channel_counts.get("unverified") or channel_counts.get("malformed_verified_field"):
        status = "unverified"
    elif channel_counts.get("discrepancy_corrected"):
        status = "verified_with_corrections"
    elif channel_counts.get("verified"):
        status = "verified"
    else:
        status = "unknown"

    return {
        "site": site_name,
        "status": status,
        "cached": cached,
        "site_url": site_url,
        "verification_proxy": proxy_metadata,
        "channel_counts": channel_counts,
        "notes": notes,
    }


def _write_reachability_report(output_dir: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "phase": "phase_0c",
        "sites": sorted(records, key=lambda item: str(item.get("site") or "")),
    }
    _write_text_atomic(_reachability_report_path(output_dir), json.dumps(payload, indent=2))
