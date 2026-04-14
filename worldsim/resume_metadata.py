"""Helpers for provenance-aware resume fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any

RESULT_FINGERPRINT_KEY = "_resume_fingerprint"


def fingerprint_payload(*parts: Any) -> str:
    """Return a stable content fingerprint for resume-sensitive payloads."""
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def instance_identity(instance: Any) -> dict[str, Any]:
    """Return the execution-affecting identity for one benchmark instance."""
    return {
        "site_name": getattr(instance, "site_name", None),
        "site_url": getattr(instance, "site_url", None),
        "reset_endpoint": getattr(instance, "reset_endpoint", None),
        "db_connection": getattr(instance, "db_connection", None),
        "url_placeholders": getattr(instance, "url_placeholders", None),
        "auth": getattr(instance, "auth", None),
    }


def instances_identity(instances: list[Any]) -> list[dict[str, Any]]:
    """Return ordered benchmark-instance identities for resume fingerprints."""
    return [instance_identity(instance) for instance in instances]
