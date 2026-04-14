"""Shared DB connection parsing for the URI formats WorldSim actually supports."""

from __future__ import annotations

import urllib.parse
from typing import Any

SUPPORTED_DB_URI_SCHEMES = frozenset({"mysql", "postgres", "postgresql"})


def parse_supported_db_connection(
    db_connection: Any,
    *,
    purpose: str = "db_connection",
) -> urllib.parse.ParseResult:
    """Parse a supported DB URI or raise a clear validation error.

    WorldSim currently supports only URI-style DSNs for MySQL and PostgreSQL.
    This helper enforces that contract before any runtime code reaches
    ``urllib.parse`` or a database driver.
    """
    if isinstance(db_connection, (bytes, bytearray)):
        raise ValueError(f"{purpose} must be a URI string, not bytes")
    if not isinstance(db_connection, str):
        raise ValueError(f"{purpose} must be a non-empty URI string")

    normalized = db_connection.strip()
    if not normalized:
        raise ValueError(f"{purpose} must be a non-empty URI string")

    parsed = urllib.parse.urlparse(normalized)
    scheme = parsed.scheme.lower()
    if not scheme:
        raise ValueError(
            f"{purpose} must include an explicit scheme like mysql:// or postgresql://"
        )
    if scheme not in SUPPORTED_DB_URI_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_DB_URI_SCHEMES))
        raise ValueError(
            f"{purpose} uses unsupported scheme {scheme!r}; supported schemes: {supported}"
        )
    if not parsed.hostname:
        raise ValueError(f"{purpose} must include a hostname")
    if not (parsed.path or "").lstrip("/"):
        raise ValueError(f"{purpose} must include a database name in the path")

    return parsed
