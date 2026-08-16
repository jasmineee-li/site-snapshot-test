"""Regular-participant session binding for the Classifieds editor."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from warp_taskgen.editors.base import EditorError
from warp_taskgen.sites.classifieds_editor import ClassifiedsEditor

_CLASSIFIEDS_CANARY_ORIGIN_HOST = "127.0.0.1"


def validate_classifieds_canary_origin(instance: Mapping[str, Any]) -> str:
    """Require the named writer lane to mutate only its loopback canary.

    The runtime composition is intentionally opt-in, but that opt-in must not
    turn an arbitrary instance ``site_url`` into a public Classifieds write.
    Keep this check on the authenticated editor boundary so both Phase 2c
    seeding and Phase 4 preflight fail before a request is made.
    """

    raw = instance.get("site_url")
    site_url = raw.strip().rstrip("/") if isinstance(raw, str) else ""
    try:
        parsed = urlsplit(site_url)
        port = parsed.port
    except ValueError as exc:
        raise EditorError(
            "site_mismatch",
            "Classifieds canary writer requires an exact loopback HTTP origin",
        ) from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != _CLASSIFIEDS_CANARY_ORIGIN_HOST
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise EditorError(
            "site_mismatch",
            "Classifieds canary writer requires an exact loopback HTTP origin",
        )
    return site_url


def _writer_storage_state_path(instance: Mapping[str, Any]) -> Path:
    auth = instance.get("auth")
    storage_state = auth.get("storage_state") if isinstance(auth, Mapping) else None
    path = storage_state.get("path") if isinstance(storage_state, Mapping) else None
    if not isinstance(auth, Mapping) or auth.get("type") != "storage_state":
        raise EditorError("auth_missing", "Classifieds writer requires auth.type='storage_state'")
    if not isinstance(path, str) or not path.strip():
        raise EditorError(
            "auth_missing",
            "Classifieds writer requires an explicit auth.storage_state.path",
        )
    return Path(path).expanduser()


def _writer_cookie_records(
    storage_state_path: Path,
    *,
    site_url: str,
) -> list[tuple[str, str, str, str]]:
    """Read and validate cookie metadata without exposing cookie values."""

    try:
        payload = json.loads(storage_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise EditorError(
            "auth_missing", "Classifieds writer storage_state is not readable JSON"
        ) from None
    if not isinstance(payload, Mapping):
        raise EditorError("auth_missing", "Classifieds writer storage_state must be a JSON object")
    cookies = payload.get("cookies")
    if not isinstance(cookies, list) or not cookies:
        raise EditorError("auth_missing", "Classifieds writer storage_state must contain cookies")
    parsed_site = urlsplit(site_url)
    site_host = parsed_site.hostname
    if parsed_site.scheme not in {"http", "https"} or not site_host:
        raise EditorError("auth_missing", "Classifieds writer requires a valid site_url")

    origins = payload.get("origins", [])
    if not isinstance(origins, list):
        raise EditorError("auth_missing", "Classifieds writer storage_state origins are malformed")
    for origin in origins:
        if not isinstance(origin, Mapping) or not isinstance(origin.get("origin"), str):
            raise EditorError(
                "auth_missing", "Classifieds writer storage_state origins are malformed"
            )
        parsed_origin = urlsplit(origin["origin"])
        if (parsed_origin.scheme, parsed_origin.hostname, parsed_origin.port) != (
            parsed_site.scheme,
            site_host,
            parsed_site.port,
        ):
            raise EditorError(
                "auth_missing", "Classifieds writer storage_state has a foreign origin"
            )

    records: list[tuple[str, str, str, str]] = []
    for cookie in cookies:
        if not isinstance(cookie, Mapping):
            raise EditorError(
                "auth_missing", "Classifieds writer storage_state cookie is malformed"
            )
        name = cookie.get("name")
        value = cookie.get("value")
        domain = cookie.get("domain")
        path = cookie.get("path", "/")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, str)
            or not isinstance(domain, str)
            or not domain.strip()
            or not isinstance(path, str)
            or not path.startswith("/")
        ):
            raise EditorError(
                "auth_missing", "Classifieds writer storage_state cookie is malformed"
            )
        normalized_domain = domain.strip().lower().lstrip(".")
        if normalized_domain != site_host.lower().strip("."):
            raise EditorError(
                "auth_missing",
                "Classifieds writer storage_state has a foreign same-origin cookie domain",
            )
        records.append((name.strip(), value, normalized_domain, path))
    return records


class ClassifiedsAuthenticatedEditor(ClassifiedsEditor):
    """Classifieds editor bound to one explicit writer storage state."""

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        # Phase 4 calls this class-level seam before constructing an editor or
        # loading writer cookies.  Reject a public origin at that boundary.
        super().probe_base_state(instance)
        validate_classifieds_canary_origin(instance)

    def __init__(self, instance: Mapping[str, Any] | None = None, session: Any = None) -> None:
        if not isinstance(instance, Mapping):
            raise EditorError("auth_missing", "Classifieds writer requires an instance mapping")
        if session is None or not hasattr(session, "cookies"):
            raise EditorError("auth_missing", "Classifieds writer requires an injected session")
        validate_classifieds_canary_origin(instance)
        records = _writer_cookie_records(
            _writer_storage_state_path(instance),
            site_url=str(instance.get("site_url") or ""),
        )
        try:
            for name, value, domain, path in records:
                session.cookies.set(name, value, domain=domain, path=path)
        except Exception:
            raise EditorError(
                "auth_missing", "Classifieds writer storage_state cookies could not be loaded"
            ) from None
        super().__init__(instance, session)


def classifieds_editor_factory(
    instance: Mapping[str, Any],
    session: Any,
) -> ClassifiedsEditor:
    return ClassifiedsAuthenticatedEditor(instance, session)


__all__ = [
    "ClassifiedsAuthenticatedEditor",
    "classifieds_editor_factory",
    "validate_classifieds_canary_origin",
]
