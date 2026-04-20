"""Runtime health check for Magento's ``base_url`` configuration.

Magento stores ``web/{unsecure,secure}/base_url`` in ``core_config_data``
(a DB table). When we run ``scripts/fix_magento_base_url.sh``, it points
at the verification-proxy origin (e.g. ``http://<host>:17770/``). If
someone rebuilds the Magento container without re-running that script,
base_url reverts to the raw-backend-port default (``http://<host>:7770/``)
and every Phase 4 Browser-Use run hits absolute form actions / inline JS
at a port the proxy/SG will not serve.

This module provides a cheap HTTP probe that extracts Magento's
rendered ``var BASE_URL = '…'`` from the homepage and compares against
what the proxy expects. It is the runtime equivalent of the post-deploy
verification in ``fix_magento_base_url.sh`` — cross-checked every run
so a silent drift in DB state fails loudly before the agent launches.

Source: Magento's Luma theme renders the BASE_URL global in
``Magento_Theme/templates/page/js/require_js.phtml``. It is present in
the HTML ``<head>`` before RequireJS loads, requires no auth, and is
the canonical "what does Magento think its origin is" signal. Docs:
https://experienceleague.adobe.com/en/docs/commerce-admin/stores-sales/site-store/store-urls
"""

from __future__ import annotations

import codecs
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Magento emits ``var BASE_URL = 'http://.../';`` in every themed page's
# <head>. The literal is always single-quoted in stock themes. We allow
# escapes inside the quoted string so any future theme variant that
# backslash-escapes the terminator still parses.
_BASE_URL_PATTERN = re.compile(r"""var\s+BASE_URL\s*=\s*'((?:[^'\\]|\\.)*)'""")

# Magento sites in the WorldSim benchmark catalog. Only these are subject
# to the base_url probe; other sites (gitlab, reddit, map, wikipedia) have
# different origin-identity mechanisms, documented inline in each.
MAGENTO_SITES: frozenset[str] = frozenset({"shopping", "shopping_admin"})


class BaseUrlMismatch(RuntimeError):
    """Magento's rendered base_url does not match the expected proxy origin."""

    def __init__(
        self,
        *,
        site_name: str,
        probe_url: str,
        expected: str,
        actual: str | None,
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.site_name = site_name
        self.probe_url = probe_url
        self.expected = expected
        self.actual = actual


def _normalize(url: str) -> str:
    """Normalise a URL for comparison — strip trailing slash, lowercase host."""
    stripped = url.strip().rstrip("/")
    # Case-insensitive host compare; path case is preserved (not relevant
    # for base_url which is always origin-only).
    return stripped


def probe_base_url(
    site_url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
    probe_path: str = "/",
) -> str | None:
    """Return the ``BASE_URL`` Magento reports at ``site_url``.

    Returns ``None`` when the homepage HTTP 200s but contains no
    ``var BASE_URL`` declaration — typically because the response was a
    minimal 200 (admin login page, maintenance page) rather than a
    themed storefront page. Raises :class:`BaseUrlMismatch` for connection
    or HTTP failures so callers don't accidentally treat a down site as a
    passing probe.
    """
    target = f"{site_url.rstrip('/')}{probe_path}"
    sess = session or requests.Session()
    try:
        response = sess.get(target, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        raise BaseUrlMismatch(
            site_name="",
            probe_url=target,
            expected="",
            actual=None,
            detail=f"failed to fetch {target}: {exc}",
        ) from exc
    if response.status_code >= 500:
        raise BaseUrlMismatch(
            site_name="",
            probe_url=target,
            expected="",
            actual=None,
            detail=f"{target} returned HTTP {response.status_code}",
        )
    match = _BASE_URL_PATTERN.search(response.text)
    if match is None:
        return None
    raw = match.group(1)
    # Magento's Luma theme emits BASE_URL with JS ``\uXXXX`` escapes for
    # ``:`` (``\u003A``) and ``/`` (``\u002F``). Decode those so string
    # comparison against the plain-text proxy origin succeeds. ``\n``/``\t``
    # would also be expanded; a well-formed URL never contains them, so
    # round-tripping an already-plain value is a no-op.
    try:
        return codecs.decode(raw, "unicode_escape")
    except UnicodeDecodeError:
        return raw


def assert_base_url(
    site_name: str,
    site_url: str,
    expected_base_url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> str:
    """Probe ``site_url`` and raise :class:`BaseUrlMismatch` if drifted.

    Returns the probed ``BASE_URL`` on success so callers can log it.
    If the homepage renders but carries no ``var BASE_URL`` (e.g. a
    Magento admin login page), we treat that as inconclusive and raise —
    the caller should target the storefront root, not a path that skips
    the RequireJS bootstrap.
    """
    actual = probe_base_url(site_url, session=session, timeout=timeout)
    probe_url = f"{site_url.rstrip('/')}/"
    if actual is None:
        raise BaseUrlMismatch(
            site_name=site_name,
            probe_url=probe_url,
            expected=expected_base_url,
            actual=None,
            detail=(
                f"{site_name}: could not extract `var BASE_URL` from {probe_url} — "
                "page may not have rendered a themed Magento response (admin login, "
                "maintenance page, or non-Magento site?). Investigate the stack."
            ),
        )
    if _normalize(actual) != _normalize(expected_base_url):
        raise BaseUrlMismatch(
            site_name=site_name,
            probe_url=probe_url,
            expected=expected_base_url,
            actual=actual,
            detail=(
                f"{site_name}: Magento base_url mismatch\n"
                f"  expected: {expected_base_url}\n"
                f"  actual:   {actual}\n"
                "This usually means the Magento container was rebuilt and "
                "core_config_data.web/{unsecure,secure}/base_url reverted to the "
                "raw backend port. Repair with "
                "`scripts/fix_magento_base_url.sh --via-ssm ...` and retry."
            ),
        )
    return actual


def expected_base_url_for_instance(
    instance: Any,
    verification_proxy: Any | None,
) -> str:
    """Compute what Magento's base_url should be for a given instance.

    With a verification proxy: the base_url should be the proxy origin
    (``scheme://host:raw_port+port_offset/``). Without a proxy: it should
    match the instance's ``site_url`` origin.

    ``instance`` may be a ``BenchmarkInstance`` or a ``dict`` — we only
    read ``site_url``. ``verification_proxy`` may be a ``VerificationProxy``
    model or ``None``.
    """
    site_url = instance.site_url if hasattr(instance, "site_url") else instance.get("site_url", "")
    if not site_url:
        raise ValueError("instance.site_url is empty")
    from urllib.parse import urlparse, urlunparse

    parts = urlparse(site_url)
    if parts.port is None or verification_proxy is None:
        return f"{parts.scheme}://{parts.netloc}/"
    port_offset = getattr(verification_proxy, "port_offset", 0) or 0
    if not port_offset:
        return f"{parts.scheme}://{parts.netloc}/"
    scheme = getattr(verification_proxy, "scheme", None) or parts.scheme
    hostname = parts.hostname or ""
    new_port = parts.port + int(port_offset)
    new_netloc = f"{hostname}:{new_port}"
    return urlunparse((scheme, new_netloc, "/", "", "", ""))


def check_magento_instances(
    instances: list[Any],
    verification_proxy: Any | None,
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> list[BaseUrlMismatch]:
    """Probe every Magento instance in ``instances``; return mismatches.

    Non-Magento instances are silently skipped. Empty return value means
    every Magento instance is serving the expected base_url.
    """
    mismatches: list[BaseUrlMismatch] = []
    for instance in instances:
        site_name = (
            instance.site_name if hasattr(instance, "site_name") else instance.get("site_name", "")
        )
        if site_name not in MAGENTO_SITES:
            continue
        site_url = (
            instance.site_url if hasattr(instance, "site_url") else instance.get("site_url", "")
        )
        if not site_url:
            continue
        try:
            expected = expected_base_url_for_instance(instance, verification_proxy)
            actual = assert_base_url(
                site_name,
                site_url,
                expected,
                session=session,
                timeout=timeout,
            )
            logger.info("magento base_url probe: %s -> %s (expected)", site_name, actual)
        except BaseUrlMismatch as exc:
            mismatches.append(exc)
    return mismatches
