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


# ---------------------------------------------------------------------------
# Pending-review backstop (Layer 3 of the 2026-04-21 long-term fix)
# ---------------------------------------------------------------------------

# Adversarial seeds use this nickname prefix in the current dataset. Phase 2c's
# render check (Layer 2) is the primary gate; this backstop catches the corner
# cases where a seeded review was approved at Phase 2c time but then manually
# moderated back to Pending before Phase 4 launches, or where the operator
# hand-applied a task dataset that bypassed Phase 2c. One DB query per Magento
# instance, total ~tens of ms.
_SEED_NICKNAME_PATTERN = "SeedNickAdv%"
_MAGENTO_PENDING_STATUS_ID = 2

# Maximum review IDs surfaced per site in the error detail. Keeps the error
# readable even when a dataset gone wrong left thousands of pending rows.
_MAX_PENDING_IDS_REPORTED = 5


class PendingSeedReviewsError(RuntimeError):
    """A Magento instance has pending-approval seeded reviews before Phase 4.

    If any are found, Phase 4 would re-run against tasks whose payloads do
    NOT render (because status_id=2 reviews stay off the storefront PDP),
    silently reproducing the 2026-04-21 misclassification bug. Refuse to
    launch until the operator either approves them manually or re-runs
    Phase 2c with ``--force-reverify`` to repair the seed state.
    """

    def __init__(
        self,
        *,
        site_name: str,
        count: int,
        sample_ids: list[int],
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.site_name = site_name
        self.count = count
        self.sample_ids = sample_ids


def _mysql_connect_from_dsn(db_connection: str) -> Any:
    """Open a pymysql connection from a ``mysql://user:pw@host:port/db`` URL.

    Separate helper so tests can monkey-patch it out without pulling in the
    full ``parse_supported_db_connection`` dependency (which is also the
    canonical parser used by rewards + seeding — see
    ``worldsim/db_urls.py``).
    """
    from worldsim.db_urls import parse_supported_db_connection

    parsed = parse_supported_db_connection(
        db_connection,
        purpose="phase 4 magento_health pending-review backstop",
    )
    scheme = parsed.scheme.lower()
    if scheme != "mysql":
        raise RuntimeError(
            f"magento pending-review backstop only supports mysql:// DSNs; got {scheme!r}"
        )
    import pymysql

    return pymysql.connect(
        host=parsed.hostname,
        port=parsed.port or 3306,
        user=parsed.username,
        password=parsed.password,
        database=(parsed.path or "").lstrip("/"),
    )


def check_pending_seed_reviews_mysql(
    site_name: str,
    db_connection: str,
) -> PendingSeedReviewsError | None:
    """Return an error when any ``SeedNickAdv*`` reviews are still Pending.

    Queries Magento's ``review`` + ``review_detail`` tables for rows with
    ``status_id=2`` and ``nickname LIKE 'SeedNickAdv%'``. Returns ``None``
    when the query succeeds and finds nothing. Non-None means Phase 4
    must refuse to launch.

    A ``pymysql`` / DB error during the probe propagates as an ordinary
    exception (wrapped in a RuntimeError by the caller) — fail-loud, not
    fail-soft, because a DB probe failure while Magento is supposedly
    healthy enough for Phase 4 is itself a stop-the-line signal.
    """
    conn = _mysql_connect_from_dsn(db_connection)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT r.review_id, rd.nickname "
                "FROM review r JOIN review_detail rd ON r.review_id = rd.review_id "
                "WHERE r.status_id = %s AND rd.nickname LIKE %s "
                "ORDER BY r.review_id ASC",
                (_MAGENTO_PENDING_STATUS_ID, _SEED_NICKNAME_PATTERN),
            )
            rows = cur.fetchall() or []
    finally:
        try:
            conn.close()
        except Exception:
            logger.exception("magento pending-review backstop: failed to close mysql conn")

    if not rows:
        return None

    count = len(rows)
    sample_ids = [int(row[0]) for row in rows[:_MAX_PENDING_IDS_REPORTED]]
    detail = (
        f"{site_name}: {count} pending-approval review(s) with nickname "
        f"matching {_SEED_NICKNAME_PATTERN!r} found before Phase 4 launch. "
        f"First {len(sample_ids)} review_id(s): {sample_ids}. "
        "Phase 4 would mislabel these trajectories as resistant_unaware "
        "because PVPO would see no payload on the storefront (2026-04-21 "
        "bug class). Repair options: (a) re-run Phase 2c with "
        "`--force-reverify` so the editor's auto-approval logic re-applies, "
        "or (b) approve manually via Magento admin and re-launch."
    )
    return PendingSeedReviewsError(
        site_name=site_name,
        count=count,
        sample_ids=sample_ids,
        detail=detail,
    )


def check_pending_seed_reviews(instances: list[Any]) -> list[PendingSeedReviewsError]:
    """Probe every Magento instance in ``instances`` for pending seed reviews.

    Non-Magento instances are silently skipped. Magento instances without a
    ``db_connection`` are logged as WARNING and skipped — Phase 4 still
    launches because Layer 2's render check is the primary gate; this
    backstop is defense-in-depth.
    """
    errors: list[PendingSeedReviewsError] = []
    for instance in instances:
        site_name = (
            instance.site_name if hasattr(instance, "site_name") else instance.get("site_name", "")
        )
        if site_name not in MAGENTO_SITES:
            continue
        db_connection = (
            instance.db_connection
            if hasattr(instance, "db_connection")
            else instance.get("db_connection")
        )
        if not db_connection:
            logger.warning(
                "magento pending-review backstop: %s has no db_connection; "
                "skipping backstop probe (Phase 2c render check is the primary gate)",
                site_name,
            )
            continue
        outcome = check_pending_seed_reviews_mysql(site_name, str(db_connection))
        if outcome is not None:
            errors.append(outcome)
        else:
            logger.info(
                "magento pending-review backstop: %s clean (no SeedNickAdv*%% pending)",
                site_name,
            )
    return errors
