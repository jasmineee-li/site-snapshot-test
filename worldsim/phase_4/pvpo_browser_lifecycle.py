"""Lifecycle controls for dedicated PVPO chrome-headless-shell endpoints.

Phase 4 assigns each worker a dedicated external CDP endpoint.  Browser Use
sessions are task-scoped, but the remote Chrome process behind ``cdp_url``
survives those sessions.  With ``--enable-begin-frame-control`` and the
per-step PVPO frame pump, a dirty remote renderer can keep consuming CPU after
the task ends.  The only reliable task boundary is therefore the browser
process, not a tab close.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_RECYCLE_ENV = "WORLDSIM_PVPO_BROWSER_RECYCLE"
_RECYCLE_TIMEOUT_ENV = "WORLDSIM_PVPO_BROWSER_RECYCLE_TIMEOUT_S"
_DEFAULT_RECYCLE_TIMEOUT_S = 20.0
_CLOSE_TIMEOUT_S = 3.0
_POLL_INTERVAL_S = 0.1


def pvpo_browser_recycle_enabled() -> bool:
    """Return whether external PVPO browsers should be hard-recycled per task."""
    raw = os.environ.get(_RECYCLE_ENV, "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def pvpo_browser_recycle_timeout_s() -> float:
    """Return the endpoint restart wait budget."""
    raw = os.environ.get(_RECYCLE_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_RECYCLE_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not numeric; using %.1fs",
            _RECYCLE_TIMEOUT_ENV,
            raw,
            _DEFAULT_RECYCLE_TIMEOUT_S,
        )
        return _DEFAULT_RECYCLE_TIMEOUT_S
    if value <= 0:
        logger.warning(
            "%s=%r is not positive; using %.1fs",
            _RECYCLE_TIMEOUT_ENV,
            raw,
            _DEFAULT_RECYCLE_TIMEOUT_S,
        )
        return _DEFAULT_RECYCLE_TIMEOUT_S
    return value


async def recycle_pvpo_browser_after_task(
    session: Any,
    cdp_url: str,
    *,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Hard-recycle a dedicated PVPO browser endpoint after one task.

    The function is intentionally CDP-only.  It sends ``Browser.close`` to the
    remote endpoint, then waits for ``/json/version`` to disappear and return.
    On the rigor hosts, the ``pvpo-chrome-*`` Docker containers have
    ``--restart unless-stopped``, so closing Chrome exits the container entrypoint
    and Docker immediately recreates a clean browser process on the same port.

    Returns a small artifact payload suitable for ``browser_runtime.json``.
    """
    started = time.monotonic()
    payload: dict[str, Any] = {
        "recycle_enabled": pvpo_browser_recycle_enabled(),
        "recycle_cdp_url": cdp_url,
        "recycle_status": "disabled",
        "recycle_down_observed": False,
        "recycle_up_observed": False,
        "recycle_elapsed_s": 0.0,
    }
    if not payload["recycle_enabled"] or not cdp_url:
        return payload

    close_error: str | None = None
    try:
        await asyncio.wait_for(_send_browser_close(session), timeout=_CLOSE_TIMEOUT_S)
    except Exception as exc:
        # Browser.close commonly severs the socket before the client receives a
        # clean response.  Treat this as diagnostic, not fatal; the endpoint
        # down/up observation below is the real source of truth.
        close_error = f"{type(exc).__name__}: {exc}"
        logger.debug("pvpo browser recycle: Browser.close returned %s", close_error)

    budget = timeout_s if timeout_s is not None else pvpo_browser_recycle_timeout_s()
    down_timeout = min(max(budget * 0.4, 2.0), 8.0)
    down_observed = await _wait_for_endpoint_state(
        cdp_url,
        reachable=False,
        timeout_s=down_timeout,
    )
    remaining = max(0.1, budget - (time.monotonic() - started))
    up_observed = await _wait_for_endpoint_state(
        cdp_url,
        reachable=True,
        timeout_s=remaining,
    )

    status = "recycled" if up_observed else "failed"
    if up_observed and not down_observed:
        status = "recycled_unconfirmed"
        logger.warning(
            "pvpo browser recycle did not observe %s go down before it became reachable; "
            "the endpoint may not have restarted",
            cdp_url,
        )
    elif not up_observed:
        logger.warning(
            "pvpo browser recycle failed to observe %s return within %.1fs; "
            "check pvpo-chrome restart policy and container logs",
            cdp_url,
            budget,
        )

    payload.update(
        {
            "recycle_status": status,
            "recycle_down_observed": down_observed,
            "recycle_up_observed": up_observed,
            "recycle_elapsed_s": round(time.monotonic() - started, 3),
        }
    )
    if close_error:
        payload["recycle_close_error"] = close_error
    return payload


async def _send_browser_close(session: Any) -> None:
    cdp_client = getattr(session, "cdp_client", None)
    sender_root = getattr(cdp_client, "send", None)
    browser_sender = getattr(sender_root, "Browser", None)
    close = getattr(browser_sender, "close", None)
    if callable(close):
        try:
            await close()
        except TypeError:
            await close(params={})
        return

    # Fallback for session objects that expose only target-scoped sessions.
    get_current_page = getattr(session, "get_current_page", None)
    get_or_create = getattr(session, "get_or_create_cdp_session", None)
    if not callable(get_current_page) or not callable(get_or_create):
        raise TypeError("PVPO browser recycle requires cdp_client.send.Browser.close")
    page = await get_current_page()
    target_id = getattr(page, "_target_id", None)
    if not target_id:
        raise TypeError("PVPO browser recycle could not find a page target id")
    cdp_session = await get_or_create(target_id=target_id, focus=False)
    from worldsim.phase_4.pvpo_cdp import normalize_cdp_session

    await normalize_cdp_session(cdp_session).send("Browser.close", {})


async def _wait_for_endpoint_state(
    cdp_url: str,
    *,
    reachable: bool,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await _endpoint_reachable(cdp_url) is reachable:
            return True
        await asyncio.sleep(_POLL_INTERVAL_S)
    return False


async def _endpoint_reachable(cdp_url: str) -> bool:
    return await asyncio.to_thread(_endpoint_reachable_sync, cdp_url)


def _endpoint_reachable_sync(cdp_url: str) -> bool:
    version_url = f"{cdp_url.rstrip('/')}/json/version"
    try:
        with urlopen(version_url, timeout=1.0) as resp:
            return 200 <= int(resp.status) < 300
    except (OSError, URLError, TimeoutError, ValueError):
        return False
