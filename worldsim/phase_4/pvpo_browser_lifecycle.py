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
import contextlib
import logging
import os
import shutil
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_RECYCLE_ENV = "WORLDSIM_PVPO_BROWSER_RECYCLE"
_RECYCLE_MODE_ENV = "WORLDSIM_PVPO_BROWSER_RECYCLE_MODE"
_RECYCLE_TIMEOUT_ENV = "WORLDSIM_PVPO_BROWSER_RECYCLE_TIMEOUT_S"
_DEFAULT_RECYCLE_TIMEOUT_S = 20.0
_CLOSE_TIMEOUT_S = 3.0
_POLL_INTERVAL_S = 0.1
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


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

    Rigor hosts manage loopback PVPO endpoints as ``pvpo-chrome-<port>`` Docker
    containers.  For those endpoints, the authoritative reset is
    ``docker restart``; CDP ``Browser.close`` can disconnect clients without
    terminating the supervised ``chrome-headless-shell`` parent process.  CDP
    close remains a fallback for unmanaged/non-Docker endpoints.

    Returns a small artifact payload suitable for ``browser_runtime.json``.
    """
    started = time.monotonic()
    budget = timeout_s if timeout_s is not None else pvpo_browser_recycle_timeout_s()
    requested_mode = _recycle_mode()
    container_name = _managed_container_name_for_cdp_url(cdp_url)
    payload: dict[str, Any] = {
        "recycle_enabled": pvpo_browser_recycle_enabled(),
        "recycle_cdp_url": cdp_url,
        "recycle_mode": requested_mode,
        "recycle_method": None,
        "recycle_container": container_name,
        "recycle_status": "disabled",
        "recycle_down_observed": False,
        "recycle_up_observed": False,
        "recycle_elapsed_s": 0.0,
    }
    if not payload["recycle_enabled"] or not cdp_url:
        return payload

    if requested_mode in {"auto", "docker"} and container_name:
        before_probe = await _managed_container_process_probe(container_name)
        docker_payload = await _restart_managed_container(
            container_name,
            cdp_url,
            timeout_s=budget,
        )
        docker_payload["container_probe_before"] = before_probe
        docker_payload["container_probe_after"] = await _managed_container_process_probe(
            container_name
        )
        payload.update(docker_payload)
        if docker_payload.get("recycle_status") == "recycled":
            payload["recycle_elapsed_s"] = round(time.monotonic() - started, 3)
            return payload
        if requested_mode == "docker":
            payload["recycle_elapsed_s"] = round(time.monotonic() - started, 3)
            return payload

    if requested_mode == "docker" and not container_name:
        payload.update(
            {
                "recycle_method": "docker_restart",
                "recycle_status": "failed",
                "recycle_failure": f"no managed pvpo-chrome container convention for {cdp_url!r}",
                "recycle_elapsed_s": round(time.monotonic() - started, 3),
            }
        )
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

    payload["recycle_method"] = "cdp_browser_close"
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


def _recycle_mode() -> str:
    raw = os.environ.get(_RECYCLE_MODE_ENV, "").strip().lower()
    if raw in {"auto", "docker", "cdp"}:
        return raw
    if raw:
        logger.warning("%s=%r is invalid; using auto", _RECYCLE_MODE_ENV, raw)
    return "auto"


def _managed_container_name_for_cdp_url(cdp_url: str) -> str | None:
    try:
        parsed = urlparse(cdp_url)
    except Exception:
        return None
    host = (parsed.hostname or "").strip().lower()
    if host not in _LOOPBACK_HOSTS or parsed.port is None:
        return None
    return f"pvpo-chrome-{parsed.port}"


async def _restart_managed_container(
    container_name: str,
    cdp_url: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recycle_method": "docker_restart",
        "recycle_status": "failed",
        "recycle_down_observed": False,
        "recycle_up_observed": False,
    }
    if shutil.which("docker") is None:
        payload["recycle_failure"] = "docker executable not found"
        return payload

    restart_task = asyncio.create_task(
        _run_docker_restart(container_name, timeout_s=timeout_s),
        name=f"pvpo-docker-restart-{container_name}",
    )
    down_task = asyncio.create_task(
        _wait_for_endpoint_state(
            cdp_url,
            reachable=False,
            timeout_s=min(max(timeout_s * 0.5, 2.0), timeout_s),
        ),
        name=f"pvpo-wait-down-{container_name}",
    )
    try:
        docker_result = await restart_task
    finally:
        if not down_task.done():
            down_task.cancel()
    down_observed = False
    try:
        down_observed = await down_task
    except asyncio.CancelledError:
        down_observed = False

    payload.update(
        {
            "recycle_down_observed": down_observed,
            "docker_restart_returncode": docker_result["returncode"],
            "docker_restart_stdout": docker_result["stdout"][:500],
            "docker_restart_stderr": docker_result["stderr"][:500],
        }
    )
    if docker_result["returncode"] != 0:
        payload["recycle_failure"] = "docker restart failed"
        return payload

    up_observed = await _wait_for_endpoint_state(
        cdp_url,
        reachable=True,
        timeout_s=max(0.1, timeout_s),
    )
    payload["recycle_up_observed"] = up_observed
    if up_observed:
        payload["recycle_status"] = "recycled"
    else:
        payload["recycle_failure"] = "endpoint did not return after docker restart"
    return payload


async def _run_docker_restart(container_name: str, *, timeout_s: float) -> dict[str, Any]:
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "restart",
            container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        stdout, stderr = await process.communicate()
        return {
            "returncode": 124,
            "stdout": stdout.decode("utf-8", errors="replace").strip(),
            "stderr": (
                stderr.decode("utf-8", errors="replace").strip() or "docker restart timed out"
            ),
        }
    return {
        "returncode": int(process.returncode or 0),
        "stdout": stdout.decode("utf-8", errors="replace").strip(),
        "stderr": stderr.decode("utf-8", errors="replace").strip(),
    }


async def _managed_container_process_probe(container_name: str) -> dict[str, Any]:
    """Return lightweight process evidence for a managed PVPO Chrome container."""
    if shutil.which("docker") is None:
        return {"status": "unavailable", "reason": "docker executable not found"}
    inspect_result = await _run_docker_command(
        "inspect",
        "-f",
        "{{.State.Running}} {{.State.Pid}} {{.RestartCount}}",
        container_name,
        timeout_s=3.0,
    )
    top_result = await _run_docker_command(
        "top",
        container_name,
        "-eo",
        "pid,ppid,stat,comm",
        timeout_s=3.0,
    )
    payload: dict[str, Any] = {
        "status": "ok" if inspect_result["returncode"] == 0 else "failed",
        "inspect_returncode": inspect_result["returncode"],
        "inspect_stdout": inspect_result["stdout"][:200],
        "inspect_stderr": inspect_result["stderr"][:200],
        "top_returncode": top_result["returncode"],
        "top_stderr": top_result["stderr"][:200],
    }
    process_lines = [
        line
        for line in top_result["stdout"].splitlines()
        if line.strip() and not line.lower().startswith("pid")
    ]
    payload["process_count"] = len(process_lines)
    payload["chrome_process_count"] = sum(
        1 for line in process_lines if "chrome" in line.lower()
    )
    payload["process_sample"] = process_lines[:12]
    parts = inspect_result["stdout"].split()
    if len(parts) >= 3:
        payload["container_running"] = parts[0].lower() == "true"
        payload["container_init_pid"] = parts[1]
        payload["container_restart_count"] = parts[2]
    return payload


async def _run_docker_command(*args: str, timeout_s: float) -> dict[str, Any]:
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        stdout, stderr = await process.communicate()
        return {
            "returncode": 124,
            "stdout": stdout.decode("utf-8", errors="replace").strip(),
            "stderr": (
                stderr.decode("utf-8", errors="replace").strip() or "docker command timed out"
            ),
        }
    return {
        "returncode": int(process.returncode or 0),
        "stdout": stdout.decode("utf-8", errors="replace").strip(),
        "stderr": stderr.decode("utf-8", errors="replace").strip(),
    }


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
