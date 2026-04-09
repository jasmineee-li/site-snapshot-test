"""Trusted helper commands intended to run inside redteam sandboxes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.eval_harness import (
    StateSnapshotServer,
    _read_json,
    get_server_process_diagnostics,
    load_verifier,
    start_app_server,
    stop_app_server,
)
from agentlab.benchmarks.redteam.app_artifacts import is_normalized_state_payload
from agentlab.benchmarks.redteam.runtime_ops import ensure_local_seed_materialization


def _bundle_allowlist(root: Path, patterns: list[str]) -> dict[str, object]:
    matched_files: list[Path] = []
    for pattern in patterns:
        matched_files.extend(path for path in root.glob(pattern) if path.is_file())
        matched_files.extend(path for path in root.glob(pattern) if path.is_dir())

    expanded: set[Path] = set()
    for path in matched_files:
        if path.is_file():
            expanded.add(path)
        elif path.is_dir():
            expanded.update(child for child in path.rglob("*") if child.is_file())

    rel_files = sorted(path.relative_to(root).as_posix() for path in expanded)
    if not rel_files:
        return {"files": [], "tar_gz_base64": ""}

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for rel_path in rel_files:
            archive.add(root / rel_path, arcname=rel_path)

    return {
        "files": rel_files,
        "tar_gz_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _sha256_file(root: Path, relative_path: str) -> dict[str, object]:
    target = root / relative_path
    if not target.exists():
        return {"sha256": None}
    return {"sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def _verify_snapshot(root: Path, verify_path: str, snapshot_path: str) -> dict[str, object]:
    snapshot = _read_json(root / snapshot_path, default={}) or {}
    with StateSnapshotServer(snapshot) as server:
        verify_fn = load_verifier(root, verify_path)
        passed, message = verify_fn(server.server_url)
    return {"passed": bool(passed), "message": str(message)}


def _make_check_payload(
    check_name: str,
    *,
    passed: bool,
    duration_ms: float,
    details: str | None = None,
    skipped: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "check_name": check_name,
        "passed": bool(passed),
        "duration_ms": round(duration_ms, 3),
        "skipped": bool(skipped),
    }
    if details:
        payload["details"] = str(details)
    return payload


def _skipped_check_payload(check_name: str, details: str) -> dict[str, object]:
    return _make_check_payload(
        check_name,
        passed=False,
        duration_ms=0.0,
        details=details,
        skipped=True,
    )


def _state_error_summary(payload: dict[str, object]) -> str | None:
    for name in ("app_load", "state_api"):
        check = payload.get(name)
        if not isinstance(check, dict):
            continue
        if check.get("skipped"):
            continue
        if check.get("passed"):
            continue
        details = check.get("details")
        return f"{name}: {details}" if details else name
    return None


def _server_diagnostics_payload(server_proc) -> dict[str, object]:
    diagnostics = get_server_process_diagnostics(server_proc) if server_proc is not None else {}
    if not isinstance(diagnostics, dict):
        return {}
    return {
        "stdout_log_path": diagnostics.get("stdout_log_path", "") or "",
        "stderr_log_path": diagnostics.get("stderr_log_path", "") or "",
        "stdout_tail": diagnostics.get("stdout_tail", "") or "",
        "stderr_tail": diagnostics.get("stderr_tail", "") or "",
    }


def _trim_list(values: list[str], limit: int = 3) -> list[str]:
    return [str(value) for value in values[:limit] if str(value)]


def _browser_diagnostics_payload(
    *,
    console_errors: list[str],
    page_errors: list[str],
    request_failures: list[str],
) -> dict[str, object]:
    return {
        "console_errors": _trim_list(console_errors),
        "page_errors": _trim_list(page_errors),
        "request_failures": _trim_list(request_failures),
    }


def _combine_runtime_details(
    message: str | None,
    *,
    server_diagnostics: dict[str, object] | None = None,
    browser_diagnostics: dict[str, object] | None = None,
) -> str:
    parts: list[str] = []
    if message:
        parts.append(str(message))
    if server_diagnostics:
        stderr_tail = str(server_diagnostics.get("stderr_tail", "") or "")
        stdout_tail = str(server_diagnostics.get("stdout_tail", "") or "")
        if stderr_tail:
            parts.append(f"server stderr: {stderr_tail}")
        if stdout_tail:
            parts.append(f"server stdout: {stdout_tail}")
    if browser_diagnostics:
        console_errors = browser_diagnostics.get("console_errors") or []
        page_errors = browser_diagnostics.get("page_errors") or []
        request_failures = browser_diagnostics.get("request_failures") or []
        if console_errors:
            parts.append(f"browser console: {'; '.join(str(item) for item in console_errors)}")
        if page_errors:
            parts.append(f"browser page errors: {'; '.join(str(item) for item in page_errors)}")
        if request_failures:
            parts.append(f"browser request failures: {'; '.join(str(item) for item in request_failures)}")
    return "; ".join(part for part in parts if part)


@contextmanager
def _temporary_candidate_data_js(root: Path, candidate_path: str):
    data_js_path = ensure_local_seed_materialization(root, variant_subdir="benign")
    candidate = root / candidate_path
    if not candidate.exists():
        raise FileNotFoundError(f"Candidate data.js not found: {candidate}")

    original_link_target: str | None = None
    original_content: str | None = None
    if data_js_path.is_symlink():
        original_link_target = data_js_path.readlink().as_posix()
    elif data_js_path.exists():
        original_content = data_js_path.read_text(encoding="utf-8")

    try:
        if data_js_path.exists() or data_js_path.is_symlink():
            data_js_path.unlink()
        try:
            data_js_path.symlink_to(Path("..") / candidate.relative_to(root))
        except OSError:
            data_js_path.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
        yield data_js_path
    finally:
        if data_js_path.exists() or data_js_path.is_symlink():
            data_js_path.unlink()
        if original_link_target is not None:
            data_js_path.symlink_to(original_link_target)
        elif original_content is not None:
            data_js_path.write_text(original_content, encoding="utf-8")


def _validate_app_runtime(root: Path, port: int | None = None) -> dict[str, object]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    diagnostics: dict[str, object] = {}
    server_proc = None

    try:
        server_proc, server_port = start_app_server(root, port=port, timeout=5)
        base_url = f"http://127.0.0.1:{server_port}"
        checks["server:startup"] = True

        try:
            req = urllib.request.Request(base_url + "/", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                checks["http:GET /"] = resp.status == 200
                if resp.status != 200:
                    errors.append(f"GET / returned {resp.status}")
        except Exception as exc:
            checks["http:GET /"] = False
            errors.append(f"GET / failed: {exc}")

        test_state = json.dumps({"test": True, "_seedVersion": 1}).encode("utf-8")
        try:
            put_req = urllib.request.Request(
                base_url + "/api/state",
                data=test_state,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(put_req, timeout=5) as resp:
                put_ok = resp.status in (200, 204)
                checks["http:PUT /api/state"] = put_ok
                if not put_ok:
                    errors.append(f"PUT /api/state returned {resp.status}")
        except Exception as exc:
            checks["http:PUT /api/state"] = False
            errors.append(f"PUT /api/state failed: {exc}")

        try:
            get_req = urllib.request.Request(base_url + "/api/state", method="GET")
            with urllib.request.urlopen(get_req, timeout=5) as resp:
                get_ok = resp.status == 200
                checks["http:GET /api/state"] = get_ok
                if get_ok:
                    body = json.loads(resp.read().decode("utf-8"))
                    state_match = is_normalized_state_payload(body) and body.get("test") is True
                    checks["state:round-trip"] = state_match
                    if not state_match:
                        errors.append(
                            "State round-trip failed: PUT data not returned by GET as normalized state"
                        )
                else:
                    errors.append(f"GET /api/state returned {resp.status}")
        except Exception as exc:
            checks["http:GET /api/state"] = False
            errors.append(f"GET /api/state failed: {exc}")
    except Exception as exc:
        checks["server:startup"] = False
        errors.append(f"Server startup failed: {exc}")
    finally:
        if server_proc is not None:
            stop_app_server(server_proc)
            server_diagnostics = _server_diagnostics_payload(server_proc)
            if any(server_diagnostics.values()):
                diagnostics["server"] = server_diagnostics

    payload: dict[str, object] = {
        "passed": len(errors) == 0,
        "checks": checks,
        "errors": errors,
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def _check_app_load(server_port: int, *, timeout_ms: int) -> tuple[dict[str, object], dict[str, object]]:
    start = time.perf_counter()
    console_errors: list[str] = []
    page_errors: list[str] = []
    request_failures: list[str] = []
    playwright = None
    browser = None
    page = None
    try:
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        def _record_console(msg):
            if getattr(msg, "type", "") == "error":
                console_errors.append(str(getattr(msg, "text", "")))

        def _record_pageerror(exc):
            page_errors.append(str(exc))

        def _record_request_failed(request):
            if "favicon.ico" in str(getattr(request, "url", "")):
                return
            failure = getattr(request, "failure", None)
            request_failures.append(f"{getattr(request, 'url', '')} ({failure})")

        page.on("console", _record_console)
        page.on("pageerror", _record_pageerror)
        page.on("requestfailed", _record_request_failed)

        page.goto(
            f"http://127.0.0.1:{server_port}/",
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        page.wait_for_timeout(500)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        browser_diagnostics = _browser_diagnostics_payload(
            console_errors=console_errors,
            page_errors=page_errors,
            request_failures=request_failures,
        )
        return (
            _make_check_payload(
                "app_load",
                passed=False,
                duration_ms=duration_ms,
                details=f"App failed to load: {exc}",
            ),
            browser_diagnostics,
        )
    finally:
        if page is not None:
            page.close()
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()

    duration_ms = (time.perf_counter() - start) * 1000
    browser_diagnostics = _browser_diagnostics_payload(
        console_errors=console_errors,
        page_errors=page_errors,
        request_failures=request_failures,
    )
    browser_errors = (
        list(browser_diagnostics["console_errors"])
        + list(browser_diagnostics["page_errors"])
        + list(browser_diagnostics["request_failures"])
    )
    if browser_errors:
        return (
            _make_check_payload(
                "app_load",
                passed=False,
                duration_ms=duration_ms,
                details="; ".join(str(item) for item in browser_errors[:3]),
            ),
            browser_diagnostics,
        )
    return (
        _make_check_payload("app_load", passed=True, duration_ms=duration_ms),
        browser_diagnostics,
    )


def _check_state_api(server_port: int, *, timeout_ms: int) -> dict[str, object]:
    start = time.perf_counter()
    deadline = time.monotonic() + (timeout_ms / 1000)
    last_error: str | None = None

    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{server_port}/api/state", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not is_normalized_state_payload(payload):
                last_error = "GET /api/state returned unsupported payload shape"
            elif payload == {}:
                last_error = "GET /api/state returned empty state"
            else:
                json.dumps(payload)
                duration_ms = (time.perf_counter() - start) * 1000
                return _make_check_payload("state_api", passed=True, duration_ms=duration_ms)
        except urllib.error.HTTPError as exc:
            last_error = f"GET /api/state returned HTTP {exc.code}"
        except json.JSONDecodeError as exc:
            last_error = f"GET /api/state returned invalid JSON: {exc}"
            break
        except Exception as exc:
            last_error = f"GET /api/state failed: {exc}"
        time.sleep(0.1)

    duration_ms = (time.perf_counter() - start) * 1000
    return _make_check_payload(
        "state_api",
        passed=False,
        duration_ms=duration_ms,
        details=last_error or "GET /api/state did not become available",
    )


def _validate_data_runtime(
    root: Path,
    *,
    candidate_path: str,
    timeout_ms: int,
) -> dict[str, object]:
    server_proc = None
    diagnostics: dict[str, object] = {}
    try:
        with _temporary_candidate_data_js(root, candidate_path):
            server_proc, server_port = start_app_server(
                root,
                timeout=timeout_ms / 1000,
            )
            app_load_payload, browser_diagnostics = _check_app_load(server_port, timeout_ms=timeout_ms)
            server_diagnostics = _server_diagnostics_payload(server_proc)
            if any(server_diagnostics.values()):
                diagnostics["server"] = server_diagnostics
            if any(browser_diagnostics.values()):
                diagnostics["browser"] = browser_diagnostics

            if not app_load_payload.get("passed"):
                app_load_payload["details"] = _combine_runtime_details(
                    app_load_payload.get("details"),
                    server_diagnostics=server_diagnostics,
                    browser_diagnostics=browser_diagnostics,
                )
                payload: dict[str, object] = {
                    "passed": False,
                    "app_load": app_load_payload,
                    "state_api": _skipped_check_payload("state_api", "app load failed"),
                }
                error_summary = _state_error_summary(payload)
                if error_summary:
                    payload["error_summary"] = error_summary
                if diagnostics:
                    payload["diagnostics"] = diagnostics
                return payload

            state_api_payload = _check_state_api(server_port, timeout_ms=timeout_ms)
            if not state_api_payload.get("passed"):
                state_api_payload["details"] = _combine_runtime_details(
                    state_api_payload.get("details"),
                    server_diagnostics=server_diagnostics,
                    browser_diagnostics=browser_diagnostics,
                )
            payload = {
                "passed": bool(state_api_payload.get("passed")),
                "app_load": app_load_payload,
                "state_api": state_api_payload,
            }
            error_summary = _state_error_summary(payload)
            if error_summary:
                payload["error_summary"] = error_summary
            if diagnostics:
                payload["diagnostics"] = diagnostics
            return payload
    except Exception as exc:
        server_diagnostics = _server_diagnostics_payload(server_proc)
        if any(server_diagnostics.values()):
            diagnostics["server"] = server_diagnostics
        payload = {
            "passed": False,
            "app_load": _make_check_payload(
                "app_load",
                passed=False,
                duration_ms=0.0,
                details=_combine_runtime_details(
                    f"App validation setup failed: {exc}",
                    server_diagnostics=server_diagnostics,
                ),
            ),
            "state_api": _skipped_check_payload("state_api", "app load failed"),
        }
        error_summary = _state_error_summary(payload)
        if error_summary:
            payload["error_summary"] = error_summary
        if diagnostics:
            payload["diagnostics"] = diagnostics
        return payload
    finally:
        if server_proc is not None:
            stop_app_server(server_proc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("bundle-allowlist")
    bundle.add_argument("--root", required=True)
    bundle.add_argument("--pattern", action="append", default=[])

    sha = subparsers.add_parser("sha256-file")
    sha.add_argument("--root", required=True)
    sha.add_argument("--path", required=True)

    verify = subparsers.add_parser("verify-snapshot")
    verify.add_argument("--root", required=True)
    verify.add_argument("--verify-path", required=True)
    verify.add_argument("--snapshot-path", required=True)

    validate_app = subparsers.add_parser("validate-app-runtime")
    validate_app.add_argument("--root", required=True)
    validate_app.add_argument("--port", type=int)

    validate_data = subparsers.add_parser("validate-data-runtime")
    validate_data.add_argument("--root", required=True)
    validate_data.add_argument("--candidate-path", required=True)
    validate_data.add_argument("--timeout-ms", type=int, default=10_000)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(getattr(args, "root"))

    if args.command == "bundle-allowlist":
        payload = _bundle_allowlist(root, args.pattern)
    elif args.command == "sha256-file":
        payload = _sha256_file(root, args.path)
    elif args.command == "verify-snapshot":
        payload = _verify_snapshot(root, args.verify_path, args.snapshot_path)
    elif args.command == "validate-app-runtime":
        payload = _validate_app_runtime(root, port=args.port)
    elif args.command == "validate-data-runtime":
        payload = _validate_data_runtime(
            root,
            candidate_path=args.candidate_path,
            timeout_ms=args.timeout_ms,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
