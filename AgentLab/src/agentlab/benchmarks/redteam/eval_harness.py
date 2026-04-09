"""Functional correctness evaluation harness.

Provides:

1. Sanity-check execution for generated verifier/solver suites.
2. Browser-based functional evaluation for generated app tasks.
3. Snapshot-based verifier replay and canonical ``functional_results.json`` writing.
4. A standalone CLI for sanity checks, evaluation, and quality-gate checks.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from bgym import HighLevelActionSetArgs

from agentlab.benchmarks.redteam.app_artifacts import (
    fetch_normalized_state,
    load_app_manifest,
    require_supported_app_manifest,
)
from agentlab.benchmarks.redteam.agent_browser_runner import (
    AGENT_BROWSER_BACKEND,
    DEFAULT_AGENT_BROWSER_TIMEOUT,
    localize_start_url,
    run_agent_browser_task,
)
from agentlab.experiments.launch_exp import import_object, run_experiments
from agentlab.experiments.loop import ExpArgs
from agentlab.benchmarks.redteam.config import Page, RedteamEnvArgs
from agentlab.benchmarks.redteam.execution import copy_workspace_tree, run_python_script, run_trusted_worker
from agentlab.benchmarks.redteam.runtime_ops import ensure_local_seed_materialization

logger = logging.getLogger(__name__)

DEFAULT_AGENT_CONFIG = "agentlab.agents.generic_agent.AGENT_37_SONNET"
GENERIC_AGENT_BACKEND = "generic-agent"
DEFAULT_FUNCTIONAL_BACKEND = AGENT_BROWSER_BACKEND
SUPPORTED_FUNCTIONAL_BACKENDS = (GENERIC_AGENT_BACKEND, AGENT_BROWSER_BACKEND)
SERVER_LOG_TAIL_BYTES = 4096
SERVER_PORT_FILE_ENV = "SERVER_PORT_FILE"
SUITE_TO_JSON = {
    "function": "function-tasks",
    "real": "real-tasks",
    "function-tasks": "function-tasks",
    "real-tasks": "real-tasks",
}
JSON_TO_VARIANT = {
    "function-tasks": "function",
    "real-tasks": "real",
}
ALLOWED_VERIFIER_DIRS = ("function-tasks", "real-tasks")


class _FunctionalBenchmarkConfig:
    """Minimal benchmark surface needed by AgentLab agents."""

    def __init__(self, name: str):
        self.name = name
        self.is_multi_tab = False
        self.high_level_action_set_args = HighLevelActionSetArgs(
            subsets=["bid", "nav", "tab", "chat"],
            multiaction=False,
        )


def normalize_functional_backend(backend: str | None) -> str:
    """Normalize a functional-eval backend identifier."""
    normalized = (backend or DEFAULT_FUNCTIONAL_BACKEND).strip().lower()
    if normalized not in SUPPORTED_FUNCTIONAL_BACKENDS:
        raise ValueError(
            f"Unsupported functional backend: {backend}. "
            f"Expected one of {', '.join(SUPPORTED_FUNCTIONAL_BACKENDS)}."
        )
    return normalized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _python_executable() -> str:
    """Return the active Python executable, falling back to ``python3``."""
    return sys.executable or "python3"


def normalize_task_suite(task_suite: str) -> str:
    """Normalize ``function``/``real`` aliases to JSON filenames."""
    if task_suite not in SUITE_TO_JSON:
        raise ValueError(f"Unsupported task suite: {task_suite}")
    return SUITE_TO_JSON[task_suite]


def suite_variant_name(task_suite: str) -> str:
    """Return the sanity-check / CLI variant name for a suite."""
    normalized = normalize_task_suite(task_suite)
    return JSON_TO_VARIANT[normalized]


def _trusted_server_template_path() -> Path:
    return Path(__file__).resolve().parent / "templates" / "server.py"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_trusted_server_launch(app_dir: str | Path) -> None:
    template_path = _trusted_server_template_path()
    server_path = Path(app_dir) / "server.py"
    if not template_path.exists():
        raise RuntimeError(f"Trusted server template not found: {template_path}")
    if not server_path.exists():
        raise RuntimeError(f"Reserved runtime file missing: {server_path}")
    if _sha256_file(server_path) != _sha256_file(template_path):
        raise RuntimeError(
            "Reserved runtime file server.py failed trusted-template checksum validation before host launch."
        )


def resolve_verifier_path(
    app_dir: str | Path,
    verify_path: str,
) -> Path:
    app_root = Path(app_dir).resolve()
    raw_path = str(verify_path or "").strip()
    if not raw_path:
        raise ValueError("Verifier path is required.")
    verify_posix = PurePosixPath(raw_path)
    if verify_posix.is_absolute():
        raise ValueError(f"Verifier path must be relative: {raw_path}")
    if ".." in verify_posix.parts:
        raise ValueError(f"Verifier path may not escape with '..': {raw_path}")
    if verify_posix.suffix != ".py":
        raise ValueError(f"Verifier path must target a Python file: {raw_path}")
    if not verify_posix.parts or verify_posix.parts[0] not in ALLOWED_VERIFIER_DIRS:
        raise ValueError(
            f"Verifier path must stay under {', '.join(ALLOWED_VERIFIER_DIRS)}: {raw_path}"
        )

    joined = app_root / verify_posix.as_posix()
    if not joined.exists():
        raise ValueError(f"Verifier path does not exist: {raw_path}")
    if joined.is_symlink():
        raise ValueError(f"Verifier path may not be a symlink: {raw_path}")

    resolved = joined.resolve()
    try:
        resolved_relative = resolved.relative_to(app_root)
    except ValueError as exc:
        raise ValueError(f"Verifier path escapes app root: {raw_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"Verifier path is not a file: {raw_path}")
    if not resolved_relative.parts or resolved_relative.parts[0] not in ALLOWED_VERIFIER_DIRS:
        raise ValueError(
            f"Verifier path must resolve under {', '.join(ALLOWED_VERIFIER_DIRS)}: {raw_path}"
        )
    return resolved


def _read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(
    app_dir: str | Path,
    task_suite: str = "function-tasks",
) -> list[dict[str, Any]]:
    """Load tasks from ``{task_suite}.json`` in *app_dir*."""
    task_suite = normalize_task_suite(task_suite)
    path = Path(app_dir) / f"{task_suite}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_tasks(
    tasks: list[dict[str, Any]],
    task_id: str | None = None,
    difficulty: str | None = None,
) -> list[dict[str, Any]]:
    """Filter tasks by ID (comma-separated) or difficulty level."""
    if task_id:
        ids = {s.strip() for s in task_id.split(",") if s.strip()}
        return [t for t in tasks if t["id"] in ids]
    if difficulty:
        return [t for t in tasks if t.get("difficulty") == difficulty]
    return tasks


def load_verifier(
    app_dir: str | Path,
    verify_path: str,
) -> Callable[[str], tuple[bool, str]]:
    """Dynamically import a verifier module and return its ``verify`` function."""
    resolved_path = resolve_verifier_path(app_dir, verify_path)
    spec = importlib.util.spec_from_file_location("verifier", str(resolved_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load verifier: {resolved_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify


def run_verifier(
    server_url: str,
    app_dir: str | Path,
    verify_path: str,
) -> tuple[bool, str]:
    """Load and execute a verifier, catching exceptions."""
    try:
        verify_fn = load_verifier(app_dir, verify_path)
        return verify_fn(server_url)
    except Exception as exc:
        return False, f"Verifier exception: {exc}"


class _StateSnapshotHandler(BaseHTTPRequestHandler):
    """Serve a saved final-state snapshot on ``/api/state`` for verifier replay."""

    snapshot: dict[str, Any] | None = None

    def do_GET(self):
        if self.path != "/api/state":
            self.send_error(404)
            return
        body = json.dumps(type(self).snapshot or {}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class StateSnapshotServer:
    """Context manager exposing a saved state dict at ``/api/state``."""

    def __init__(self, snapshot: dict[str, Any]):
        self.snapshot = snapshot
        self.port = _find_free_port()
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", self.port),
            _StateSnapshotHandler,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def server_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        _StateSnapshotHandler.snapshot = self.snapshot
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        _StateSnapshotHandler.snapshot = None


def run_verifier_on_snapshot(
    app_dir: str | Path,
    verify_path: str,
    snapshot: dict[str, Any] | str | Path,
) -> tuple[bool, str]:
    """Replay a verifier against a saved final-state snapshot."""
    app_dir = Path(app_dir)
    resolved_verify_path = resolve_verifier_path(app_dir, verify_path)
    if isinstance(snapshot, (str, Path)):
        snapshot = _read_json(snapshot, default={}) or {}

    with tempfile.TemporaryDirectory(prefix="redteam-verifier-") as tmpdir:
        staged_app_dir = Path(tmpdir) / "app"
        copy_workspace_tree(
            app_dir,
            staged_app_dir,
            ignore_patterns=("__pycache__", "*.pyc"),
        )
        snapshot_dir = staged_app_dir / ".sandbox_inputs"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / "final_state.json"
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=False),
            encoding="utf-8",
        )
        result = run_trusted_worker(
            staged_app_dir,
            "verify-snapshot",
            timeout=300,
            argv=[
                "--root",
                ".",
                "--verify-path",
                resolved_verify_path.relative_to(app_dir).as_posix(),
                "--snapshot-path",
                ".sandbox_inputs/final_state.json",
            ],
            block_network=True,
        )
    if result.returncode != 0:
        error_output = result.stderr or result.stdout or "Verifier execution failed."
        return False, error_output
    payload = json.loads(result.stdout or "{}")
    return bool(payload.get("passed")), str(payload.get("message", ""))


def _read_server_log_tail(path: str | None, max_bytes: int = SERVER_LOG_TAIL_BYTES) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()


def _read_server_port_file(path: str | None) -> int | None:
    if not path:
        return None
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def get_server_process_diagnostics(proc: subprocess.Popen) -> dict[str, str]:
    """Return cached or live diagnostics for a managed app server process."""
    cached = getattr(proc, "_redteam_server_logs", None)
    if isinstance(cached, dict):
        return dict(cached)

    stdout_path = getattr(proc, "_redteam_server_stdout_log", None)
    stderr_path = getattr(proc, "_redteam_server_stderr_log", None)
    return {
        "stdout_log_path": stdout_path or "",
        "stderr_log_path": stderr_path or "",
        "stdout_tail": _read_server_log_tail(stdout_path),
        "stderr_tail": _read_server_log_tail(stderr_path),
    }


def _finalize_server_log_capture(proc: subprocess.Popen) -> None:
    diagnostics = get_server_process_diagnostics(proc)
    setattr(proc, "_redteam_server_logs", diagnostics)
    for attr in (
        "_redteam_server_stdout_log",
        "_redteam_server_stderr_log",
        "_redteam_server_port_file",
    ):
        path = getattr(proc, attr, None)
        if path:
            Path(path).unlink(missing_ok=True)


def _format_server_diagnostics(diagnostics: dict[str, str]) -> str:
    parts: list[str] = []
    if diagnostics.get("stderr_tail"):
        parts.append(f"stderr tail: {diagnostics['stderr_tail']}")
    if diagnostics.get("stdout_tail"):
        parts.append(f"stdout tail: {diagnostics['stdout_tail']}")
    return " | ".join(parts)


def _format_server_startup_failure(proc: subprocess.Popen, message: str) -> str:
    details = _format_server_diagnostics(get_server_process_diagnostics(proc))
    if not details:
        return message
    return f"{message}. {details}"


def start_app_server(
    app_dir: str | Path,
    port: int | None = None,
    timeout: float = 10.0,
    env: Mapping[str, str] | None = None,
) -> tuple[subprocess.Popen, int]:
    """Start the app's ``server.py`` and wait for it to be ready."""
    app_dir = Path(app_dir)
    if (app_dir / "benign" / "data.js").exists() or (app_dir / "js" / "data.js").exists():
        ensure_local_seed_materialization(app_dir, variant_subdir="benign")
    validate_trusted_server_launch(app_dir)
    requested_port = 0 if port is None else port

    stdout_log = tempfile.NamedTemporaryFile(
        prefix="redteam-server-stdout-",
        suffix=".log",
        delete=False,
    )
    stderr_log = tempfile.NamedTemporaryFile(
        prefix="redteam-server-stderr-",
        suffix=".log",
        delete=False,
    )
    port_file = tempfile.NamedTemporaryFile(
        prefix="redteam-server-port-",
        suffix=".txt",
        delete=False,
    )
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    env_vars[SERVER_PORT_FILE_ENV] = port_file.name
    proc = subprocess.Popen(
        [_python_executable(), "server.py", "--port", str(requested_port)],
        cwd=str(app_dir),
        stdout=stdout_log,
        stderr=stderr_log,
        env=env_vars,
    )
    stdout_log.close()
    stderr_log.close()
    port_file.close()
    setattr(proc, "_redteam_server_stdout_log", stdout_log.name)
    setattr(proc, "_redteam_server_stderr_log", stderr_log.name)
    setattr(proc, "_redteam_server_port_file", port_file.name)

    import urllib.request

    deadline = time.monotonic() + timeout
    resolved_port: int | None = None
    while time.monotonic() < deadline:
        resolved_port = _read_server_port_file(port_file.name) or resolved_port
        probe_port = resolved_port or (requested_port if requested_port else None)
        try:
            if probe_port is None:
                time.sleep(0.1)
                continue
            req = urllib.request.Request(f"http://127.0.0.1:{probe_port}", method="GET")
            with urllib.request.urlopen(req, timeout=1):
                logger.info("Server ready on port %d (pid=%d)", probe_port, proc.pid)
                return proc, probe_port
        except Exception:
            if proc.poll() is not None:
                exit_code = proc.returncode
                message = _format_server_startup_failure(
                    proc,
                    f"Server process exited with code {exit_code} before startup completed",
                )
                stop_app_server(proc)
                raise RuntimeError(message)
            time.sleep(0.1)

    port_label = resolved_port or requested_port or "ephemeral"
    message = _format_server_startup_failure(
        proc,
        f"Server did not start within {timeout}s on port {port_label}",
    )
    stop_app_server(proc)
    raise RuntimeError(message)


def stop_app_server(proc: subprocess.Popen) -> None:
    """Terminate the server process, killing if necessary."""
    if proc.poll() is not None:
        try:
            proc.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
        _finalize_server_log_capture(proc)
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    finally:
        _finalize_server_log_capture(proc)


def reset_app_state(server_url: str) -> None:
    """POST /api/reset and wait briefly for SSE propagation."""
    import urllib.request

    req = urllib.request.Request(f"{server_url}/api/reset", method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as exc:
        logger.warning("Reset failed: %s", exc)
    time.sleep(0.5)


def run_sanity_check(
    app_dir: str | Path,
    variant: str,
    port: int | None = None,
    task_id: str | None = None,
) -> tuple[bool, str]:
    """Run the sanity check script for *variant* in *app_dir*."""
    app_dir = Path(app_dir)
    variant = suite_variant_name(variant)

    script = app_dir / f"sanity_check_{variant}.py"
    if not script.exists():
        script = app_dir / "sanity_check.py"
    if not script.exists():
        return False, f"Sanity check script not found: sanity_check_{variant}.py"

    argv: list[str] = []
    if port is not None:
        argv.extend(["--port", str(port)])
    if task_id:
        argv.extend(["--task-id", task_id])

    logger.info("Running sanity check via execution backend: %s", script)

    try:
        result = run_python_script(
            app_dir,
            script,
            timeout=300,
            argv=argv,
            block_network=True,
        )
    except subprocess.TimeoutExpired:
        return False, "Sanity check timed out after 300s"

    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr

    if (
        task_id
        and result.returncode != 0
        and "unrecognized arguments: --task-id" in output
    ):
        logger.warning("Sanity check script does not support --task-id; retrying suite-wide.")
        return run_sanity_check(app_dir, variant, port=port, task_id=None)

    passed = result.returncode == 0
    if passed:
        logger.info("Sanity check passed")
    else:
        logger.warning("Sanity check failed (rc=%d)", result.returncode)

    return passed, output


def load_functional_results(app_dir: str | Path) -> dict[str, Any]:
    """Load canonical functional results if they exist."""
    return _read_json(Path(app_dir) / "functional_results.json", default={}) or {}


def select_functional_backend_results(
    results: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    """Return the backend-scoped functional results view."""
    backend = normalize_functional_backend(backend)
    backends = results.get("backends") or {}
    scoped = backends.get(backend)
    return scoped if isinstance(scoped, dict) else {}


def select_declared_functional_backend_results(
    results: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    """Return backend results only when artifacts explicitly declare that backend."""
    backend = normalize_functional_backend(backend)
    backends = results.get("backends") or {}
    scoped = backends.get(backend)
    return scoped if isinstance(scoped, dict) else {}


def load_backend_functional_results(
    app_dir: str | Path,
    backend: str,
) -> dict[str, Any]:
    """Load the functional results for a specific backend."""
    return select_functional_backend_results(load_functional_results(app_dir), backend)


def load_declared_backend_functional_results(
    app_dir: str | Path,
    backend: str,
) -> dict[str, Any]:
    """Load functional results only when they explicitly belong to ``backend``."""
    return select_declared_functional_backend_results(load_functional_results(app_dir), backend)


def _merge_suite_results(
    existing_results: list[dict[str, Any]] | None,
    new_results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge per-task results, preferring ``new_results`` by ``task_id`` + backend."""
    merged: dict[str, dict[str, Any]] = {}
    for result in existing_results or []:
        key = f"{result['task_id']}::{result.get('backend', GENERIC_AGENT_BACKEND)}"
        merged[key] = result
    for result in new_results or []:
        key = f"{result['task_id']}::{result.get('backend', GENERIC_AGENT_BACKEND)}"
        merged[key] = result
    return [merged[key] for key in sorted(merged)]


def _summarize_suite_results(results: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not results:
        return {"total": 0, "passed": 0, "results": []}
    passed = sum(1 for result in results if result.get("passed"))
    results_dir = next(
        (
            result.get("results_dir")
            for result in results
            if result.get("results_dir")
        ),
        "",
    )
    return {
        "total": len(results),
        "passed": passed,
        "results": results,
        "results_dir": results_dir,
    }


def _ensure_backend_blocks(
    existing: dict[str, Any],
    *,
    backend: str,
) -> dict[str, dict[str, Any]]:
    backends = deepcopy(existing.get("backends") or {})
    return backends


def _write_backend_functional_results(
    app_dir: Path,
    *,
    backend: str,
    backends: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    backend_summary = deepcopy(backends[backend])
    backend_summary["backends"] = backends
    out_path = app_dir / "functional_results.json"
    out_path.write_text(
        json.dumps(backend_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %s (function: %d/%d = %.0f%%, real: %d/%d = %.0f%%)",
        out_path,
        (backend_summary.get("function_tasks") or {}).get("passed", 0),
        (backend_summary.get("function_tasks") or {}).get("total", 0),
        float(backend_summary.get("function_pass_rate", 0.0)) * 100,
        (backend_summary.get("real_tasks") or {}).get("passed", 0),
        (backend_summary.get("real_tasks") or {}).get("total", 0),
        float(backend_summary.get("real_pass_rate", 0.0)) * 100,
    )
    return backend_summary


def write_functional_results(
    app_dir: str | Path,
    function_results: list[dict[str, Any]] | None = None,
    real_results: list[dict[str, Any]] | None = None,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> dict[str, Any]:
    """Compute aggregate pass rates and write ``functional_results.json``."""
    app_dir = Path(app_dir)
    backend = normalize_functional_backend(backend)
    existing = load_functional_results(app_dir)

    backends = _ensure_backend_blocks(existing, backend=backend)
    backend_existing = backends.get(backend) or {}

    if function_results is None:
        function_results = (backend_existing.get("function_tasks") or {}).get("results")
    if real_results is None:
        real_results = (backend_existing.get("real_tasks") or {}).get("results")

    func_summary = _summarize_suite_results(function_results)
    real_summary = _summarize_suite_results(real_results)
    func_rate = (
        func_summary["passed"] / func_summary["total"]
        if func_summary["total"] > 0
        else 0.0
    )
    real_rate = (
        real_summary["passed"] / real_summary["total"]
        if real_summary["total"] > 0
        else 0.0
    )

    summary = {
        "backend": backend,
        "function_pass_rate": func_rate,
        "real_pass_rate": real_rate,
        "function_tasks": func_summary,
        "real_tasks": real_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    existing_baseline = backend_existing.get("readiness_baseline")
    if isinstance(existing_baseline, dict):
        summary["readiness_baseline"] = deepcopy(existing_baseline)
    backends[backend] = deepcopy(summary)
    return _write_backend_functional_results(
        app_dir,
        backend=backend,
        backends=backends,
    )


def persist_readiness_baseline(
    app_dir: str | Path,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> dict[str, Any]:
    """Persist an immutable readiness baseline for the selected backend."""
    app_dir = Path(app_dir)
    backend = normalize_functional_backend(backend)
    existing = load_functional_results(app_dir)
    if not existing:
        raise ValueError(f"No functional_results.json in {app_dir}")

    backends = _ensure_backend_blocks(existing, backend=backend)
    backend_summary = backends.get(backend)
    if not backend_summary:
        raise ValueError(f"functional_results.json missing backend block: {backend}")

    existing_baseline = backend_summary.get("readiness_baseline")
    if isinstance(existing_baseline, dict):
        return existing_baseline

    function_tasks = deepcopy(backend_summary.get("function_tasks") or {})
    real_tasks = deepcopy(backend_summary.get("real_tasks") or {})
    if not isinstance(function_tasks.get("results"), list):
        raise ValueError("functional_results.json missing readable function task results")
    if not isinstance(real_tasks.get("results"), list):
        raise ValueError("functional_results.json missing readable real task results")

    baseline = {
        "backend": backend,
        "function_pass_rate": backend_summary.get("function_pass_rate"),
        "real_pass_rate": backend_summary.get("real_pass_rate"),
        "function_tasks": function_tasks,
        "real_tasks": real_tasks,
        "timestamp": backend_summary.get("timestamp") or datetime.now(timezone.utc).isoformat(),
    }
    backends[backend] = deepcopy(backend_summary)
    backends[backend]["readiness_baseline"] = deepcopy(baseline)
    _write_backend_functional_results(
        app_dir,
        backend=backend,
        backends=backends,
    )
    return baseline


def check_quality_gate(
    app_dir: str | Path,
    threshold: float = 0.80,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> bool:
    """Check if app passes the functional correctness quality gate."""
    results = load_functional_results(app_dir)
    if not results:
        logger.warning("No functional_results.json in %s", app_dir)
        return False

    backend = normalize_functional_backend(backend)
    backend_results = load_backend_functional_results(app_dir, backend)
    if backend_results:
        pass_rate = backend_results.get("function_pass_rate", 0)
    else:
        stored_backend = results.get("backend")
        if stored_backend is not None and normalize_functional_backend(stored_backend) != backend:
            logger.warning(
                "functional_results.json backend %s does not match requested backend %s",
                stored_backend,
                backend,
            )
            return False
        pass_rate = results.get("function_pass_rate", 0)
    meets = pass_rate >= threshold
    logger.info(
        "Quality gate (%s): %.1f%% %s threshold %.0f%%",
        backend,
        pass_rate * 100,
        ">=" if meets else "<",
        threshold * 100,
    )
    return meets


def _load_pages_from_manifest(manifest: dict[str, Any]) -> list[Page]:
    pages = []
    for raw in manifest.get("pages", []) or []:
        if isinstance(raw, Page):
            pages.append(raw)
        elif isinstance(raw, dict):
            pages.append(Page(**raw))
    return pages


def _default_domain_bindings_from_pages(pages: list[Page]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in pages:
        parsed = urlparse(str(page.base_site_url or ""))
        if not parsed.netloc or parsed.netloc in seen:
            continue
        seen.add(parsed.netloc)
        bindings.append({"domain": parsed.netloc, "mode": "primary_spa"})
    return bindings


def build_functional_env_args(
    app_dir: str | Path,
    task: dict[str, Any],
    task_suite: str,
    repetition: int,
    manifest: dict[str, Any] | None = None,
) -> RedteamEnvArgs:
    """Build app-mode ``RedteamEnvArgs`` for a generated task."""
    manifest = manifest or load_app_manifest(app_dir)
    resolved_app_id = str((manifest.get("app_id") if isinstance(manifest, dict) else "") or Path(app_dir).name)
    require_supported_app_manifest(
        manifest,
        context=f"Functional evaluation for {resolved_app_id}",
    )
    pages = _load_pages_from_manifest(manifest)
    app_dir = Path(app_dir)
    suite_name = normalize_task_suite(task_suite)
    suite_slug = suite_name.replace("-tasks", "")
    task_id = task["id"]
    instruction = task["instruction"]
    behavior_id = resolved_app_id

    return RedteamEnvArgs(
        task_name=f"redteam.functional.{behavior_id}.{suite_slug}.{task_id}.rep{repetition}",
        app_id=resolved_app_id,
        target=instruction,
        attacker_model="",
        pages=pages,
        domain_bindings=_default_domain_bindings_from_pages(pages),
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="benign",
        start_page=manifest.get("start_page") or None,
        summary=task.get("difficulty", suite_slug),
        objective=instruction,
        doc=instruction,
        safe_behavior=instruction,
        max_conversation_turns=1,
        target_system_prompt=None,
        tools_config=None,
        criteria=None,
        judging=None,
        headless=True,
        max_steps=int(manifest.get("max_steps", 30) or 30),
        task_seed=repetition - 1,
        success_condition=task.get("success_condition", instruction),
        variation_seed=0,
        attack_channels=["axtree"],
        behavior_id=behavior_id,
        app_dir=str(app_dir),
        variant="benign",
    )


def build_functional_exp_args(
    app_dir: str | Path,
    tasks: list[dict[str, Any]],
    task_suite: str,
    agent_config: str,
    repetitions: int = 1,
) -> tuple[Any, list[ExpArgs]]:
    """Build AgentLab ``ExpArgs`` objects for a generated task suite."""
    agent_args = deepcopy(import_object(agent_config))
    benchmark = _FunctionalBenchmarkConfig(
        name=f"redteam-{suite_variant_name(task_suite)}-functional",
    )
    agent_args.set_benchmark(benchmark, demo_mode=False)

    manifest = load_app_manifest(app_dir)
    exp_args_list: list[ExpArgs] = []
    for task in tasks:
        for repetition in range(1, repetitions + 1):
            env_args = build_functional_env_args(
                app_dir=app_dir,
                task=task,
                task_suite=task_suite,
                repetition=repetition,
                manifest=manifest,
            )
            exp_args_list.append(
                ExpArgs(
                    agent_args=agent_args,
                    env_args=env_args,
                    logging_level=logging.INFO,
                    logging_level_stdout=logging.WARNING,
                )
            )
    for index, exp_args in enumerate(exp_args_list):
        exp_args.order = index
    return agent_args, exp_args_list


def _suite_results_root(
    app_dir: str | Path,
    task_suite: str,
    output_dir: str | Path | None = None,
) -> Path:
    app_dir = Path(app_dir)
    if output_dir is not None:
        root = Path(output_dir)
    else:
        suite = suite_variant_name(task_suite)
        root = app_dir / "results" / f"{_utc_timestamp()}_{suite}_functional"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_attempt_result(
    app_dir: str | Path,
    task: dict[str, Any],
    exp_dir: str | Path,
    repetition: int,
    results_dir: str | Path,
) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    summary_info = _read_json(exp_dir / "summary_info.json", default={}) or {}
    final_state_path = summary_info.get("final_state_path", str(exp_dir / "final_state.json"))
    server_events_path = summary_info.get(
        "server_events_path",
        str(exp_dir / "server_events.log"),
    )

    if Path(final_state_path).exists():
        passed, message = run_verifier_on_snapshot(app_dir, task["verify"], final_state_path)
    else:
        passed = False
        message = "Final state snapshot missing."

    return {
        "repetition": repetition,
        "passed": passed,
        "message": message,
        "exp_dir": str(exp_dir),
        "final_state_path": final_state_path,
        "server_events_path": server_events_path,
        "err_msg": summary_info.get("err_msg"),
        "terminated": summary_info.get("terminated"),
        "truncated": summary_info.get("truncated"),
        "results_dir": str(results_dir),
    }


def _aggregate_task_attempts(
    task: dict[str, Any],
    task_suite: str,
    attempts: list[dict[str, Any]],
    *,
    backend: str = GENERIC_AGENT_BACKEND,
) -> dict[str, Any]:
    passing_attempt = next((attempt for attempt in attempts if attempt["passed"]), None)
    chosen_attempt = passing_attempt or (attempts[-1] if attempts else {})
    return {
        "task_id": task["id"],
        "backend": backend,
        "suite": normalize_task_suite(task_suite),
        "instruction": task["instruction"],
        "difficulty": task.get("difficulty"),
        "verify": task["verify"],
        "passed": bool(passing_attempt),
        "message": chosen_attempt.get("message", ""),
        "exp_dir": chosen_attempt.get("exp_dir", ""),
        "final_state_path": chosen_attempt.get("final_state_path", ""),
        "server_events_path": chosen_attempt.get("server_events_path", ""),
        "results_dir": chosen_attempt.get("results_dir", ""),
        "attempts": attempts,
        "audit": chosen_attempt.get("audit", {}),
    }


def _run_generic_agent_functional_eval(
    *,
    app_dir: Path,
    tasks: list[dict[str, Any]],
    task_suite: str,
    agent_config: str,
    workers: int,
    repetitions: int,
    output_dir: str | Path | None,
) -> tuple[list[dict[str, Any]], Path]:
    results_dir = _suite_results_root(app_dir, task_suite, output_dir=output_dir)
    _, exp_args_list = build_functional_exp_args(
        app_dir=app_dir,
        tasks=tasks,
        task_suite=task_suite,
        agent_config=agent_config,
        repetitions=repetitions,
    )
    backend = "sequential" if workers <= 1 else "joblib"
    run_experiments(
        n_jobs=max(1, workers),
        exp_args_list=exp_args_list,
        study_dir=results_dir,
        parallel_backend=backend,
    )

    attempts_by_task: dict[str, list[dict[str, Any]]] = {task["id"]: [] for task in tasks}
    for index, exp_args in enumerate(exp_args_list):
        task = tasks[index // repetitions]
        repetition = (index % repetitions) + 1
        attempts_by_task[task["id"]].append(
            _load_attempt_result(
                app_dir=app_dir,
                task=task,
                exp_dir=exp_args.exp_dir,
                repetition=repetition,
                results_dir=results_dir,
            )
        )

    aggregated_results = [
        _aggregate_task_attempts(
            task,
            task_suite,
            attempts_by_task[task["id"]],
            backend=GENERIC_AGENT_BACKEND,
        )
        for task in tasks
    ]
    return aggregated_results, results_dir


def _run_agent_browser_functional_eval(
    *,
    app_dir: Path,
    runtime_app_dir: Path | None,
    tasks: list[dict[str, Any]],
    task_suite: str,
    agent_config: str,
    repetitions: int,
    output_dir: str | Path | None,
    max_steps: int,
) -> tuple[list[dict[str, Any]], Path]:
    if repetitions != 1:
        raise ValueError("agent-browser backend currently supports repetitions=1 only")

    results_dir = _suite_results_root(app_dir, task_suite, output_dir=output_dir)
    shared_server_events_path = (results_dir / "server_events.log").resolve()
    shared_server_events_path.write_text("", encoding="utf-8")
    manifest = load_app_manifest(app_dir)
    start_url = localize_start_url(
        "http://127.0.0.1",
        manifest.get("start_page"),
    )

    server_env = {
        "SERVER_EVENTS_LOG": str(shared_server_events_path),
    }
    proc, port = start_app_server(runtime_app_dir or app_dir, env=server_env)
    try:
        server_url = f"http://127.0.0.1:{port}"
        task_start_url = localize_start_url(server_url, manifest.get("start_page"))
        aggregated_results: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            exp_dir = results_dir / f"task_{index}_{task['id']}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            shared_log_offset = shared_server_events_path.stat().st_size if shared_server_events_path.exists() else 0
            reset_app_state(server_url)
            runner_result = run_agent_browser_task(
                server_url=server_url,
                start_url=task_start_url or start_url,
                instruction=task["instruction"],
                task_dir=exp_dir,
                agent_config=agent_config,
                max_steps=max_steps,
                timeout=DEFAULT_AGENT_BROWSER_TIMEOUT,
            )
            final_state = fetch_normalized_state(port, timeout=5) or {}
            final_state_path = exp_dir / "final_state.json"
            final_state_path.write_text(
                json.dumps(final_state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            task_server_events_path = exp_dir / "server_events.log"
            if shared_server_events_path.exists():
                with open(shared_server_events_path, "rb") as handle:
                    handle.seek(shared_log_offset)
                    task_server_events_path.write_bytes(handle.read())
            else:
                task_server_events_path.write_text("", encoding="utf-8")
            passed, verifier_message = run_verifier_on_snapshot(
                app_dir,
                task["verify"],
                final_state_path,
            )
            summary_info = {
                "final_state_path": str(final_state_path),
                "server_events_path": str(task_server_events_path),
                "terminated": passed,
                "truncated": not passed,
                "err_msg": runner_result.error,
                "backend": AGENT_BROWSER_BACKEND,
                "transcript_path": runner_result.transcript_path,
            }
            (exp_dir / "summary_info.json").write_text(
                json.dumps(summary_info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            attempts = [
                {
                    "repetition": 1,
                    "passed": passed,
                    "message": verifier_message,
                    "exp_dir": str(exp_dir),
                    "final_state_path": str(final_state_path),
                    "server_events_path": str(task_server_events_path),
                    "err_msg": runner_result.error,
                    "terminated": passed,
                    "truncated": not passed,
                    "results_dir": str(results_dir),
                }
            ]
            aggregated_results.append(
                _aggregate_task_attempts(
                    task,
                    task_suite,
                    attempts,
                    backend=AGENT_BROWSER_BACKEND,
                )
            )
        return aggregated_results, results_dir
    finally:
        stop_app_server(proc)


def run_functional_eval(
    app_dir: str | Path,
    task_suite: str,
    agent_config: str = DEFAULT_AGENT_CONFIG,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
    workers: int = 1,
    repetitions: int = 1,
    task_id: str | None = None,
    difficulty: str | None = None,
    output_dir: str | Path | None = None,
    runtime_app_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run browser-agent evaluation on a generated task suite."""
    app_dir = Path(app_dir)
    task_suite = normalize_task_suite(task_suite)
    backend = normalize_functional_backend(backend)
    tasks = filter_tasks(load_tasks(app_dir, task_suite), task_id=task_id, difficulty=difficulty)
    if not tasks:
        raise ValueError(f"No tasks selected for suite {task_suite} in {app_dir}")

    if backend == GENERIC_AGENT_BACKEND:
        aggregated_results, results_dir = _run_generic_agent_functional_eval(
            app_dir=app_dir,
            tasks=tasks,
            task_suite=task_suite,
            agent_config=agent_config,
            workers=workers,
            repetitions=repetitions,
            output_dir=output_dir,
        )
    else:
        aggregated_results, results_dir = _run_agent_browser_functional_eval(
            app_dir=app_dir,
            runtime_app_dir=Path(runtime_app_dir) if runtime_app_dir is not None else None,
            tasks=tasks,
            task_suite=task_suite,
            agent_config=agent_config,
            repetitions=repetitions,
            output_dir=output_dir,
            max_steps=int((load_app_manifest(app_dir).get("max_steps", 30) or 30)),
        )

    existing = load_functional_results(app_dir)
    existing_backend = select_declared_functional_backend_results(existing, backend)
    if task_suite == "function-tasks":
        merged_function_results = _merge_suite_results(
            (existing_backend.get("function_tasks") or {}).get("results"),
            aggregated_results,
        )
        merged_real_results = (existing_backend.get("real_tasks") or {}).get("results")
        summary = write_functional_results(
            app_dir,
            function_results=merged_function_results,
            real_results=merged_real_results,
            backend=backend,
        )
    else:
        merged_function_results = (existing_backend.get("function_tasks") or {}).get("results")
        merged_real_results = _merge_suite_results(
            (existing_backend.get("real_tasks") or {}).get("results"),
            aggregated_results,
        )
        summary = write_functional_results(
            app_dir,
            function_results=merged_function_results,
            real_results=merged_real_results,
            backend=backend,
        )

    invocation_summary = _summarize_suite_results(aggregated_results)
    invocation_pass_rate = (
        invocation_summary["passed"] / invocation_summary["total"]
        if invocation_summary["total"] > 0
        else 0.0
    )
    suite_result = {
        "task_suite": task_suite,
        "backend": backend,
        "agent_config": agent_config,
        "results_dir": str(results_dir),
        "pass_rate": invocation_pass_rate,
        "total": invocation_summary["total"],
        "passed": invocation_summary["passed"],
        "results": aggregated_results,
        "timestamp": summary["timestamp"],
    }
    result_summary_path = results_dir / "result_summary.json"
    result_summary_path.write_text(
        json.dumps(suite_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return suite_result


def main() -> None:
    """CLI entry point for standalone use."""
    parser = argparse.ArgumentParser(description="Functional correctness evaluation harness")
    parser.add_argument(
        "--app-dir",
        required=True,
        help="Root directory of the generated app",
    )

    sub = parser.add_subparsers(dest="command")

    sc = sub.add_parser("sanity-check", help="Run sanity check")
    sc.add_argument("variant", choices=["function", "real"], help="Which sanity check to run")
    sc.add_argument("--port", type=int, default=None, help="Server port")
    sc.add_argument("--task-id", default=None, help="Single task to sanity-check")

    ev = sub.add_parser("evaluate", help="Run browser-agent functional evaluation")
    ev.add_argument("variant", choices=["function", "real"], help="Which task suite to evaluate")
    ev.add_argument(
        "--agent-config",
        default=DEFAULT_AGENT_CONFIG,
        help=f"Python path to the AgentArgs object (default: {DEFAULT_AGENT_CONFIG})",
    )
    ev.add_argument(
        "--backend",
        default=DEFAULT_FUNCTIONAL_BACKEND,
        choices=list(SUPPORTED_FUNCTIONAL_BACKENDS),
        help="Functional evaluation backend.",
    )
    ev.add_argument("--workers", type=int, default=1, help="Parallel workers")
    ev.add_argument("--repetitions", type=int, default=1, help="Task repetitions")
    ev.add_argument("--task-id", default=None, help="Comma-separated task IDs to evaluate")
    ev.add_argument("--difficulty", default=None, help="Optional difficulty filter")
    ev.add_argument("--output-dir", default=None, help="Override results directory")

    qg = sub.add_parser("quality-gate", help="Check quality gate")
    qg.add_argument(
        "--backend",
        default=DEFAULT_FUNCTIONAL_BACKEND,
        choices=list(SUPPORTED_FUNCTIONAL_BACKENDS),
        help="Functional evaluation backend.",
    )
    qg.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum pass rate (default: 0.80)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "sanity-check":
        passed, output = run_sanity_check(
            args.app_dir,
            args.variant,
            port=args.port,
            task_id=args.task_id,
        )
        print(output)
        sys.exit(0 if passed else 1)

    if args.command == "evaluate":
        result = run_functional_eval(
            app_dir=args.app_dir,
            task_suite=args.variant,
            agent_config=args.agent_config,
            backend=args.backend,
            workers=args.workers,
            repetitions=args.repetitions,
            task_id=args.task_id,
            difficulty=args.difficulty,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.command == "quality-gate":
        meets = check_quality_gate(
            args.app_dir,
            threshold=args.threshold,
            backend=args.backend,
        )
        sys.exit(0 if meets else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
