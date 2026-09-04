from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from warp_taskgen.cli import _impl as cli_impl
from warp_taskgen.phase_1.run_lock import Phase1AlreadyRunning, phase_1_run_lock
from warp_taskgen.run_transition import resolve_run_request


def test_phase_1_run_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    with phase_1_run_lock(tmp_path):
        with pytest.raises(
            Phase1AlreadyRunning,
            match=str(tmp_path / "phase_1" / ".phase1_run.lock"),
        ):
            with phase_1_run_lock(tmp_path):
                pass


def test_phase_1_run_lock_allows_different_state_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    with phase_1_run_lock(first_root):
        with phase_1_run_lock(second_root):
            pass


def test_phase_1_run_lock_releases_after_owner_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="owner failed"):
        with phase_1_run_lock(tmp_path):
            raise RuntimeError("owner failed")

    with phase_1_run_lock(tmp_path):
        pass


def test_phase_1_run_lock_ignores_inert_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "phase_1" / ".phase1_run.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("pid=stale", encoding="utf-8")

    with phase_1_run_lock(tmp_path):
        pass


@pytest.mark.parametrize("generate_novel", [False, True])
def test_phase_1_cli_refuses_same_root_before_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    generate_novel: bool,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        cli_impl,
        "_dispatch_phase_with_run_context",
        lambda _args: pytest.fail("Phase 1 body must not run while its lock is held"),
    )

    with phase_1_run_lock(tmp_path):
        result = cli_impl._dispatch_phase(
            cli_impl.argparse.Namespace(
                command="phase",
                phase="1",
                generate_novel=generate_novel,
            )
        )

    assert result == 2
    assert str(tmp_path / "phase_1" / ".phase1_run.lock") in capsys.readouterr().err


def test_phase_1_cli_allows_different_state_root_while_first_is_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    entered: list[Path] = []

    def run_phase(_args: object) -> int:
        entered.append(second_root)
        return 0

    monkeypatch.setattr(cli_impl, "_dispatch_phase_with_run_context", run_phase)
    with phase_1_run_lock(first_root):
        monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(second_root))
        result = cli_impl._dispatch_phase(cli_impl.argparse.Namespace(command="phase", phase="1"))

    assert result == 0
    assert entered == [second_root]


def _write_identified_phase_1_state(root: Path) -> bytes:
    inputs = {
        "benchmark_name": "webarena_verified",
        "benchmark_path": str(root / "benchmark"),
        "manifest_path": str(root / "phase_0a" / "BENCHMARK_MANIFEST.json"),
        "sandbox_model": "claude-sonnet-4-6",
        "generate_novel": False,
        "novel_tasks_per_site": 30,
    }
    transition = resolve_run_request(
        inputs,
        existing_state=None,
        new_run_id="run-lock-owner",
    )
    assert transition.definition is not None
    state = {
        **inputs,
        "step": "phase_1",
        "status": "running",
        "timestamp": "2026-09-03T00:00:00+00:00",
        "logs_dir": str(root),
        "run_definition": transition.definition.to_dict(),
    }
    state_path = root / "pipeline_state.json"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    return state_path.read_bytes()


def test_phase_1_cli_subprocess_lock_covers_identified_root_transition(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2]
    state_before = _write_identified_phase_1_state(tmp_path)
    ready_path = tmp_path / "owner-ready"
    owner_code = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from warp_taskgen.phase_1.run_lock import phase_1_run_lock",
            "root = Path(sys.argv[1])",
            "ready = Path(sys.argv[2])",
            "with phase_1_run_lock(root):",
            "    ready.write_text('ready', encoding='utf-8')",
            "    sys.stdin.read()",
        )
    )
    env = {
        **os.environ,
        "WARP_TASKGEN_STATE_DIR": str(tmp_path),
        "WORLDSIM_STATE_DIR": str(tmp_path),
    }
    owner = subprocess.Popen(
        [sys.executable, "-c", owner_code, str(tmp_path), str(ready_path)],
        cwd=package_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready_path.exists() and time.monotonic() < deadline:
            if owner.poll() is not None:
                break
            time.sleep(0.01)
        owner_error = owner.stderr.read() if owner.poll() is not None and owner.stderr else ""
        assert ready_path.exists(), f"owner did not acquire lock: {owner_error}"
        assert owner.poll() is None

        contender = subprocess.run(
            [
                sys.executable,
                "-m",
                "warp_taskgen.main",
                "phase",
                "1",
                "--generate-novel",
            ],
            cwd=package_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lock_path = tmp_path / "phase_1" / ".phase1_run.lock"
        assert contender.returncode == 2, contender.stderr
        assert str(lock_path) in contender.stderr
        assert "isolated Derived Run" not in contender.stderr
        assert not (tmp_path / "phase_1" / "benign_tasks.json").exists()
        assert (tmp_path / "pipeline_state.json").read_bytes() == state_before
    finally:
        if owner.poll() is None:
            owner.terminate()
        try:
            owner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owner.kill()
            owner.wait(timeout=5)

    with phase_1_run_lock(tmp_path):
        pass
