from __future__ import annotations

import base64
import io
import json
import os
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentlab.benchmarks.redteam import execution


def _bundle_payload(files: dict[str, str]) -> dict[str, object]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative_path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=relative_path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return {
        "files": sorted(files),
        "tar_gz_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def test_resolve_execution_policy_defaults_to_modal(monkeypatch) -> None:
    monkeypatch.delenv(execution.EXECUTION_BACKEND_ENV, raising=False)
    monkeypatch.delenv(execution.UNSAFE_LOCAL_EXECUTION_ENV, raising=False)

    policy = execution.resolve_execution_policy()

    assert policy.backend == "modal"
    assert policy.unsafe_local is False


def test_resolve_execution_policy_requires_explicit_local_opt_in(monkeypatch) -> None:
    monkeypatch.setenv(execution.EXECUTION_BACKEND_ENV, "local")
    monkeypatch.delenv(execution.UNSAFE_LOCAL_EXECUTION_ENV, raising=False)

    with pytest.raises(RuntimeError, match="unsafe host execution"):
        execution.resolve_execution_policy()


def test_run_python_script_uses_modal_runner_remote_path(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    script = app_dir / "sanity_check_function.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_command(self, *, argv, timeout, sync_allowlist=(), env=None, block_network=True):
        captured["argv"] = list(argv)
        captured["timeout"] = timeout
        captured["block_network"] = block_network
        captured["profile"] = self.profile.name
        captured["image_family"] = self.profile.image_family
        return execution.SandboxCommandResult(0, "ok\n", "", "modal")

    monkeypatch.setattr(execution.ModalSandboxRunner, "run_command", fake_run_command)

    result = execution.run_python_script(
        app_dir,
        script,
        timeout=30,
    )

    assert result.returncode == 0
    assert captured["argv"] == ["python", "/workspace/app/sanity_check_function.py"]
    assert captured["block_network"] is True
    assert captured["profile"] == execution.FRESH_PYTHON_WORKER_PROFILE.name
    assert captured["image_family"] == "python-worker-image"


def test_stage_workspace_ingress_copies_workspace_separately_from_base_image(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "js").mkdir()
    (app_dir / "js" / "data.js").write_text("const ok = true;\n", encoding="utf-8")
    runner = execution.ModalSandboxRunner(app_dir)

    with execution.tempfile.TemporaryDirectory(prefix="redteam-test-stage-") as tmpdir:
        ingress = runner._stage_workspace_ingress(Path(tmpdir))

        assert ingress.remote_path == execution.SANDBOX_APP_ROOT
        assert ingress.staged_dir != app_dir
        assert (ingress.staged_dir / "js" / "data.js").read_text(encoding="utf-8") == "const ok = true;\n"


def test_stage_workspace_ingress_rejects_escape_symlink(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (app_dir / "escape.txt").symlink_to(outside)
    runner = execution.ModalSandboxRunner(app_dir)

    with execution.tempfile.TemporaryDirectory(prefix="redteam-test-stage-") as tmpdir:
        with pytest.raises(RuntimeError, match="symlinked workspace path"):
            runner._stage_workspace_ingress(Path(tmpdir))


def test_stage_workspace_ingress_rejects_in_root_symlink(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    target = app_dir / "data.txt"
    target.write_text("ok\n", encoding="utf-8")
    (app_dir / "linked.txt").symlink_to(target)
    runner = execution.ModalSandboxRunner(app_dir)

    with execution.tempfile.TemporaryDirectory(prefix="redteam-test-stage-") as tmpdir:
        with pytest.raises(RuntimeError, match="symlinked workspace path"):
            runner._stage_workspace_ingress(Path(tmpdir))


def test_build_base_image_selects_family_without_staged_workspace_content() -> None:
    class _FakeImage:
        def __init__(self):
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def add_local_python_source(self, *args, **kwargs):
            self.calls.append(("add_local_python_source", args))
            return self

        def apt_install(self, *args):
            self.calls.append(("apt_install", args))
            return self

        def pip_install(self, *args):
            self.calls.append(("pip_install", args))
            return self

        def run_commands(self, *args):
            self.calls.append(("run_commands", args))
            return self

    fake_images: list[_FakeImage] = []

    class _FakeImageFactory:
        @staticmethod
        def debian_slim(**kwargs):
            image = _FakeImage()
            image.calls.append(("debian_slim", tuple(sorted(kwargs.items()))))
            fake_images.append(image)
            return image

    class _FakeModal:
        Image = _FakeImageFactory

    authoring = execution.ModalAuthoringSession(
        Path("/tmp/authoring"),
        owner_scope="scope",
        workspace_identity="/tmp/authoring",
        workspace_fingerprint="fingerprint",
        behavior_id="behavior",
    )
    python_worker = execution.ModalSandboxRunner(
        Path("/tmp/python-worker"),
        profile=execution.FRESH_PYTHON_WORKER_PROFILE,
    )
    browser_worker = execution.ModalSandboxRunner(
        Path("/tmp/browser-worker"),
        profile=execution.FRESH_BROWSER_WORKER_PROFILE,
    )

    authoring_image = authoring._build_base_image(_FakeModal)
    python_image = python_worker._build_base_image(_FakeModal)
    browser_image = browser_worker._build_base_image(_FakeModal)

    assert authoring_image is fake_images[0]
    assert python_image is fake_images[1]
    assert browser_image is fake_images[2]
    assert ("apt_install", ("nodejs", "npm")) in fake_images[0].calls
    assert any(call[0] == "run_commands" for call in fake_images[0].calls)
    assert all(call[0] != "apt_install" for call in fake_images[1].calls)
    assert all(call[0] != "pip_install" for call in fake_images[1].calls)
    assert all(call[0] != "run_commands" for call in fake_images[1].calls)
    assert all(call[0] != "apt_install" for call in fake_images[2].calls)
    assert ("pip_install", ("playwright==1.44.0",)) in fake_images[2].calls
    assert ("run_commands", ("playwright install --with-deps chromium",)) in fake_images[2].calls


def test_run_trusted_worker_defaults_to_python_worker_profile(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    captured: dict[str, object] = {}

    def fake_run_command(self, *, argv, timeout, sync_allowlist=(), env=None, block_network=True):
        captured["argv"] = list(argv)
        captured["profile"] = self.profile.name
        captured["image_family"] = self.profile.image_family
        captured["block_network"] = block_network
        return execution.SandboxCommandResult(0, "{}", "", "modal")

    monkeypatch.setattr(execution.ModalSandboxRunner, "run_command", fake_run_command)

    result = execution.run_trusted_worker(
        app_dir,
        "verify-snapshot",
        timeout=30,
        argv=["--root", "."],
    )

    assert result.returncode == 0
    assert captured["argv"] == [
        "python",
        "-m",
        execution.SANDBOX_WORKER_MODULE,
        "verify-snapshot",
        "--root",
        ".",
    ]
    assert captured["profile"] == execution.FRESH_PYTHON_WORKER_PROFILE.name
    assert captured["image_family"] == "python-worker-image"
    assert captured["block_network"] is True


def test_run_trusted_worker_allows_browser_worker_profile_selection(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    captured: dict[str, object] = {}

    def fake_run_command(self, *, argv, timeout, sync_allowlist=(), env=None, block_network=True):
        captured["profile"] = self.profile.name
        captured["image_family"] = self.profile.image_family
        return execution.SandboxCommandResult(0, "{}", "", "modal")

    monkeypatch.setattr(execution.ModalSandboxRunner, "run_command", fake_run_command)

    result = execution.run_trusted_worker(
        app_dir,
        "validate-app-runtime",
        timeout=30,
        argv=["--root", "."],
        profile=execution.FRESH_BROWSER_WORKER_PROFILE,
    )

    assert result.returncode == 0
    assert captured["profile"] == execution.FRESH_BROWSER_WORKER_PROFILE.name
    assert captured["image_family"] == "browser-worker-image"


def test_execute_authoring_command_reuses_modal_session_and_enables_pty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    trusted_server = app_dir / "server.py"
    trusted_server.write_text("print('trusted')\n", encoding="utf-8")
    create_calls: list[dict[str, object]] = []
    exec_calls: list[dict[str, object]] = []
    terminated: list[str] = []

    class _FakeSandbox:
        object_id = "sb-authoring"

        def terminate(self):
            terminated.append("terminated")
            return None

    def fake_create_sandbox(self, workspace_ingress, *, timeout, block_network, idle_timeout=None):
        create_calls.append(
            {
                "timeout": timeout,
                "block_network": block_network,
                "idle_timeout": idle_timeout,
                "remote_path": str(workspace_ingress.remote_path),
                "staged_server_exists": (workspace_ingress.staged_dir / "server.py").exists(),
            }
        )
        return _FakeSandbox()

    def fake_execute_in_sandbox(sandbox, *, argv, timeout, env=None, pty=False):
        exec_calls.append(
            {
                "argv": list(argv),
                "timeout": timeout,
                "pty": pty,
            }
        )
        return {"returncode": 0, "stdout": "ok\n", "stderr": ""}

    def fake_worker_json(self, sandbox, subcommand, argv):
        assert subcommand == "sha256-file"
        return {"sha256": execution._sha256_path(trusted_server)}

    monkeypatch.setattr(execution.ModalAuthoringSession, "_create_sandbox", fake_create_sandbox)
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_execute_in_sandbox",
        staticmethod(fake_execute_in_sandbox),
    )
    monkeypatch.setattr(execution.ModalAuthoringSession, "_worker_json", fake_worker_json)

    first = execution.execute_authoring_command(
        app_dir,
        ["claude", "--print", "first"],
        timeout=30,
    )
    second = execution.execute_authoring_command(
        app_dir,
        ["claude", "--print", "second"],
        timeout=45,
    )

    execution.close_authoring_session(app_dir)

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.metadata["profile"] == execution.TRUSTED_AUTHORING_PROFILE.name
    assert second.metadata["profile"] == execution.TRUSTED_AUTHORING_PROFILE.name
    assert first.metadata["image_family"] == "authoring-image"
    assert first.metadata["image_revision"] == execution.AUTHORING_IMAGE_REVISION
    assert first.metadata["resource_key"] == second.metadata["resource_key"]
    assert first.metadata["owner_scope"] == second.metadata["owner_scope"]
    assert len(create_calls) == 1
    assert all(call["pty"] is True for call in exec_calls)
    assert create_calls[0]["block_network"] is False
    assert create_calls[0]["remote_path"] == str(execution.SANDBOX_APP_ROOT)
    assert create_calls[0]["staged_server_exists"] is True
    assert terminated == ["terminated"]


def test_execute_authoring_command_warm_start_uses_named_sandbox_when_opted_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    trusted_server = app_dir / "server.py"
    trusted_server.write_text("print('trusted')\n", encoding="utf-8")
    attached_calls: list[str] = []

    class _FakeSandbox:
        object_id = "sb-authoring-warm"

        def terminate(self):
            return None

    def fake_attach(self):
        attached_calls.append(self._warm_start_name)
        return _FakeSandbox()

    def fake_execute_in_sandbox(sandbox, *, argv, timeout, env=None, pty=False):
        return {"returncode": 0, "stdout": "ok\n", "stderr": ""}

    def fake_worker_json(self, sandbox, subcommand, argv):
        assert subcommand == "sha256-file"
        return {"sha256": execution._sha256_path(trusted_server)}

    monkeypatch.setenv(execution.TRUSTED_AUTHORING_WARM_START_ENV, "1")
    monkeypatch.setenv(execution.TRUSTED_AUTHORING_MODAL_APP_ENV, "deployed-authoring")
    monkeypatch.setattr(execution.ModalAuthoringSession, "_attach_named_sandbox", fake_attach)
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_execute_in_sandbox",
        staticmethod(fake_execute_in_sandbox),
    )
    monkeypatch.setattr(execution.ModalAuthoringSession, "_worker_json", fake_worker_json)

    result = execution.execute_authoring_command(
        app_dir,
        ["python", "-c", "print('ok')"],
        timeout=30,
    )

    assert result.returncode == 0
    assert attached_calls
    assert result.metadata["warm_start_enabled"] is True
    assert result.metadata["warm_start_attempted"] is True
    assert result.metadata["warm_start_hit"] is True
    assert result.metadata["warm_start_mode"] == "named-sandbox"
    assert result.metadata["warm_start_name"].startswith("redteam-authoring-")
    assert result.metadata["warm_start_lookup_app"] == "deployed-authoring"
    execution.close_all_authoring_sessions()


def test_execute_authoring_command_warm_start_lookup_miss_falls_back_to_named_cold_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    trusted_server = app_dir / "server.py"
    trusted_server.write_text("print('trusted')\n", encoding="utf-8")
    created: list[tuple[str, str]] = []

    class _FakeSandbox:
        object_id = "sb-authoring-created"

        def terminate(self):
            return None

    def fake_create_named(self, workspace_ingress, *, timeout, idle_timeout, modal_app_name, sandbox_name):
        created.append((modal_app_name, sandbox_name))
        return _FakeSandbox()

    def fake_execute_in_sandbox(sandbox, *, argv, timeout, env=None, pty=False):
        return {"returncode": 0, "stdout": "ok\n", "stderr": ""}

    def fake_worker_json(self, sandbox, subcommand, argv):
        assert subcommand == "sha256-file"
        return {"sha256": execution._sha256_path(trusted_server)}

    monkeypatch.setenv(execution.TRUSTED_AUTHORING_WARM_START_ENV, "1")
    monkeypatch.setenv(execution.TRUSTED_AUTHORING_MODAL_APP_ENV, "deployed-authoring")
    monkeypatch.setattr(execution.ModalAuthoringSession, "_attach_named_sandbox", lambda self: None)
    monkeypatch.setattr(execution.ModalAuthoringSession, "_create_named_sandbox", fake_create_named)
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_execute_in_sandbox",
        staticmethod(fake_execute_in_sandbox),
    )
    monkeypatch.setattr(execution.ModalAuthoringSession, "_worker_json", fake_worker_json)

    result = execution.execute_authoring_command(
        app_dir,
        ["python", "-c", "print('ok')"],
        timeout=30,
    )

    assert result.returncode == 0
    assert created == [("deployed-authoring", result.metadata["warm_start_name"])]
    assert result.metadata["warm_start_enabled"] is True
    assert result.metadata["warm_start_attempted"] is True
    assert result.metadata["warm_start_hit"] is False
    assert result.metadata["warm_start_mode"] == "cold-start"
    execution.close_all_authoring_sessions()


def test_execute_authoring_command_warm_start_missing_lookup_app_falls_back_to_current_trusted_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    trusted_server = app_dir / "server.py"
    trusted_server.write_text("print('trusted')\n", encoding="utf-8")

    class _FakeSandbox:
        object_id = "sb-authoring-cold"

        def terminate(self):
            return None

    monkeypatch.setenv(execution.TRUSTED_AUTHORING_WARM_START_ENV, "1")
    monkeypatch.delenv(execution.TRUSTED_AUTHORING_MODAL_APP_ENV, raising=False)
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_create_sandbox",
        lambda self, workspace_ingress, *, timeout, block_network, idle_timeout=None: _FakeSandbox(),
    )
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, *, argv, timeout, env=None, pty=False: {"returncode": 0, "stdout": "ok\n", "stderr": ""}),
    )
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_worker_json",
        lambda self, sandbox, subcommand, argv: {"sha256": execution._sha256_path(trusted_server)},
    )

    result = execution.execute_authoring_command(
        app_dir,
        ["python", "-c", "print('ok')"],
        timeout=30,
    )

    assert result.returncode == 0
    assert result.metadata["warm_start_enabled"] is True
    assert result.metadata["warm_start_attempted"] is False
    assert result.metadata["warm_start_hit"] is False
    assert result.metadata["warm_start_mode"] == "cold-start"
    assert "warm_start_lookup_app" not in result.metadata
    execution.close_all_authoring_sessions()


def test_modal_runner_syncs_allowlisted_files_and_removes_deleted_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    working_dir = tmp_path / "app"
    (working_dir / "js").mkdir(parents=True)
    (working_dir / "js" / "stale.js").write_text("stale\n", encoding="utf-8")

    class _FakeSandbox:
        def terminate(self):
            return None

    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_create_sandbox",
        lambda self, workspace_ingress, timeout, block_network: _FakeSandbox(),
    )
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, argv, timeout, env=None: {"returncode": 0, "stdout": "ok\n", "stderr": ""}),
    )
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_worker_json",
        lambda self, sandbox, subcommand, argv: _bundle_payload({"js/data.js": "fresh\n"}),
    )

    runner = execution.ModalSandboxRunner(working_dir)
    result = runner.run_command(
        argv=["python", "-V"],
        timeout=30,
        sync_allowlist=["js/**"],
        block_network=True,
    )

    assert result.returncode == 0
    assert (working_dir / "js" / "data.js").read_text(encoding="utf-8") == "fresh\n"
    assert (working_dir / "js" / "stale.js").exists() is False
    assert result.metadata["egress"]["applied"] is True


def test_modal_runner_failure_does_not_sync_or_delete_allowlisted_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    working_dir = tmp_path / "app"
    (working_dir / "js").mkdir(parents=True)
    (working_dir / "js" / "existing.js").write_text("known-good\n", encoding="utf-8")

    class _FakeSandbox:
        def terminate(self):
            return None

    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_create_sandbox",
        lambda self, workspace_ingress, timeout, block_network: _FakeSandbox(),
    )
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, argv, timeout, env=None: {"returncode": 1, "stdout": "", "stderr": "boom\n"}),
    )
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_worker_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("allowlist bundle should not be fetched")),
    )

    runner = execution.ModalSandboxRunner(working_dir)
    result = runner.run_command(
        argv=["python", "-V"],
        timeout=30,
        sync_allowlist=["js/**"],
        block_network=True,
    )

    assert result.returncode == 1
    assert (working_dir / "js" / "existing.js").read_text(encoding="utf-8") == "known-good\n"
    assert result.metadata["egress"]["applied"] is False
    assert result.metadata["removed_files"] == []
    assert result.metadata["returned_files"] == []


def test_modal_runner_passes_block_network_flag(tmp_path: Path, monkeypatch) -> None:
    working_dir = tmp_path / "app"
    working_dir.mkdir()
    captured: dict[str, object] = {}

    class _FakeSandbox:
        def terminate(self):
            return None

    def fake_create_sandbox(self, workspace_ingress, *, timeout, block_network):
        captured["timeout"] = timeout
        captured["block_network"] = block_network
        return _FakeSandbox()

    monkeypatch.setattr(execution.ModalSandboxRunner, "_create_sandbox", fake_create_sandbox)
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, argv, timeout, env=None: {"returncode": 0, "stdout": "", "stderr": ""}),
    )

    runner = execution.ModalSandboxRunner(working_dir)
    runner.run_command(
        argv=["python", "-V"],
        timeout=45,
        sync_allowlist=[],
        block_network=True,
    )

    assert captured["timeout"] == 45
    assert captured["block_network"] is True


def test_browser_worker_profile_selects_browser_worker_image(tmp_path: Path, monkeypatch) -> None:
    working_dir = tmp_path / "app"
    working_dir.mkdir()

    class _FakeSandbox:
        object_id = "sb-browser"

        def terminate(self):
            return None

    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_create_sandbox",
        lambda self, workspace_ingress, timeout, block_network: _FakeSandbox(),
    )
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, argv, timeout, env=None: {"returncode": 0, "stdout": "", "stderr": ""}),
    )

    runner = execution.ModalSandboxRunner(
        working_dir,
        profile=execution.FRESH_BROWSER_WORKER_PROFILE,
    )
    result = runner.run_command(
        argv=["python", "-V"],
        timeout=30,
        block_network=True,
    )

    assert result.metadata["profile"] == execution.FRESH_BROWSER_WORKER_PROFILE.name
    assert result.metadata["image_family"] == "browser-worker-image"
    assert result.metadata["image_revision"] == execution.BROWSER_WORKER_IMAGE_REVISION
    assert result.metadata["secret_set_name"] is None


def test_execute_authoring_command_failure_does_not_sync_or_delete_allowlisted_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    (app_dir / "js").mkdir(parents=True)
    (app_dir / "js" / "existing.js").write_text("known-good\n", encoding="utf-8")
    trusted_server = app_dir / "server.py"
    trusted_server.write_text("print('trusted')\n", encoding="utf-8")
    terminated: list[str] = []

    class _FakeSandbox:
        object_id = "sb-authoring-failure"

        def terminate(self):
            terminated.append("terminated")
            return None

    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_create_sandbox",
        lambda self, workspace_ingress, timeout, block_network, idle_timeout=None: _FakeSandbox(),
    )
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, argv, timeout, env=None, pty=False: {"returncode": 1, "stdout": "", "stderr": "boom\n"}),
    )

    def fake_worker_json(self, sandbox, subcommand, argv):
        assert subcommand == "sha256-file"
        return {"sha256": execution._sha256_path(trusted_server)}

    monkeypatch.setattr(execution.ModalAuthoringSession, "_worker_json", fake_worker_json)

    result = execution.execute_authoring_command(
        app_dir,
        ["claude", "--print", "fail"],
        timeout=30,
        sync_allowlist=["js/**"],
    )

    assert result.returncode == 1
    assert (app_dir / "js" / "existing.js").read_text(encoding="utf-8") == "known-good\n"
    assert result.metadata["profile"] == execution.TRUSTED_AUTHORING_PROFILE.name
    assert result.metadata["egress"]["applied"] is False
    assert result.metadata["removed_files"] == []
    assert result.metadata["returned_files"] == []
    assert terminated == ["terminated"]


def test_trusted_authoring_resource_key_stable_and_changes_on_relevant_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "server.py").write_text("print('trusted')\n", encoding="utf-8")
    monkeypatch.setenv(execution.AUTHORING_OWNER_SCOPE_ENV, "scope-a")

    first = execution._get_or_create_authoring_session(app_dir)
    second = execution._get_or_create_authoring_session(app_dir)

    assert first.resource_key == second.resource_key

    original_revision = execution.IMAGE_FAMILY_REVISIONS["authoring-image"]
    monkeypatch.setitem(execution.IMAGE_FAMILY_REVISIONS, "authoring-image", "rev-b")
    third = execution.ModalAuthoringSession(
        app_dir,
        owner_scope="scope-a",
        workspace_identity=execution._workspace_identity(app_dir),
        workspace_fingerprint=execution._workspace_fingerprint(app_dir),
        behavior_id=None,
    )
    assert third.resource_key != first.resource_key
    monkeypatch.setitem(execution.IMAGE_FAMILY_REVISIONS, "authoring-image", original_revision)

    before_mutation = first.resource_key
    (app_dir / "notes.txt").write_text("changed\n", encoding="utf-8")
    mutated = execution._get_or_create_authoring_session(app_dir)
    assert mutated.resource_key != before_mutation

    worker_key = execution._trusted_authoring_resource_key(
        profile=execution.FRESH_PYTHON_WORKER_PROFILE,
        workspace_identity=execution._workspace_identity(app_dir),
        behavior_id=None,
        owner_scope="scope-a",
        workspace_fingerprint=execution._workspace_fingerprint(app_dir),
        image_revision=execution.PYTHON_WORKER_IMAGE_REVISION,
    )
    assert worker_key != mutated.resource_key

    execution.close_all_authoring_sessions()


def test_trusted_authoring_resource_key_ignores_host_bookkeeping_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "server.py").write_text("print('trusted')\n", encoding="utf-8")
    monkeypatch.setenv(execution.AUTHORING_OWNER_SCOPE_ENV, "scope-bookkeeping")

    baseline = execution._get_or_create_authoring_session(app_dir).resource_key
    (app_dir / "app_manifest.json").write_text("{}", encoding="utf-8")
    (app_dir / "functional_results.json").write_text("{}", encoding="utf-8")
    (app_dir / "server_events.log").write_text("event\n", encoding="utf-8")
    (app_dir / "results").mkdir(exist_ok=True)
    (app_dir / "results" / "audit_report.json").write_text("{}", encoding="utf-8")
    (app_dir / "results" / "audit_summary.md").write_text("summary\n", encoding="utf-8")

    session = execution._get_or_create_authoring_session(app_dir)

    assert session.resource_key == baseline
    execution.close_all_authoring_sessions()


def test_cleanup_terminates_stale_trusted_sessions(tmp_path: Path, monkeypatch) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    terminated: list[str] = []

    class _FakeSandbox:
        def terminate(self):
            terminated.append("terminated")
            return None

    session = execution.ModalAuthoringSession(
        app_dir,
        owner_scope="scope-cleanup",
        workspace_identity=execution._workspace_identity(app_dir),
        workspace_fingerprint=execution._workspace_fingerprint(app_dir),
        behavior_id=None,
    )
    session._sandbox = _FakeSandbox()
    session._idle_timeout = 1
    session.last_used_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    execution._AUTHORING_SESSIONS[session.resource_key] = session

    execution._cleanup_stale_authoring_sessions("scope-cleanup")

    assert terminated == ["terminated"]
    assert session.resource_key not in execution._AUTHORING_SESSIONS


def test_busy_trusted_session_fails_closed(tmp_path: Path, monkeypatch) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setenv(execution.AUTHORING_OWNER_SCOPE_ENV, "scope-busy")
    monkeypatch.setattr(execution, "AUTHORING_SESSION_BUSY_WAIT_SECONDS", 0.0)

    session = execution._get_or_create_authoring_session(app_dir)
    assert session._busy_lock.acquire(blocking=False) is True

    try:
        with pytest.raises(RuntimeError, match="failing closed"):
            execution.execute_authoring_command(
                app_dir,
                ["claude", "--print", "busy"],
                timeout=30,
            )
    finally:
        session._busy_lock.release()
        execution.close_all_authoring_sessions()


def test_execution_provenance_appends_records_and_preserves_legacy_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    working_dir = tmp_path / "app"
    working_dir.mkdir()
    provenance_path = working_dir / execution.EXECUTION_PROVENANCE_FILENAME
    provenance_path.write_text(
        json.dumps({"legacy_stateless": {"sandbox_id": "old-sandbox"}}, indent=2),
        encoding="utf-8",
    )

    class _FakeSandbox:
        object_id = "sb-worker"

        def terminate(self):
            return None

    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_create_sandbox",
        lambda self, workspace_ingress, timeout, block_network: _FakeSandbox(),
    )
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, argv, timeout, env=None: {"returncode": 0, "stdout": "ok\n", "stderr": ""}),
    )

    runner = execution.ModalSandboxRunner(working_dir)
    runner.run_command(argv=["python", "-V"], timeout=30)
    runner.run_command(argv=["python", "-V"], timeout=30)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["legacy_stateless"] == {"sandbox_id": "old-sandbox"}
    assert len(provenance["records"]) == 2
    assert all(record["operation_class"] == execution.FRESH_PYTHON_WORKER_PROFILE.name for record in provenance["records"])
    assert all(record["image_family"] == "python-worker-image" for record in provenance["records"])
    assert all(record["image_revision"] == execution.PYTHON_WORKER_IMAGE_REVISION for record in provenance["records"])
    assert all(record["resource_key"] is None for record in provenance["records"])
    assert provenance["records"][0]["operation_id"] != provenance["records"][1]["operation_id"]


def test_worker_flows_remain_fresh_and_non_reusable_even_when_authoring_warm_start_is_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    script = app_dir / "sanity_check_function.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    create_calls: list[dict[str, object]] = []

    class _FakeSandbox:
        object_id = "sb-worker"

        def terminate(self):
            return None

    def fake_create_sandbox(self, workspace_ingress, *, timeout, block_network, idle_timeout=None):
        create_calls.append(
            {
                "profile": self.profile.name,
                "image_family": self.profile.image_family,
                "timeout": timeout,
                "block_network": block_network,
            }
        )
        return _FakeSandbox()

    monkeypatch.setenv(execution.TRUSTED_AUTHORING_WARM_START_ENV, "1")
    monkeypatch.setenv(execution.TRUSTED_AUTHORING_MODAL_APP_ENV, "deployed-authoring")
    monkeypatch.setattr(execution.ModalSandboxRunner, "_create_sandbox", fake_create_sandbox)
    monkeypatch.setattr(
        execution.ModalSandboxRunner,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, *, argv, timeout, env=None, pty=False: {"returncode": 0, "stdout": "ok\n", "stderr": ""}),
    )

    execution.run_python_script(app_dir, script, timeout=30)
    execution.run_python_script(app_dir, script, timeout=30)
    execution.run_trusted_worker(app_dir, "verify-snapshot", timeout=30, argv=["--root", "."])
    execution.run_trusted_worker(
        app_dir,
        "validate-app-runtime",
        timeout=30,
        argv=["--root", "."],
        profile=execution.FRESH_BROWSER_WORKER_PROFILE,
    )

    assert len(create_calls) == 4
    assert [call["profile"] for call in create_calls] == [
        execution.FRESH_PYTHON_WORKER_PROFILE.name,
        execution.FRESH_PYTHON_WORKER_PROFILE.name,
        execution.FRESH_PYTHON_WORKER_PROFILE.name,
        execution.FRESH_BROWSER_WORKER_PROFILE.name,
    ]
    assert all(call["block_network"] is True for call in create_calls)


def test_execute_authoring_command_provenance_marks_warm_start_explicitly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    trusted_server = app_dir / "server.py"
    trusted_server.write_text("print('trusted')\n", encoding="utf-8")

    class _FakeSandbox:
        object_id = "sb-authoring-warm"

        def terminate(self):
            return None

    monkeypatch.setenv(execution.TRUSTED_AUTHORING_WARM_START_ENV, "1")
    monkeypatch.setenv(execution.TRUSTED_AUTHORING_MODAL_APP_ENV, "deployed-authoring")
    monkeypatch.setattr(execution.ModalAuthoringSession, "_attach_named_sandbox", lambda self: _FakeSandbox())
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_execute_in_sandbox",
        staticmethod(lambda sandbox, *, argv, timeout, env=None, pty=False: {"returncode": 0, "stdout": "ok\n", "stderr": ""}),
    )
    monkeypatch.setattr(
        execution.ModalAuthoringSession,
        "_worker_json",
        lambda self, sandbox, subcommand, argv: {"sha256": execution._sha256_path(trusted_server)},
    )

    execution.execute_authoring_command(
        app_dir,
        ["python", "-c", "print('ok')"],
        timeout=30,
    )

    provenance = json.loads((app_dir / execution.EXECUTION_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    record = provenance["records"][0]
    assert record["operation_class"] == execution.TRUSTED_AUTHORING_PROFILE.name
    assert record["warm_start_enabled"] is True
    assert record["warm_start_attempted"] is True
    assert record["warm_start_hit"] is True
    assert record["warm_start_mode"] == "named-sandbox"
    assert record["warm_start_name"].startswith("redteam-authoring-")
    assert record["warm_start_lookup_app"] == "deployed-authoring"
    execution.close_all_authoring_sessions()


@pytest.mark.skipif(
    os.getenv("AGENTLAB_REDTEAM_MODAL_SMOKE") != "1",
    reason="Set AGENTLAB_REDTEAM_MODAL_SMOKE=1 to run the real Modal smoke test.",
)
def test_modal_smoke_sanity_script_round_trips_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(execution.EXECUTION_BACKEND_ENV, "modal")
    app_dir = tmp_path / "app"
    (app_dir / "results").mkdir(parents=True)
    script = app_dir / "sanity_check_function.py"
    script.write_text(
        (
            "from pathlib import Path\n"
            "Path('results/smoke.txt').write_text('ok', encoding='utf-8')\n"
            "print('ok')\n"
        ),
        encoding="utf-8",
    )

    result = execution.run_python_script(
        app_dir,
        script,
        timeout=120,
        sync_allowlist=["results/**"],
        block_network=True,
    )

    assert result.returncode == 0
    assert "ok" in result.stdout
    assert (app_dir / "results" / "smoke.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.skipif(
    os.getenv("AGENTLAB_REDTEAM_MODAL_SMOKE") != "1"
    or not os.getenv(execution.TRUSTED_AUTHORING_MODAL_APP_ENV),
    reason=(
        "Set AGENTLAB_REDTEAM_MODAL_SMOKE=1 and "
        f"{execution.TRUSTED_AUTHORING_MODAL_APP_ENV} to run the trusted authoring warm-start smoke test."
    ),
)
def test_modal_smoke_trusted_authoring_warm_start_reuses_named_sandbox(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(execution.EXECUTION_BACKEND_ENV, "modal")
    monkeypatch.setenv(execution.TRUSTED_AUTHORING_WARM_START_ENV, "1")
    execution.close_all_authoring_sessions()
    app_dir = tmp_path / "app"
    (app_dir / "results").mkdir(parents=True)
    (app_dir / "server.py").write_text("print('trusted')\n", encoding="utf-8")

    first = execution.execute_authoring_command(
        app_dir,
        [
            "python",
            "-c",
            (
                "from pathlib import Path\n"
                "Path('results/authoring_one.txt').write_text('one', encoding='utf-8')\n"
                "print('one')\n"
            ),
        ],
        timeout=180,
        sync_allowlist=["results/**"],
    )
    assert first.returncode == 0
    assert (app_dir / "results" / "authoring_one.txt").read_text(encoding="utf-8") == "one"

    resource_key = first.metadata["resource_key"]
    execution._AUTHORING_SESSIONS.pop(resource_key, None)

    second = execution.execute_authoring_command(
        app_dir,
        [
            "python",
            "-c",
            (
                "from pathlib import Path\n"
                "Path('results/authoring_two.txt').write_text('two', encoding='utf-8')\n"
                "print('two')\n"
            ),
        ],
        timeout=180,
        sync_allowlist=["results/**"],
    )

    assert second.returncode == 0
    assert second.metadata["warm_start_attempted"] is True
    assert second.metadata["warm_start_hit"] is True
    assert second.metadata["warm_start_mode"] == "named-sandbox"
    assert (app_dir / "results" / "authoring_two.txt").read_text(encoding="utf-8") == "two"
    execution.close_all_authoring_sessions()


# ---------------------------------------------------------------------------
# build_claude_env / build_claude_command
# ---------------------------------------------------------------------------


def test_build_claude_env_forwards_oauth_token(monkeypatch) -> None:
    """CLAUDE_CODE_OAUTH_TOKEN is forwarded when set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")

    env = execution.build_claude_env()

    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}


def test_build_claude_env_oauth_token_suppresses_api_key(monkeypatch) -> None:
    """CLAUDE_CODE_OAUTH_TOKEN suppresses ANTHROPIC_API_KEY (precedence issue)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-2")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")

    env = execution.build_claude_env()

    assert env == {
        "OPENROUTER_API_KEY": "key-2",
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test",
    }
    assert "ANTHROPIC_API_KEY" not in env


def test_build_claude_env_api_key_forwarded_without_oauth(monkeypatch) -> None:
    """ANTHROPIC_API_KEY is forwarded normally when no OAuth token is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-2")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    env = execution.build_claude_env()

    assert env == {
        "ANTHROPIC_API_KEY": "key-1",
        "OPENROUTER_API_KEY": "key-2",
    }


def test_build_claude_env_omits_unset_vars(monkeypatch) -> None:
    """Unset env vars are excluded from the result."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    env = execution.build_claude_env()

    assert env == {}


def test_build_claude_command_basic() -> None:
    cmd = execution.build_claude_command("hello world")

    assert cmd == [
        "claude",
        "--print",
        "--permission-mode",
        "plan",
        "--verbose",
        "--effort",
        "high",
        "hello world",
    ]


def test_build_claude_command_skip_permissions(monkeypatch) -> None:
    monkeypatch.setenv(execution.CLAUDE_SKIP_PERMISSIONS_ENV, "1")

    cmd = execution.build_claude_command("hello")

    assert cmd == [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--permission-mode",
        "plan",
        "--verbose",
        "--effort",
        "high",
        "hello",
    ]
