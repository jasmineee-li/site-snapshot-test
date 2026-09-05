"""Installed-wheel, historical-readback, and boundary evidence for #136."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts import readiness_audit

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "namespace_compatibility"
HISTORICAL_ROOT = FIXTURE_ROOT / "historical_worldsim_run"
ADAPTER_WHEEL_FIXTURE_ROOT = FIXTURE_ROOT / "adapter_wheel_0_1_0"
SIDECAR_ROOT = PACKAGE_ROOT / "packages" / "worldsim-agentlab-runner"

RUN_ID = "run-fe8344cba2614126b55b2a5c6a0f0c65"
DEFINITION_DIGEST = "35629a1f4cd9dac92a5a5820dcd17c0029fbe1881f781ff526b442276b791c56"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHON_DOTENV_DISABLED"] = "1"
    return env


def _historical_snapshot(root: Path, namespace: str, env_name: str) -> dict[str, object]:
    code = """
import json
import sys
from importlib import import_module
from pathlib import Path

root = Path(sys.argv[1])
state_module = import_module(sys.argv[2] + ".state")
status_module = import_module(sys.argv[2] + ".cli.status")
state = state_module.load_state()
assert isinstance(state, dict)
status = status_module.build_status_payload(root)
assert isinstance(status, dict)
manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
checkpoints = {}
for path in sorted(root.rglob("*.json")):
    relative = path.relative_to(root).as_posix()
    if "checkpoint" in relative or relative.endswith(".manifest.json"):
        checkpoints[relative] = json.loads(path.read_text(encoding="utf-8"))
summary = status.get("phase4_summary")
assert isinstance(summary, dict)
print(json.dumps({
    "run_id": state.get("run_id"),
    "definition_digest": state.get("definition_digest"),
    "lifecycle_state": state.get("status"),
    "result_summary": {
        "total": summary.get("total"),
        "final_status_counts": summary.get("final_status_counts"),
        "gate1_valid": summary.get("gate1_valid"),
        "final_asr": summary.get("final_asr"),
    },
    "artifact_paths": [item.get("path") for item in manifest.get("artifacts", [])],
    "checkpoint_metadata": checkpoints,
}, sort_keys=True))
"""
    env = _subprocess_env()
    env.pop("WARP_TASKGEN_STATE_DIR", None)
    env.pop("WORLDSIM_STATE_DIR", None)
    env[env_name] = str(root)
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), namespace],
        cwd=root.parent,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _prepare_historical_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "historical-worldsim-run"
    shutil.copytree(HISTORICAL_ROOT, root)
    state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["logs_dir"] = str(root)
    state["state_file"] = str(state_path)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def test_canonical_reader_preserves_historical_worldsim_readback(tmp_path: Path) -> None:
    root = _prepare_historical_fixture(tmp_path)
    canonical = _historical_snapshot(root, "warp_taskgen", "WARP_TASKGEN_STATE_DIR")
    legacy_environment = _historical_snapshot(root, "warp_taskgen", "WORLDSIM_STATE_DIR")

    assert canonical == legacy_environment
    assert canonical["run_id"] == RUN_ID
    assert canonical["definition_digest"] == DEFINITION_DIGEST
    assert canonical["lifecycle_state"] == "complete"
    assert canonical["result_summary"] == {
        "final_asr": 1.0,
        "final_status_counts": {"complied": 1},
        "gate1_valid": 1,
        "total": 1,
    }
    assert canonical["artifact_paths"] == [
        "phase_2/shards/shard-000.manifest.json",
        "phase_2/text_fill/checkpoints/plan-000.json",
        "phase_2/feasibility_checkpoints/adv_historical_gitlab.json",
        "phase_4/20260812_200538/adv_historical_gitlab/results.json",
        "phase_4/20260812_200538/adv_historical_gitlab/eval_awareness_iterator_checkpoint.json",
    ]
    checkpoints = canonical["checkpoint_metadata"]
    assert isinstance(checkpoints, dict)
    assert checkpoints["phase_2/shards/shard-000.manifest.json"]["schema_version"] == (
        "worldsim-phase-2a-shard-manifest-v1"
    )
    assert (
        checkpoints["phase_2/text_fill/checkpoints/plan-000.json"]["schema_version"]
        == "worldsim-phase-2b-text-fill-checkpoint-v1"
    )
    assert (
        checkpoints["phase_2/feasibility_checkpoints/adv_historical_gitlab.json"]["schema_version"]
        == "worldsim-phase-2c-feasibility-checkpoint-v1"
    )
    assert (
        checkpoints[
            "phase_4/20260812_200538/adv_historical_gitlab/eval_awareness_iterator_checkpoint.json"
        ]["schema_version"]
        == "worldsim-phase-4-eval-awareness-iterator-checkpoint-v1"
    )


def test_environment_alias_precedence_and_legacy_fallback(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    legacy_root = tmp_path / "legacy"
    code = """
import os
from pathlib import Path
from warp_taskgen._paths import find_repo_root
from warp_taskgen.state import get_state_dir

import json
print(json.dumps({
    "state_dir": str(get_state_dir()),
    "repo_root": str(find_repo_root()),
}))
"""

    def run(extra: dict[str, str]) -> dict[str, object]:
        env = _subprocess_env()
        for key in (
            "WARP_TASKGEN_STATE_DIR",
            "WORLDSIM_STATE_DIR",
            "WARP_TASKGEN_REPO_ROOT",
            "WORLDSIM_REPO_ROOT",
            "WARP_TASKGEN_AGENTLAB_RUNNER_CMD",
            "WORLDSIM_AGENTLAB_RUNNER_CMD",
        ):
            env.pop(key, None)
        env.update(extra)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    canonical = run(
        {
            "WARP_TASKGEN_STATE_DIR": str(canonical_root),
            "WORLDSIM_STATE_DIR": str(legacy_root),
            "WARP_TASKGEN_REPO_ROOT": "/canonical/repo",
            "WORLDSIM_REPO_ROOT": "/legacy/repo",
            "WARP_TASKGEN_AGENTLAB_RUNNER_CMD": "canonical-runner",
            "WORLDSIM_AGENTLAB_RUNNER_CMD": "legacy-runner",
        }
    )
    assert canonical == {
        "state_dir": str(canonical_root),
        "repo_root": "/canonical/repo",
    }

    legacy = run(
        {
            "WORLDSIM_STATE_DIR": str(legacy_root),
            "WORLDSIM_REPO_ROOT": "/legacy/repo",
            "WORLDSIM_AGENTLAB_RUNNER_CMD": "legacy-runner",
        }
    )
    assert legacy == {
        "state_dir": str(legacy_root),
        "repo_root": "/legacy/repo",
    }


def test_agentlab_sidecar_keeps_historical_identity_and_canonical_core_bridge() -> None:
    metadata = tomllib.loads((SIDECAR_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "warp-taskgen-agentlab-runner"
    assert metadata["project"]["scripts"] == {
        "warp-taskgen-agentlab-runner": "worldsim_agentlab_runner.cli:main",
        "worldsim-agentlab-runner": "worldsim_agentlab_runner.cli:main",
    }
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (SIDECAR_ROOT / "src" / "worldsim_agentlab_runner").glob("*.py")
    )
    assert "from worldsim." not in source
    assert "import worldsim." not in source
    assert "warp_taskgen.phase_4" in (
        SIDECAR_ROOT / "src" / "worldsim_agentlab_runner" / "sync_pvpo.py"
    ).read_text(encoding="utf-8")


def test_named_modal_secret_prefers_canonical_and_accepts_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warp_taskgen import modal_sandbox

    selected: list[str] = []
    monkeypatch.setattr(
        modal_sandbox.modal.Secret,
        "from_name",
        lambda name: selected.append(name) or name,
    )
    monkeypatch.setenv("WARP_TASKGEN_CLAUDE_MODAL_SECRET", "canonical-secret-name")
    monkeypatch.setenv("WORLDSIM_CLAUDE_MODAL_SECRET", "legacy-secret-name")
    assert modal_sandbox._build_claude_secrets() == ["canonical-secret-name"]
    assert selected == ["canonical-secret-name"]

    selected.clear()
    monkeypatch.delenv("WARP_TASKGEN_CLAUDE_MODAL_SECRET")
    assert modal_sandbox._build_claude_secrets() == ["legacy-secret-name"]
    assert selected == ["legacy-secret-name"]


def test_active_source_readiness_has_no_legacy_namespace_imports() -> None:
    # Keep the repository-wide import scope, but avoid the unrelated line,
    # generated-file, and token scans owned by the canonical audit command.
    paths = [path for path in readiness_audit._git_ls_files() if path.endswith(".py")]
    assert readiness_audit._legacy_namespace_import_findings(paths) == []
    assert readiness_audit._legacy_phase_import_findings(paths) == []
    assert readiness_audit._active_facade_import_findings(paths) == []


def test_historical_fixture_is_secret_free() -> None:
    paths = [str(path) for path in FIXTURE_ROOT.rglob("*") if path.is_file()]
    assert readiness_audit._token_findings(paths) == []


def test_core_distribution_version_is_bumped_without_bumping_sidecars() -> None:
    package_metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    import warp_taskgen

    assert package_metadata["project"]["version"] == "0.1.1"
    assert warp_taskgen.__version__ == "0.1.1"
    assert (
        tomllib.loads((SIDECAR_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "version"
        ]
        == "0.1.0"
    )
    evaluator_root = PACKAGE_ROOT / "packages" / "warp-taskgen-webarena-verified"
    assert (
        tomllib.loads((evaluator_root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "version"
        ]
        == "0.1.0"
    )


def test_adapter_wheel_fixture_is_explicitly_old_and_retains_retired_surfaces() -> None:
    metadata = tomllib.loads(
        (ADAPTER_WHEEL_FIXTURE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["name"] == "warp-taskgen"
    assert metadata["project"]["version"] == "0.1.0"
    assert metadata["project"]["scripts"]["worldsim"] == "warp_taskgen.main:main"
    assert "worldsim" in metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert (ADAPTER_WHEEL_FIXTURE_ROOT / "worldsim" / "__init__.py").is_file()
