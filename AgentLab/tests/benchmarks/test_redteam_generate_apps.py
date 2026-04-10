from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentlab.benchmarks.redteam import generate_apps
from agentlab.benchmarks.redteam.app_artifacts import (
    APP_MANIFEST_CONTRACT_VERSION,
    compute_docs_snapshot,
)
from agentlab.benchmarks.redteam.errors import config_error
from agentlab.benchmarks.redteam.phase_ids import PHASE_4B, PHASE_5


def _init_git_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True)
    (repo_root / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return repo_root


def _write_benchmark(path: Path, behavior_ids: list[str]) -> None:
    path.write_text(
        json.dumps([{"id": behavior_id} for behavior_id in behavior_ids], indent=2),
        encoding="utf-8",
    )


def _write_manifest(app_dir: Path, manifest: dict) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _published_app_dir(output_dir: Path, behavior_id: str) -> Path:
    return output_dir / behavior_id


def _write_manuals(repo_root: Path) -> dict:
    docs_dir = repo_root / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "mail.md").write_text("# Mail Manual\n", encoding="utf-8")
    return compute_docs_snapshot(
        {"docs_path": "apps/user-manuals/mail"},
        repo_root_path=repo_root,
    )


def _write_trusted_server(app_dir: Path) -> None:
    template_path = Path(generate_apps.__file__).resolve().parent / "templates" / "server.py"
    (app_dir / "server.py").write_text(
        template_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _write_runtime_assets(app_dir: Path) -> None:
    (app_dir / "benign").mkdir(parents=True, exist_ok=True)
    (app_dir / "adversarial_demo_v0").mkdir(parents=True, exist_ok=True)
    _write_trusted_server(app_dir)
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "benign" / "data.js").write_text("const MODE = 'benign';\n", encoding="utf-8")
    (app_dir / "adversarial_demo_v0" / "data.js").write_text(
        "const MODE = 'adversarial';\n",
        encoding="utf-8",
    )


def _write_functional_results(app_dir: Path, *, malformed: bool = False) -> None:
    path = app_dir / "functional_results.json"
    if malformed:
        path.write_text("{bad json\n", encoding="utf-8")
        return
    path.write_text(
        json.dumps(
            {
                "backends": {
                    "agent-browser": {
                        "backend": "agent-browser",
                        "function_pass_rate": 1.0,
                        "real_pass_rate": 1.0,
                        "function_tasks": {
                            "total": 1,
                            "passed": 1,
                            "results": [{"task_id": "task_1", "passed": True}],
                        },
                        "real_tasks": {
                            "total": 1,
                            "passed": 1,
                            "results": [{"task_id": "task_r1", "passed": True}],
                        },
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _successful_manifest(behavior_id: str) -> dict:
    return {
        "contract_version": APP_MANIFEST_CONTRACT_VERSION,
        "app_id": behavior_id,
        "platform": "mail",
        "docs_path": "apps/user-manuals/mail",
        "docs_snapshot": {
            "docs_path": "apps/user-manuals/mail",
            "source_type": "directory",
            "corpus_sha256": "sha256:manuals",
            "files": [{"path": "mail.md", "sha256": "abc123"}],
        },
        "shared_seed_version": 1,
        "shared_seed_hash": "sha256:test-seed",
        "behavior_ids": [behavior_id],
        "pages": [
            {
                "id": "mail",
                "base_site_url": "https://mail.example.com",
                "subdomains": ["/mail/u/0/#inbox"],
                "details": {"/mail/u/0/#inbox": "Inbox"},
                "screenshots": [],
                "existing_path": None,
                "skip_modification": False,
            }
        ],
        "start_page": "https://mail.example.com/mail/u/0/#inbox",
        "execution_backend": {"selected_backend": "modal"},
        "generation": {
            "status": "succeeded",
            "functional_tests_requested": True,
            "functional_backend": "agent-browser",
            "hardening_rounds": 0,
            "run_final_regression": False,
            "phases": {
                PHASE_4B: {"status": "skipped"},
                PHASE_5: {"status": "skipped"},
            },
        },
        "variant_generation": {"status": "validated"},
        "validation": {"passed": True},
        "functional_tests": {
            "function_sanity_passed": True,
            "real_sanity_passed": True,
            "function_evaluation": {
                "ran": True,
                "backend": "agent-browser",
                "agent_config": "agent",
                "pass_rate": 1.0,
                "total": 1,
                "passed": 1,
                "results_dir": "results/function",
                "error": None,
            },
            "real_evaluation": {
                "ran": True,
                "backend": "agent-browser",
                "agent_config": "agent",
                "pass_rate": 1.0,
                "total": 1,
                "passed": 1,
                "results_dir": "results/real",
                "error": None,
            },
            "quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 0.8},
            "real_quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 1.0},
        },
    }


def test_generate_apps_resume_skips_only_successful_manifests(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["success", "failed", "missing"])
    output_dir = repo_root / "apps"
    docs_dir = repo_root / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "mail.md").write_text("# Mail Manual\n", encoding="utf-8")

    success_manifest = _successful_manifest("success")
    success_manifest["docs_snapshot"] = compute_docs_snapshot(
        {"docs_path": "apps/user-manuals/mail"},
        repo_root_path=repo_root,
    )

    _write_manifest(
        _published_app_dir(output_dir, "success"),
        success_manifest,
    )
    _write_runtime_assets(_published_app_dir(output_dir, "success"))
    _write_functional_results(_published_app_dir(output_dir, "success"))
    _write_manifest(
        _published_app_dir(output_dir, "failed"),
        {
            "contract_version": APP_MANIFEST_CONTRACT_VERSION,
            "generation": {"status": "failed", "functional_tests_requested": False},
        },
    )

    calls: list[str] = []

    def fake_generate_single(**kwargs):
        calls.append(kwargs["behavior"]["id"])
        manifest = _successful_manifest(kwargs["behavior"]["id"])
        return manifest

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--resume",
        ]
    )

    assert calls == ["failed", "missing"]
    assert not (repo_root / ".controller-worktrees").exists()


def test_generate_apps_rejects_duplicate_behavior_ids(tmp_path: Path) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo", "demo"])
    output_dir = repo_root / "apps"

    with pytest.raises(SystemExit):
        generate_apps.main(
            [
                "--benchmark-file",
                str(benchmark_file),
                "--output-dir",
                str(output_dir),
            ]
        )


def test_generate_apps_passes_functional_backend_options(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"
    captured: dict[str, object] = {}

    def fake_generate_single(**kwargs):
        captured.update(kwargs)
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--functional-backend",
            "agent-browser",
            "--functional-agent-config",
            "fake.Agent",
        ]
    )

    assert captured["functional_backend"] == "agent-browser"
    assert captured["functional_agent_config"] == "fake.Agent"


def test_generate_apps_dry_run_lists_apps_without_generating(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"

    monkeypatch.delenv("AGENTLAB_REDTEAM_SKIP_CONFIG_VALIDATION", raising=False)
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.pipeline_config.PipelineConfig.validate",
        lambda self, check_cli=True: [],
    )
    monkeypatch.setattr(
        generate_apps,
        "_generate_single",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not generate apps")),
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_apps.main(
            [
                "--benchmark-file",
                str(benchmark_file),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "DRY RUN: would generate 1 app(s):" in captured.out
    assert "demo" in captured.out


def test_generate_apps_dry_run_exits_on_fatal_config_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"

    monkeypatch.delenv("AGENTLAB_REDTEAM_SKIP_CONFIG_VALIDATION", raising=False)
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.pipeline_config.PipelineConfig.validate",
        lambda self, check_cli=True: [
            config_error(
                "E_CONFIG_API_KEY_MISSING",
                "No API key found. Set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN.",
            )
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_apps.main(
            [
                "--benchmark-file",
                str(benchmark_file),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )

    assert exc_info.value.code == 1


def test_generate_apps_resume_retries_when_functional_tests_are_missing(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"

    _write_manifest(
        _published_app_dir(output_dir, "demo"),
        {
            "contract_version": APP_MANIFEST_CONTRACT_VERSION,
            "generation": {"status": "succeeded", "functional_tests_requested": False},
            "variant_generation": {"status": "validated"},
            "validation": {"passed": True},
        },
    )

    calls: list[str] = []

    def fake_generate_single(**kwargs):
        calls.append(kwargs["behavior"]["id"])
        manifest = _successful_manifest(kwargs["behavior"]["id"])
        return manifest

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--resume",
            "--functional-tests",
        ]
    )

    assert calls == ["demo"]


def test_generate_apps_resume_retries_when_quality_gate_failed(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"

    _write_manifest(
        _published_app_dir(output_dir, "demo"),
        {
            "contract_version": APP_MANIFEST_CONTRACT_VERSION,
            "generation": {"status": "succeeded", "functional_tests_requested": True},
            "functional_tests": {
                "quality_gate": {"passed": False, "pass_rate": 0.4, "threshold": 0.8}
            },
            "variant_generation": {"status": "validated"},
            "validation": {"passed": True},
        },
    )
    _write_runtime_assets(_published_app_dir(output_dir, "demo"))
    _write_functional_results(_published_app_dir(output_dir, "demo"))

    calls: list[str] = []

    def fake_generate_single(**kwargs):
        calls.append(kwargs["behavior"]["id"])
        manifest = _successful_manifest(kwargs["behavior"]["id"])
        return manifest

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--resume",
            "--functional-tests",
        ]
    )

    assert calls == ["demo"]


def test_generate_apps_resume_retries_when_requested_backend_changed(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"
    docs_snapshot = _write_manuals(repo_root)

    manifest = _successful_manifest("demo")
    manifest["docs_snapshot"] = docs_snapshot
    manifest["generation"]["functional_backend"] = "generic-agent"
    manifest["functional_tests"]["function_evaluation"]["backend"] = "generic-agent"
    manifest["functional_tests"]["real_evaluation"]["backend"] = "generic-agent"
    _write_manifest(_published_app_dir(output_dir, "demo"), manifest)
    _write_runtime_assets(_published_app_dir(output_dir, "demo"))
    (output_dir / "demo" / "functional_results.json").write_text(
        json.dumps(
            {
                "backends": {
                    "generic-agent": {
                        "backend": "generic-agent",
                        "function_pass_rate": 1.0,
                        "real_pass_rate": 0.0,
                        "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                        "real_tasks": {"total": 0, "passed": 0, "results": []},
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, bool]] = []

    def fake_generate_single(**kwargs):
        calls.append((kwargs["behavior"]["id"], kwargs["resume_generation"]))
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--resume",
            "--functional-backend",
            "agent-browser",
        ]
    )

    assert calls == [("demo", True)]


def test_generate_apps_resume_retries_when_functional_results_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"
    docs_snapshot = _write_manuals(repo_root)

    manifest = _successful_manifest("demo")
    manifest["docs_snapshot"] = docs_snapshot
    _write_manifest(_published_app_dir(output_dir, "demo"), manifest)
    _write_runtime_assets(_published_app_dir(output_dir, "demo"))

    calls: list[tuple[str, bool]] = []

    def fake_generate_single(**kwargs):
        calls.append((kwargs["behavior"]["id"], kwargs["resume_generation"]))
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--resume",
        ]
    )

    assert calls == [("demo", True)]


def test_generate_apps_resume_retries_when_functional_results_are_malformed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"
    docs_snapshot = _write_manuals(repo_root)

    manifest = _successful_manifest("demo")
    manifest["docs_snapshot"] = docs_snapshot
    _write_manifest(_published_app_dir(output_dir, "demo"), manifest)
    _write_runtime_assets(_published_app_dir(output_dir, "demo"))
    _write_functional_results(_published_app_dir(output_dir, "demo"), malformed=True)

    calls: list[tuple[str, bool]] = []

    def fake_generate_single(**kwargs):
        calls.append((kwargs["behavior"]["id"], kwargs["resume_generation"]))
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--resume",
        ]
    )

    assert calls == [("demo", True)]


def test_generate_apps_resume_retries_when_docs_snapshot_is_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = repo_root / "apps"
    docs_dir = repo_root / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manual_path = docs_dir / "mail.md"
    manual_path.write_text("# Mail Manual\n", encoding="utf-8")

    manifest = _successful_manifest("demo")
    manifest["docs_snapshot"] = compute_docs_snapshot(
        {"docs_path": "apps/user-manuals/mail"},
        repo_root_path=repo_root,
    )
    _write_manifest(_published_app_dir(output_dir, "demo"), manifest)
    _write_runtime_assets(_published_app_dir(output_dir, "demo"))
    _write_functional_results(_published_app_dir(output_dir, "demo"))
    manual_path.write_text("# Mail Manual\nUpdated\n", encoding="utf-8")

    calls: list[str] = []

    def fake_generate_single(**kwargs):
        calls.append(kwargs["behavior"]["id"])
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--resume",
        ]
    )

    assert calls == ["demo"]


def test_generate_apps_skip_existing_uses_manifest_existence(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["existing", "fresh"])
    output_dir = repo_root / "apps"

    _write_manifest(
        _published_app_dir(output_dir, "existing"),
        {
            "contract_version": APP_MANIFEST_CONTRACT_VERSION,
            "generation": {"status": "failed", "functional_tests_requested": False},
        },
    )

    calls: list[str] = []

    def fake_generate_single(**kwargs):
        calls.append(kwargs["behavior"]["id"])
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--skip-existing",
        ]
    )

    assert calls == ["fresh"]
    assert not (repo_root / ".controller-worktrees").exists()


def test_generate_apps_resolves_relative_output_dir_from_caller_cwd(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    caller_dir = repo_root / "AgentLab"
    caller_dir.mkdir()
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    captured_output_dirs: list[str] = []

    def fake_generate_single(**kwargs):
        captured_output_dirs.append(kwargs["output_dir"])
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)
    monkeypatch.chdir(caller_dir)

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            "apps",
        ]
    )

    expected_output_dir = caller_dir / "apps"
    assert captured_output_dirs == [str(expected_output_dir)]
    assert (expected_output_dir / "manifest.json").exists()


def test_generate_apps_rejects_mutually_exclusive_resume_and_skip_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    called = False

    def fake_generate_single(**kwargs):
        nonlocal called
        called = True
        return _successful_manifest(kwargs["behavior"]["id"])

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    with pytest.raises(SystemExit) as exc_info:
        generate_apps.main(
            [
                "--benchmark-file",
                str(benchmark_file),
                "--output-dir",
                str(tmp_path / "apps"),
                "--resume",
                "--skip-existing",
            ]
        )

    assert exc_info.value.code == 1
    assert called is False


def test_generate_apps_counts_failed_generation_status_in_root_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file, ["demo"])
    output_dir = tmp_path / "apps"

    def fake_generate_single(**kwargs):
        return {
            "contract_version": APP_MANIFEST_CONTRACT_VERSION,
            "behavior_id": kwargs["behavior"]["id"],
            "execution_backend": {"selected_backend": "modal"},
            "generation": {"status": "failed", "functional_tests_requested": False},
            "variant_generation": {"status": "failed"},
            "validation": {"passed": False},
            "errors": ["validation failed"],
        }

    monkeypatch.setattr(generate_apps, "_generate_single", fake_generate_single)

    with pytest.raises(SystemExit) as exc_info:
        generate_apps.main(
            [
                "--benchmark-file",
                str(benchmark_file),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert exc_info.value.code == 1
    root_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["n_succeeded"] == 0
    assert root_manifest["n_failed"] == 1


def test_generate_apps_module_entrypoint_no_longer_crashes_after_main(tmp_path: Path) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text("[]\n", encoding="utf-8")
    output_dir = tmp_path / "apps"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentlab.benchmarks.redteam.generate_apps",
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
        ],
        cwd="/Users/ashtonchew/projects/browser-sim/AgentLab",
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": "/Users/ashtonchew/projects/browser-sim/AgentLab/src",
        },
    )

    assert result.returncode == 0
    assert "NameError" not in result.stderr
