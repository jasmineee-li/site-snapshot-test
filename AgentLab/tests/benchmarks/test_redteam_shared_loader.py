from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlab.benchmarks.redteam.benchmark import PregeneratedAppManifestError, RedteamBenchmark
from agentlab.benchmarks.redteam.app_artifacts import compute_docs_snapshot
from agentlab.benchmarks.redteam.phase_ids import PHASE_4B, PHASE_5


def _write_trusted_server(app_dir: Path) -> None:
    from agentlab.benchmarks.redteam import benchmark as benchmark_module

    template_path = Path(benchmark_module.__file__).resolve().parent / "templates" / "server.py"
    (app_dir / "server.py").write_text(
        template_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _patch_benchmark_action_args(monkeypatch) -> None:
    class FakeHighLevelActionSetArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.benchmark.HighLevelActionSetArgs",
        FakeHighLevelActionSetArgs,
    )


def _write_benchmark(
    path: Path,
    *,
    behavior_id: str = "demo",
    behavior_ids: list[str] | None = None,
) -> None:
    selected_behavior_ids = behavior_ids or [behavior_id]
    path.write_text(
        json.dumps(
            [
                {
                    "id": selected_behavior_id,
                    "doc": "Do something safe",
                    "description": "target description",
                    "simulation_config": {
                        "pages": [
                            {
                                "id": "mail",
                                "base_site_url": "https://mail.example.com",
                                "subdomains": ["/mail/u/0/#inbox"],
                                "details": {"/mail/u/0/#inbox": "Inbox"},
                            }
                        ],
                    },
                }
                for selected_behavior_id in selected_behavior_ids
            ]
        ),
        encoding="utf-8",
    )


def _write_platform_manifest(path: Path, *, behavior_id: str = "demo", app_id: str = "shared-mail") -> None:
    path.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": app_id,
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": behavior_id,
                                "primary_platform": "mail",
                                "entry_route": "#inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [
                                    {"domain": "mail.example.com", "mode": "primary_spa"}
                                ],
                                "mock_endpoints": ["/mock/mail/submit"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_app_fixture(app_dir: Path, *, repo_root: Path, behavior_id: str = "demo") -> None:
    (app_dir / "js").mkdir(parents=True, exist_ok=True)
    (app_dir / "css").mkdir(exist_ok=True)
    (app_dir / "benign").mkdir(exist_ok=True)
    (app_dir / f"adversarial_{behavior_id}_v0").mkdir(exist_ok=True)
    _write_trusted_server(app_dir)
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "js" / "state.js").write_text("const AppState = {};\n", encoding="utf-8")
    (app_dir / "js" / "components.js").write_text("const Components = {};\n", encoding="utf-8")
    (app_dir / "js" / "views.js").write_text("const Views = {};\n", encoding="utf-8")
    (app_dir / "js" / "app.js").write_text("const App = {};\n", encoding="utf-8")
    (app_dir / "js" / "data.js").write_text("const SOURCE = 'runtime';\n", encoding="utf-8")
    (app_dir / "benign" / "data.js").write_text("const SOURCE = 'benign';\n", encoding="utf-8")
    (app_dir / f"adversarial_{behavior_id}_v0" / "data.js").write_text(
        "const SOURCE = 'adversarial';\n",
        encoding="utf-8",
    )
    (app_dir / "real-tasks").mkdir(exist_ok=True)
    (app_dir / "real-tasks" / "task_r1.py").write_text(
        "def verify(server_url):\n    return True, 'ok'\n",
        encoding="utf-8",
    )
    (app_dir / "real-tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "task_r1",
                    "instruction": "Reply safely",
                    "success_condition": "Open the inbox",
                    "verify": "real-tasks/task_r1.py",
                    "behavior_compatibility": {"behavior_id": behavior_id},
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": 3,
                "app_id": "shared-mail",
                "platform": "mail",
                "docs_path": "apps/user-manuals/mail",
                "docs_snapshot": compute_docs_snapshot(
                    {"docs_path": "apps/user-manuals/mail"},
                    repo_root_path=repo_root,
                ),
                "shared_seed_version": 1,
                "shared_seed_hash": "sha256:test-seed",
                "behavior_ids": [behavior_id],
                "pages": [
                    {
                        "id": "mail",
                        "base_site_url": "https://mail.example.com",
                        "subdomains": ["/mail/u/0/#inbox"],
                        "details": {"/mail/u/0/#inbox": "Inbox"},
                    }
                ],
                "start_page": "https://mail.example.com/mail/u/0/#inbox",
                "variant_generation": {"status": "validated"},
                "validation": {"passed": True},
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
                        "results": [{"task_id": "task_r1", "passed": True}],
                        "error": None,
                    },
                    "quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 0.8},
                    "real_quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )
    behavior_contract_path = app_dir / "behaviors" / f"{behavior_id}.json"
    behavior_contract_path.parent.mkdir(parents=True, exist_ok=True)
    behavior_contract_path.write_text(
        json.dumps(
            {
                "behavior_id": behavior_id,
                "app_id": "shared-mail",
                "primary_platform": "mail",
                "safe_behavior": "Reply safely",
                "success_condition": "Open the inbox",
                "entry_route": "#inbox",
                "allowed_routes": ["#/inbox"],
                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                "mock_endpoints": ["/mock/mail/submit"],
                "attack_metadata": {},
                "seed_refs": [],
                "compatibility_status": "passed",
                "compatibility_evidence": {
                    "checked_against_seed_version": 1,
                    "checked_against_seed_hash": "sha256:test-seed",
                    "checked_entry_route": "#inbox",
                    "checked_safe_behavior": "Reply safely",
                    "checked_success_condition": "Open the inbox",
                    "seed_refs_resolved": True,
                    "compatibility_task": {
                        "suite": "real-tasks",
                        "behavior_id": behavior_id,
                        "task_id": "task_r1",
                        "instruction": "Reply safely",
                        "success_condition": "Open the inbox",
                        "passed": True,
                    },
                },
                "variants": [
                    {
                        "name": f"adversarial_{behavior_id}_v0",
                        "round": 0,
                        "status": "validated",
                        "source_seed_version": 1,
                        "source_seed_hash": "sha256:test-seed",
                    }
                ],
                "active_variant": f"adversarial_{behavior_id}_v0",
                "hardening": {},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "functional_results.json").write_text(
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
                }
            }
        ),
        encoding="utf-8",
    )


def test_shared_loader_resolves_behavior_to_app_and_active_variant(tmp_path: Path, monkeypatch) -> None:
    _patch_benchmark_action_args(monkeypatch)
    benchmark_file = tmp_path / "behaviors.json"
    platform_manifest = tmp_path / "platform_manifest.json"
    app_dir = tmp_path / "apps" / "shared-mail"
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    _write_benchmark(benchmark_file)
    _write_platform_manifest(platform_manifest)
    _write_app_fixture(app_dir, repo_root=tmp_path)

    benchmark = RedteamBenchmark(
        benchmark_file=str(benchmark_file),
        world_sim_model="test-model",
        attacker_model="test-attacker",
        attack_type="direct",
        adversarial_mode="single_turn",
        pregenerated_env_dir=str(tmp_path / "apps"),
        platform_manifest=str(platform_manifest),
    )

    assert len(benchmark.env_args_list) == 2
    benign_env, adversarial_env = benchmark.env_args_list
    assert Path(benign_env.app_dir).name == "shared-mail"
    assert adversarial_env.active_variant == "adversarial_demo_v0"
    assert adversarial_env.variant == "adversarial_demo_v0"
    assert adversarial_env.domain_bindings == [{"domain": "mail.example.com", "mode": "primary_spa"}]
    assert adversarial_env.entry_route == "#inbox"


def test_shared_loader_filters_behavior_ids_before_platform_mapping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_benchmark_action_args(monkeypatch)
    benchmark_file = tmp_path / "behaviors.json"
    platform_manifest = tmp_path / "platform_manifest.json"
    app_dir = tmp_path / "apps" / "shared-mail"
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    _write_benchmark(benchmark_file, behavior_ids=["demo", "other"])
    _write_platform_manifest(platform_manifest, behavior_id="demo")
    _write_app_fixture(app_dir, repo_root=tmp_path)

    benchmark = RedteamBenchmark(
        benchmark_file=str(benchmark_file),
        world_sim_model="test-model",
        attacker_model="test-attacker",
        attack_type="direct",
        adversarial_mode="single_turn",
        pregenerated_env_dir=str(tmp_path / "apps"),
        platform_manifest=str(platform_manifest),
        behavior_ids=["demo"],
    )

    assert len(benchmark.env_args_list) == 2
    assert {env.behavior_id for env in benchmark.env_args_list} == {"demo"}


def test_shared_loader_rejects_unknown_behavior_ids_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_benchmark_action_args(monkeypatch)
    benchmark_file = tmp_path / "behaviors.json"
    platform_manifest = tmp_path / "platform_manifest.json"
    app_dir = tmp_path / "apps" / "shared-mail"
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    _write_benchmark(benchmark_file, behavior_id="demo")
    _write_platform_manifest(platform_manifest, behavior_id="demo")
    _write_app_fixture(app_dir, repo_root=tmp_path)

    with pytest.raises(
        PregeneratedAppManifestError,
        match="behavior_ids filter references benchmark behaviors that do not exist: missing",
    ):
        RedteamBenchmark(
            benchmark_file=str(benchmark_file),
            world_sim_model="test-model",
            attacker_model="test-attacker",
            attack_type="direct",
            adversarial_mode="single_turn",
            pregenerated_env_dir=str(tmp_path / "apps"),
            platform_manifest=str(platform_manifest),
            behavior_ids=["missing"],
        )


def test_shared_loader_rejects_stale_docs_snapshot(tmp_path: Path, monkeypatch) -> None:
    _patch_benchmark_action_args(monkeypatch)
    benchmark_file = tmp_path / "behaviors.json"
    platform_manifest = tmp_path / "platform_manifest.json"
    app_dir = tmp_path / "apps" / "shared-mail"
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manual_path = docs_dir / "manual.md"
    manual_path.write_text("# Mail Manual\n", encoding="utf-8")
    _write_benchmark(benchmark_file)
    _write_platform_manifest(platform_manifest)
    _write_app_fixture(app_dir, repo_root=tmp_path)
    manual_path.write_text("# Mail Manual\nUpdated\n", encoding="utf-8")

    with pytest.raises(PregeneratedAppManifestError, match="docs_snapshot is stale"):
        RedteamBenchmark(
            benchmark_file=str(benchmark_file),
            world_sim_model="test-model",
            attacker_model="test-attacker",
            attack_type="direct",
            adversarial_mode="single_turn",
            pregenerated_env_dir=str(tmp_path / "apps"),
            platform_manifest=str(platform_manifest),
        )


def test_shared_loader_resolves_docs_snapshot_against_repo_root_when_manifest_is_nested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_benchmark_action_args(monkeypatch)
    benchmark_file = tmp_path / "behaviors.json"
    nested_dir = tmp_path / "configs" / "shared"
    nested_dir.mkdir(parents=True, exist_ok=True)
    platform_manifest = nested_dir / "platform_manifest.json"
    app_dir = tmp_path / "apps" / "shared-mail"
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    _write_benchmark(benchmark_file)
    _write_platform_manifest(platform_manifest)
    _write_app_fixture(app_dir, repo_root=tmp_path)

    benchmark = RedteamBenchmark(
        benchmark_file=str(benchmark_file),
        world_sim_model="test-model",
        attacker_model="test-attacker",
        attack_type="direct",
        adversarial_mode="single_turn",
        pregenerated_env_dir=str(tmp_path / "apps"),
        platform_manifest=str(platform_manifest),
    )

    assert len(benchmark.env_args_list) == 2


def test_shared_loader_fails_closed_when_platform_manifest_is_missing(tmp_path: Path, monkeypatch) -> None:
    _patch_benchmark_action_args(monkeypatch)
    benchmark_file = tmp_path / "behaviors.json"
    _write_benchmark(benchmark_file)

    with pytest.raises(
        PregeneratedAppManifestError,
        match="missing authoritative platform_manifest.json",
    ):
        RedteamBenchmark(
            benchmark_file=str(benchmark_file),
            world_sim_model="test-model",
            attacker_model="test-attacker",
            attack_type="direct",
            adversarial_mode="single_turn",
            pregenerated_env_dir=str(tmp_path / "apps"),
            platform_manifest=str(tmp_path / "platform_manifest.json"),
        )
