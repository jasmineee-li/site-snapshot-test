from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlab.benchmarks.redteam import generate_apps
from agentlab.benchmarks.redteam.app_artifacts import (
    APP_MANIFEST_CONTRACT_VERSION,
    compute_docs_snapshot,
)
from agentlab.benchmarks.redteam.generate_apps import _group_behaviors_by_app
from agentlab.benchmarks.redteam.phase_ids import PHASE_4B, PHASE_5


def test_group_behaviors_by_app_collapses_multiple_behaviors_onto_shared_app(tmp_path: Path) -> None:
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "primary_platform": "mail",
                                "entry_route": "#inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [
                                    {"domain": "mail.example.com", "mode": "primary_spa"}
                                ],
                            },
                            {
                                "behavior_id": "demo-b",
                                "primary_platform": "mail",
                                "entry_route": "#inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [
                                    {"domain": "mail.example.com", "mode": "primary_spa"}
                                ],
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    grouped = _group_behaviors_by_app(
        behaviors=[
            {"id": "demo-a", "doc": "A"},
            {"id": "demo-b", "doc": "B"},
        ],
        platform_manifest_path=platform_manifest,
    )

    assert len(grouped) == 1
    assert grouped[0]["app_id"] == "shared-mail"
    assert [item["id"] for item in grouped[0]["mapped_behaviors"]] == ["demo-a", "demo-b"]


def test_group_behaviors_by_app_rejects_duplicate_behavior_ids(tmp_path: Path) -> None:
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate benchmark behavior_id 'demo-a'"):
        _group_behaviors_by_app(
            behaviors=[
                {"id": "demo-a", "doc": "A"},
                {"id": "demo-a", "doc": "B"},
            ],
            platform_manifest_path=platform_manifest,
        )


def test_generate_apps_root_manifest_counts_apps_separately_from_behaviors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text(
        json.dumps([{"id": "demo-a"}, {"id": "demo-b"}], indent=2),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [
                                    {"domain": "mail.example.com", "mode": "primary_spa"}
                                ],
                            },
                            {
                                "behavior_id": "demo-b",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [
                                    {"domain": "mail.example.com", "mode": "primary_spa"}
                                ],
                            },
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        generate_apps,
        "_generate_single",
        lambda **kwargs: {
            "contract_version": APP_MANIFEST_CONTRACT_VERSION,
            "app_id": kwargs["behavior"]["app_id"],
            "platform": "mail",
            "docs_path": "apps/user-manuals/mail",
            "docs_snapshot": {"docs_path": "apps/user-manuals/mail", "corpus_sha256": "sha256:manuals"},
            "shared_seed_version": 1,
            "shared_seed_hash": "sha256:test",
            "behavior_ids": ["demo-a", "demo-b"],
            "pages": [
                {
                    "id": "mail",
                    "base_site_url": "https://mail.example.com",
                    "subdomains": ["/mail/u/0/#inbox"],
                    "details": {"/mail/u/0/#inbox": "Inbox"},
                }
            ],
            "start_page": "https://mail.example.com/mail/u/0/#inbox",
            "execution_backend": {"selected_backend": "modal"},
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
                },
                "real_evaluation": {
                    "ran": True,
                    "backend": "agent-browser",
                    "agent_config": "agent",
                    "pass_rate": 1.0,
                    "total": 1,
                    "passed": 1,
                },
                "quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 0.8},
                "real_quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 1.0},
            },
        },
    )

    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    output_dir = tmp_path / "apps"
    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--platform-manifest",
            str(platform_manifest),
        ]
    )

    root_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["n_total"] == 1
    assert root_manifest["n_total_apps"] == 1
    assert root_manifest["n_total_behaviors"] == 2


def _write_successful_shared_app(
    app_dir: Path,
    *,
    include_behavior_b: bool = True,
    include_variant_b: bool = True,
) -> None:
    (app_dir / "js").mkdir(parents=True, exist_ok=True)
    (app_dir / "css").mkdir(exist_ok=True)
    (app_dir / "benign").mkdir(exist_ok=True)
    (app_dir / "adversarial_demo-a_v0").mkdir(exist_ok=True)
    if include_variant_b:
        (app_dir / "adversarial_demo-b_v0").mkdir(exist_ok=True)
    (app_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "benign" / "data.js").write_text("const MODE = 'benign';\n", encoding="utf-8")
    (app_dir / "adversarial_demo-a_v0" / "data.js").write_text(
        "const MODE = 'adversarial-a';\n",
        encoding="utf-8",
    )
    if include_variant_b:
        (app_dir / "adversarial_demo-b_v0" / "data.js").write_text(
            "const MODE = 'adversarial-b';\n",
            encoding="utf-8",
        )
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "app_id": "shared-mail",
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
                "behavior_ids": ["demo-a", "demo-b"],
                "pages": [
                    {
                        "id": "mail",
                        "base_site_url": "https://mail.example.com",
                        "subdomains": ["/mail/u/0/#inbox"],
                        "details": {"/mail/u/0/#inbox": "Inbox"},
                    }
                ],
                "start_page": "https://mail.example.com/mail/u/0/#inbox",
                "execution_backend": {"selected_backend": "modal"},
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
                        "error": None,
                    },
                    "quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 0.8},
                    "real_quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 1.0},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    behavior_dir = app_dir / "behaviors"
    behavior_dir.mkdir(exist_ok=True)
    (behavior_dir / "demo-a.json").write_text(
        json.dumps(
            {
                "behavior_id": "demo-a",
                "app_id": "shared-mail",
                "primary_platform": "mail",
                "safe_behavior": "A",
                "success_condition": "Open inbox",
                "entry_route": "#/inbox",
                "allowed_routes": ["#/inbox"],
                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                "compatibility_status": "passed",
                "compatibility_evidence": {
                    "checked_against_seed_version": 1,
                    "checked_against_seed_hash": "sha256:test-seed",
                },
                "variants": [{"name": "adversarial_demo-a_v0", "round": 0, "status": "validated"}],
                "active_variant": "adversarial_demo-a_v0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if include_behavior_b:
        (behavior_dir / "demo-b.json").write_text(
            json.dumps(
                {
                    "behavior_id": "demo-b",
                    "app_id": "shared-mail",
                    "primary_platform": "mail",
                    "safe_behavior": "B",
                    "success_condition": "Open inbox",
                    "entry_route": "#/inbox",
                    "allowed_routes": ["#/inbox"],
                    "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                    "compatibility_status": "passed",
                    "compatibility_evidence": {
                        "checked_against_seed_version": 1,
                        "checked_against_seed_hash": "sha256:test-seed",
                    },
                    "variants": [{"name": "adversarial_demo-b_v0", "round": 0, "status": "validated"}],
                    "active_variant": "adversarial_demo-b_v0",
                },
                indent=2,
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_generate_apps_supports_behavior_subset_selection_in_shared_app_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text(
        json.dumps([{"id": "demo-a", "doc": "A"}, {"id": "demo-b", "doc": "B"}], indent=2),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                            {
                                "behavior_id": "demo-b",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    captured: list[dict] = []
    monkeypatch.setattr(
        generate_apps,
        "_generate_single",
        lambda **kwargs: (
            captured.append(kwargs["behavior"]),
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "app_id": kwargs["behavior"]["app_id"],
                "platform": "mail",
                "docs_path": "apps/user-manuals/mail",
                "docs_snapshot": {
                    "docs_path": "apps/user-manuals/mail",
                    "source_type": "directory",
                    "corpus_sha256": "sha256:manuals",
                    "files": [{"path": "mail.md", "sha256": "abc123"}],
                },
                "shared_seed_version": 1,
                "shared_seed_hash": "sha256:test",
                "behavior_ids": ["demo-a", "demo-b"],
                "pages": [
                    {
                        "id": "mail",
                        "base_site_url": "https://mail.example.com",
                        "subdomains": ["/mail/u/0/#inbox"],
                        "details": {"/mail/u/0/#inbox": "Inbox"},
                    }
                ],
                "start_page": "https://mail.example.com/mail/u/0/#inbox",
                "execution_backend": {"selected_backend": "modal"},
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
                    "function_evaluation": {"ran": True, "backend": "agent-browser", "pass_rate": 1.0, "total": 1, "passed": 1},
                    "real_evaluation": {"ran": True, "backend": "agent-browser", "pass_rate": 1.0, "total": 1, "passed": 1},
                    "quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 0.8},
                    "real_quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 1.0},
                },
            },
        )[-1],
    )

    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    output_dir = tmp_path / "apps"
    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--platform-manifest",
            str(platform_manifest),
            "--behaviors",
            "demo-a",
        ]
    )

    assert len(captured) == 1
    assert captured[0]["app_id"] == "shared-mail"
    assert [item["id"] for item in captured[0]["mapped_behaviors"]] == ["demo-a", "demo-b"]
    root_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["n_total_apps"] == 1
    assert root_manifest["n_total_behaviors"] == 2


def test_generate_apps_filtered_shared_subset_ignores_unrelated_invalid_manifest_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text(
        json.dumps(
            [
                {"id": "demo-a", "doc": "A"},
                {"id": "demo-b", "doc": "B"},
                {"id": "other-demo", "doc": "Other"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                            {
                                "behavior_id": "demo-b",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                        ],
                    },
                    {
                        "app_id": "broken-app",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/does-not-matter",
                        "behaviors": [
                            {
                                "behavior_id": "other-demo",
                                "entry_route": "#/broken",
                                "allowed_routes": [],
                                "domain_bindings": [{"domain": "broken.example.com", "mode": "primary_spa"}],
                            }
                        ],
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")

    captured: list[dict] = []
    monkeypatch.setattr(
        generate_apps,
        "_generate_single",
        lambda **kwargs: (
            captured.append(kwargs["behavior"]),
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "app_id": kwargs["behavior"]["app_id"],
                "platform": "mail",
                "docs_path": "apps/user-manuals/mail",
                "docs_snapshot": {
                    "docs_path": "apps/user-manuals/mail",
                    "source_type": "directory",
                    "corpus_sha256": "sha256:manuals",
                    "files": [{"path": "mail.md", "sha256": "abc123"}],
                },
                "shared_seed_version": 1,
                "shared_seed_hash": "sha256:test",
                "behavior_ids": ["demo-a", "demo-b"],
                "pages": [
                    {
                        "id": "mail",
                        "base_site_url": "https://mail.example.com",
                        "subdomains": ["/mail/u/0/#inbox"],
                        "details": {"/mail/u/0/#inbox": "Inbox"},
                    }
                ],
                "start_page": "https://mail.example.com/mail/u/0/#inbox",
                "execution_backend": {"selected_backend": "modal"},
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
                    "function_evaluation": {"ran": True, "backend": "agent-browser", "pass_rate": 1.0, "total": 1, "passed": 1},
                    "real_evaluation": {"ran": True, "backend": "agent-browser", "pass_rate": 1.0, "total": 1, "passed": 1},
                    "quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 0.8},
                    "real_quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 1.0},
                },
            },
        )[-1],
    )

    output_dir = tmp_path / "apps-out"
    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--platform-manifest",
            str(platform_manifest),
            "--behaviors",
            "demo-a",
        ]
    )

    assert len(captured) == 1
    assert captured[0]["app_id"] == "shared-mail"
    assert [item["id"] for item in captured[0]["mapped_behaviors"]] == ["demo-a", "demo-b"]


def test_generate_apps_shared_rejects_missing_manual_corpus_for_selected_subset(tmp_path: Path) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text(
        json.dumps([{"id": "demo-a", "doc": "A"}], indent=2),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            }
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        generate_apps.main(
            [
                "--benchmark-file",
                str(benchmark_file),
                "--output-dir",
                str(tmp_path / "apps"),
                "--platform-manifest",
                str(platform_manifest),
                "--behaviors",
                "demo-a",
            ]
        )


def test_generate_apps_resume_retries_incomplete_shared_apps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text(
        json.dumps([{"id": "demo-a", "doc": "A"}, {"id": "demo-b", "doc": "B"}], indent=2),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                            {
                                "behavior_id": "demo-b",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "apps"
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    _write_successful_shared_app(output_dir / "shared-mail", include_behavior_b=False)

    calls: list[str] = []
    monkeypatch.setattr(
        generate_apps,
        "_generate_single",
        lambda **kwargs: (
            calls.append(kwargs["behavior"]["app_id"]),
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "app_id": kwargs["behavior"]["app_id"],
                "platform": "mail",
                "docs_path": "apps/user-manuals/mail",
                "docs_snapshot": {
                    "docs_path": "apps/user-manuals/mail",
                    "source_type": "directory",
                    "corpus_sha256": "sha256:manuals",
                    "files": [{"path": "mail.md", "sha256": "abc123"}],
                },
                "shared_seed_version": 1,
                "shared_seed_hash": "sha256:test",
                "behavior_ids": ["demo-a", "demo-b"],
                "pages": [
                    {
                        "id": "mail",
                        "base_site_url": "https://mail.example.com",
                        "subdomains": ["/mail/u/0/#inbox"],
                        "details": {"/mail/u/0/#inbox": "Inbox"},
                    }
                ],
                "start_page": "https://mail.example.com/mail/u/0/#inbox",
                "execution_backend": {"selected_backend": "modal"},
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
                    "function_evaluation": {"ran": True, "backend": "agent-browser", "pass_rate": 1.0, "total": 1, "passed": 1},
                    "real_evaluation": {"ran": True, "backend": "agent-browser", "pass_rate": 1.0, "total": 1, "passed": 1},
                    "quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 0.8},
                    "real_quality_gate": {"passed": True, "pass_rate": 1.0, "threshold": 1.0},
                },
            },
        )[-1],
    )

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--platform-manifest",
            str(platform_manifest),
            "--resume",
        ]
    )

    assert calls == ["shared-mail"]


def test_generate_apps_shared_rejects_duplicate_behavior_ids_via_cli(
    tmp_path: Path,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text(
        json.dumps([{"id": "demo-a"}, {"id": "demo-a"}], indent=2),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            }
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        generate_apps.main(
            [
                "--benchmark-file",
                str(benchmark_file),
                "--output-dir",
                str(tmp_path / "apps"),
                "--platform-manifest",
                str(platform_manifest),
            ]
        )


def test_generate_apps_resume_retries_shared_app_when_docs_snapshot_is_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "behaviors.json"
    benchmark_file.write_text(
        json.dumps([{"id": "demo-a", "doc": "A"}, {"id": "demo-b", "doc": "B"}], indent=2),
        encoding="utf-8",
    )
    platform_manifest = tmp_path / "platform_manifest.json"
    platform_manifest.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                            {
                                "behavior_id": "demo-b",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/inbox"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            },
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manual_path = docs_dir / "mail.md"
    manual_path.write_text("# Mail Manual\n", encoding="utf-8")
    output_dir = tmp_path / "apps"
    shared_app_dir = output_dir / "shared-mail"
    _write_successful_shared_app(shared_app_dir)
    manifest = json.loads((shared_app_dir / "app_manifest.json").read_text(encoding="utf-8"))
    manifest["docs_snapshot"] = compute_docs_snapshot(
        {"docs_path": "apps/user-manuals/mail"},
        repo_root_path=tmp_path,
    )
    (shared_app_dir / "app_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    manual_path.write_text("# Mail Manual\nUpdated\n", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(
        generate_apps,
        "_generate_single",
        lambda **kwargs: (
            calls.append(kwargs["behavior"]["app_id"]),
            manifest,
        )[-1],
    )

    generate_apps.main(
        [
            "--benchmark-file",
            str(benchmark_file),
            "--output-dir",
            str(output_dir),
            "--platform-manifest",
            str(platform_manifest),
            "--resume",
        ]
    )

    assert calls == ["shared-mail"]
