from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentlab.benchmarks.redteam import app_pipeline, controller, variant_ops
from agentlab.benchmarks.redteam.data_validator import CheckResult


def _init_git_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True, capture_output=True)
    (repo_root / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True, capture_output=True)
    return repo_root


def _shared_behavior_spec() -> dict:
    simulation_config = {
        "pages": [
            {
                "id": "mail",
                "base_site_url": "https://mail.example.com",
                "subdomains": ["/mail/u/0/#inbox"],
                "details": {"/mail/u/0/#inbox": "Inbox"},
            }
        ],
        "start_page": "https://mail.example.com/mail/u/0/#inbox",
        "success_condition": "Open the inbox",
    }
    return {
        "id": "shared-mail",
        "app_id": "shared-mail",
        "platform": "mail",
        "docs_path": "apps/user-manuals/mail",
        "simulation_config": simulation_config,
        "mapped_behaviors": [
            {
                "id": "demo-a",
                "doc": "Review the inbox safely",
                "safe_behavior": "Review the inbox safely",
                "entry_route": "#/inbox",
                "allowed_routes": ["#/inbox"],
                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                "mock_endpoints": ["/mock/mail/submit"],
                "simulation_config": simulation_config,
            },
            {
                "id": "demo-b",
                "doc": "Open the latest thread",
                "safe_behavior": "Open the latest thread",
                "entry_route": "#/inbox",
                "allowed_routes": ["#/inbox"],
                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                "mock_endpoints": ["/mock/mail/submit"],
                "simulation_config": simulation_config,
            },
        ],
    }


def _passing_validation() -> dict:
    return {
        "passed": True,
        "checks": {
            "parse": CheckResult(check_name="parse", passed=True, duration_ms=1.0).to_dict(),
            "schema": CheckResult(check_name="schema", passed=True, duration_ms=1.0).to_dict(),
        },
        "errors": [],
    }


def _write_app_shell(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "js").mkdir(exist_ok=True)
    (app_dir / "css").mkdir(exist_ok=True)
    (app_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "js" / "data.js").write_text(
        "const SEED_DATA_VERSION = 1;\nconst CURRENT_USER = { id: 'u1' };\n",
        encoding="utf-8",
    )


def test_run_behavior_persists_docs_snapshot_and_seed_metadata(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    output_dir = repo_root / "apps"
    docs_dir = repo_root / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    spec = _shared_behavior_spec()

    monkeypatch.setattr(
        app_pipeline,
        "generate_app_scaffold",
        lambda **kwargs: (_write_app_shell(Path(kwargs["app_dir"])) or {"generated": True, "errors": []}),
    )

    def fake_generate_variants_result(**kwargs):
        app_dir = Path(kwargs["app_dir"])
        (app_dir / "benign").mkdir(parents=True, exist_ok=True)
        (app_dir / "benign" / "data.js").write_text(
            "const SEED_DATA_VERSION = 1;\nconst CURRENT_USER = { id: 'u1' };\n",
            encoding="utf-8",
        )
        for behavior_id in ("demo-a", "demo-b"):
            variant_dir = app_dir / f"adversarial_{behavior_id}_v0"
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "data.js").write_text(
                "const SEED_DATA_VERSION = 1;\n"
                "const CURRENT_USER = { id: 'u1' };\n"
                f"const NOTICE = '{behavior_id}';\n",
                encoding="utf-8",
            )
        return variant_ops.VariantGenerationResult(
            variants=["benign", "adversarial_demo-a_v0", "adversarial_demo-b_v0"],
            status="validated",
            validation={
                "benign": {"passed": True, "checks": {}},
                "adversarial_demo-a_v0": {"passed": True, "checks": {}},
                "adversarial_demo-b_v0": {"passed": True, "checks": {}},
            },
        )

    monkeypatch.setattr(variant_ops, "generate_variants_result", fake_generate_variants_result)
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.validation.validate_runtime",
        lambda app_dir: _passing_validation(),
    )

    result = controller.run_behavior(
        spec,
        output_dir,
        controller.ControllerConfig(generate_functional_tests=False),
    )

    manifest = json.loads((output_dir / "shared-mail" / "app_manifest.json").read_text(encoding="utf-8"))
    assert manifest["app_id"] == "shared-mail"
    assert manifest["behavior_ids"] == ["demo-a", "demo-b"]
    assert manifest["docs_snapshot"]["docs_path"] == "apps/user-manuals/mail"
    assert manifest["docs_snapshot"]["file_count"] == 1
    assert manifest["shared_seed_version"] == 1
    assert manifest["shared_seed_hash"].startswith("sha256:")
    assert "doc" not in manifest
    assert "target" not in manifest
    assert "attack_channels" not in manifest

    contract_a = json.loads((output_dir / "shared-mail" / "behaviors" / "demo-a.json").read_text(encoding="utf-8"))
    assert contract_a["active_variant"] == "adversarial_demo-a_v0"
    assert contract_a["compatibility_evidence"]["checked_against_seed_hash"] == manifest["shared_seed_hash"]
    assert contract_a["variants"][0]["append_only_vs_benign"] is True
    assert "Functional readiness generation was skipped" in result.errors[-1]


def test_resume_behavior_fails_closed_when_docs_snapshot_is_stale(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    output_dir = repo_root / "apps"
    docs_dir = repo_root / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manual_path = docs_dir / "manual.md"
    manual_path.write_text("# Mail Manual\n", encoding="utf-8")
    spec = _shared_behavior_spec()

    monkeypatch.setattr(
        app_pipeline,
        "generate_app_scaffold",
        lambda **kwargs: (_write_app_shell(Path(kwargs["app_dir"])) or {"generated": True, "errors": []}),
    )
    def fake_generate_variants_result(**kwargs):
        app_dir = Path(kwargs["app_dir"])
        (app_dir / "benign").mkdir(parents=True, exist_ok=True)
        (app_dir / "benign" / "data.js").write_text(
            "const SEED_DATA_VERSION = 1;\nconst CURRENT_USER = { id: 'u1' };\n",
            encoding="utf-8",
        )
        for behavior_id in ("demo-a", "demo-b"):
            variant_dir = app_dir / f"adversarial_{behavior_id}_v0"
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "data.js").write_text(
                "const SEED_DATA_VERSION = 1;\n"
                "const CURRENT_USER = { id: 'u1' };\n"
                f"const NOTICE = '{behavior_id}';\n",
                encoding="utf-8",
            )
        return variant_ops.VariantGenerationResult(
            variants=["benign", "adversarial_demo-a_v0", "adversarial_demo-b_v0"],
            status="validated",
            validation={},
        )

    monkeypatch.setattr(variant_ops, "generate_variants_result", fake_generate_variants_result)
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.validation.validate_runtime",
        lambda app_dir: _passing_validation(),
    )

    controller.run_behavior(
        spec,
        output_dir,
        controller.ControllerConfig(generate_functional_tests=False),
    )

    manual_path.write_text("# Mail Manual\nUpdated\n", encoding="utf-8")
    monkeypatch.setattr(
        app_pipeline,
        "generate_app_scaffold",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("resume should fail before generation")),
    )

    result = controller.resume_behavior(
        spec,
        output_dir,
        controller.ControllerConfig(generate_functional_tests=False),
    )

    assert any("docs_snapshot is stale" in error for error in result.errors)
    state = json.loads(Path(result.controller_state_path).read_text(encoding="utf-8"))
    assert state["stop_reason"] == "docs_snapshot_mismatch"


def test_run_behavior_rejects_shared_behavior_without_safe_behavior(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    output_dir = repo_root / "apps"
    docs_dir = repo_root / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Mail Manual\n", encoding="utf-8")
    spec = _shared_behavior_spec()
    spec["mapped_behaviors"][0]["safe_behavior"] = ""

    monkeypatch.setattr(
        app_pipeline,
        "generate_app_scaffold",
        lambda **kwargs: (_write_app_shell(Path(kwargs["app_dir"])) or {"generated": True, "errors": []}),
    )

    def fake_generate_variants_result(**kwargs):
        app_dir = Path(kwargs["app_dir"])
        (app_dir / "benign").mkdir(parents=True, exist_ok=True)
        (app_dir / "benign" / "data.js").write_text(
            "const SEED_DATA_VERSION = 1;\nconst CURRENT_USER = { id: 'u1' };\n",
            encoding="utf-8",
        )
        for behavior_id in ("demo-a", "demo-b"):
            variant_dir = app_dir / f"adversarial_{behavior_id}_v0"
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "data.js").write_text(
                "const SEED_DATA_VERSION = 1;\n"
                "const CURRENT_USER = { id: 'u1' };\n"
                f"const NOTICE = '{behavior_id}';\n",
                encoding="utf-8",
            )
        return variant_ops.VariantGenerationResult(
            variants=["benign", "adversarial_demo-a_v0", "adversarial_demo-b_v0"],
            status="validated",
            validation={},
        )

    monkeypatch.setattr(variant_ops, "generate_variants_result", fake_generate_variants_result)
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.validation.validate_runtime",
        lambda app_dir: _passing_validation(),
    )

    with pytest.raises(ValueError, match="demo-a: missing non-empty safe_behavior"):
        controller.run_behavior(
            spec,
            output_dir,
            controller.ControllerConfig(generate_functional_tests=False),
        )


def test_write_behavior_contracts_preserves_existing_lineage_state(tmp_path: Path) -> None:
    app_dir = tmp_path / "shared-mail"
    app_dir.mkdir(parents=True, exist_ok=True)
    spec = _shared_behavior_spec()["mapped_behaviors"][0]
    manifest = {
        "app_id": "shared-mail",
        "platform": "mail",
        "pages": [
            {
                "id": "mail",
                "base_site_url": "https://mail.example.com",
                "subdomains": ["/mail/u/0/#inbox"],
                "details": {"/mail/u/0/#inbox": "Inbox"},
            }
        ],
        "start_page": "https://mail.example.com/mail/u/0/#inbox",
        "shared_seed_version": 1,
        "shared_seed_hash": "sha256:test-seed",
    }

    original_payload = controller._behavior_contract_payload(
        behavior_spec=spec,
        app_manifest=manifest,
    )
    original_payload["variants"].append(
        {
            "name": "adversarial_demo-a_v1",
            "round": 1,
            "status": "validated",
            "base_variant": "adversarial_demo-a_v0",
            "derived_from": "adversarial_demo-a_v0/data.js",
            "source_seed_version": 1,
            "source_seed_hash": "sha256:test-seed",
            "append_only_vs_benign": True,
        }
    )
    original_payload["active_variant"] = "adversarial_demo-a_v1"
    original_payload["hardening"] = {
        "latest_round": 1,
        "latest_variant": "adversarial_demo-a_v1",
    }
    contract_path = app_dir / "behaviors" / "demo-a.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(original_payload, indent=2), encoding="utf-8")

    controller._write_behavior_contracts(app_dir, [spec], manifest)

    rewritten_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert rewritten_contract["active_variant"] == "adversarial_demo-a_v1"
    assert [item["name"] for item in rewritten_contract["variants"]] == [
        "adversarial_demo-a_v0",
        "adversarial_demo-a_v1",
    ]
    assert rewritten_contract["hardening"] == {
        "latest_round": 1,
        "latest_variant": "adversarial_demo-a_v1",
    }


def test_write_behavior_contracts_resets_lineage_when_behavior_surface_changes(tmp_path: Path) -> None:
    app_dir = tmp_path / "shared-mail"
    app_dir.mkdir(parents=True, exist_ok=True)
    spec = _shared_behavior_spec()["mapped_behaviors"][0]
    manifest = {
        "app_id": "shared-mail",
        "platform": "mail",
        "pages": [
            {
                "id": "mail",
                "base_site_url": "https://mail.example.com",
                "subdomains": ["/mail/u/0/#inbox"],
                "details": {"/mail/u/0/#inbox": "Inbox"},
            }
        ],
        "start_page": "https://mail.example.com/mail/u/0/#inbox",
        "shared_seed_version": 1,
        "shared_seed_hash": "sha256:test-seed",
    }

    original_payload = controller._behavior_contract_payload(
        behavior_spec=spec,
        app_manifest=manifest,
    )
    original_payload["variants"].append(
        {
            "name": "adversarial_demo-a_v1",
            "round": 1,
            "status": "validated",
            "base_variant": "adversarial_demo-a_v0",
            "derived_from": "adversarial_demo-a_v0/data.js",
            "source_seed_version": 1,
            "source_seed_hash": "sha256:test-seed",
            "append_only_vs_benign": True,
        }
    )
    original_payload["active_variant"] = "adversarial_demo-a_v1"
    original_payload["hardening"] = {
        "latest_round": 1,
        "latest_variant": "adversarial_demo-a_v1",
    }
    contract_path = app_dir / "behaviors" / "demo-a.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(original_payload, indent=2), encoding="utf-8")

    changed_spec = dict(spec)
    changed_spec["entry_route"] = "#/compose"
    changed_spec["allowed_routes"] = ["#/compose"]

    controller._write_behavior_contracts(app_dir, [changed_spec], manifest)

    rewritten_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert rewritten_contract["active_variant"] == "adversarial_demo-a_v0"
    assert [item["name"] for item in rewritten_contract["variants"]] == [
        "adversarial_demo-a_v0",
    ]
    assert rewritten_contract["hardening"] == {}
    assert rewritten_contract["compatibility_evidence"]["checked_entry_route"] == "/#/compose"


def test_write_behavior_contracts_honors_explicit_active_variant_override(tmp_path: Path) -> None:
    app_dir = tmp_path / "shared-mail"
    app_dir.mkdir(parents=True, exist_ok=True)
    spec = _shared_behavior_spec()["mapped_behaviors"][0]
    manifest = {
        "app_id": "shared-mail",
        "platform": "mail",
        "pages": [
            {
                "id": "mail",
                "base_site_url": "https://mail.example.com",
                "subdomains": ["/mail/u/0/#inbox"],
                "details": {"/mail/u/0/#inbox": "Inbox"},
            }
        ],
        "start_page": "https://mail.example.com/mail/u/0/#inbox",
        "shared_seed_version": 1,
        "shared_seed_hash": "sha256:test-seed",
    }

    original_payload = controller._behavior_contract_payload(
        behavior_spec=spec,
        app_manifest=manifest,
    )
    original_payload["variants"].append(
        {
            "name": "adversarial_demo-a_v1",
            "round": 1,
            "status": "validated",
            "base_variant": "adversarial_demo-a_v0",
            "derived_from": "adversarial_demo-a_v0/data.js",
            "source_seed_version": 1,
            "source_seed_hash": "sha256:test-seed",
            "append_only_vs_benign": True,
        }
    )
    original_payload["active_variant"] = "adversarial_demo-a_v1"
    original_payload["hardening"] = {
        "latest_round": 1,
        "latest_variant": "adversarial_demo-a_v1",
    }
    contract_path = app_dir / "behaviors" / "demo-a.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(original_payload, indent=2), encoding="utf-8")

    override_spec = dict(spec)
    override_spec["active_variant"] = "adversarial_demo-a_v0"

    controller._write_behavior_contracts(app_dir, [override_spec], manifest)

    rewritten_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert rewritten_contract["active_variant"] == "adversarial_demo-a_v0"
    assert [item["name"] for item in rewritten_contract["variants"]] == [
        "adversarial_demo-a_v0",
        "adversarial_demo-a_v1",
    ]
    assert rewritten_contract["hardening"] == {
        "latest_round": 1,
        "latest_variant": "adversarial_demo-a_v1",
    }


def test_run_behavior_rejects_unsafe_app_id_before_workspace_creation(tmp_path: Path) -> None:
    repo_root = _init_git_repo(tmp_path)
    output_dir = repo_root / "apps"
    spec = _shared_behavior_spec()
    spec["app_id"] = "../escape"

    with pytest.raises(ValueError, match="app_id must be a single path component"):
        controller.run_behavior(
            spec,
            output_dir,
            controller.ControllerConfig(generate_functional_tests=False),
        )

    assert not (output_dir / "escape").exists()


def test_run_behavior_rejects_unsafe_mapped_behavior_id_before_generation(tmp_path: Path, monkeypatch) -> None:
    repo_root = _init_git_repo(tmp_path)
    output_dir = repo_root / "apps"
    spec = _shared_behavior_spec()
    spec["mapped_behaviors"][0]["id"] = "nested/demo-a"

    monkeypatch.setattr(
        app_pipeline,
        "generate_app_scaffold",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("generation should not run")),
    )

    with pytest.raises(ValueError, match="behavior_id must be a single path component"):
        controller.run_behavior(
            spec,
            output_dir,
            controller.ControllerConfig(generate_functional_tests=False),
        )

    assert not (output_dir / "shared-mail" / "behaviors" / "nested").exists()
    assert not (output_dir / "shared-mail" / "adversarial_nested").exists()
