from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "provision_tier3_gitlab_fixtures.py"
    spec = importlib.util.spec_from_file_location("provision_tier3_gitlab_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document() -> dict[str, Any]:
    return {
        "benchmark_name": "WebArena Verified",
        "instances": [
            {"site_name": "gitlab", "site_url": "http://gitlab-1.test", "replica_index": 0},
            {"site_name": "reddit", "site_url": "http://reddit.test"},
            {"site_name": "gitlab", "site_url": "http://gitlab-2.test", "replica_index": 1},
        ],
    }


def test_provision_document_verifies_every_gitlab_replica(monkeypatch) -> None:
    provision = _load_module()
    ensured: list[str] = []

    def fake_ensure(instance, *, project_paths, create_missing):
        ensured.append(instance["site_url"])
        assert project_paths == ["byteblaze/worldsim-tier3-fixture-01"]
        assert create_missing is True

    def fake_verify(instance):
        assert instance["tier3_fixtures"]["gitlab"]["repository_content"]["scope"] == (
            "worldsim_disposable"
        )
        return [object()], {"status": "ready", "verified_projects": 1}

    monkeypatch.setattr(provision, "_ensure_projects_on_instance", fake_ensure)
    monkeypatch.setattr(provision, "verify_gitlab_repository_fixture_pool", fake_verify)

    document = _document()
    report = provision.provision_document(
        document,
        project_paths=["byteblaze/worldsim-tier3-fixture-01"],
    )

    assert report["status"] == "ready"
    assert report["gitlab_replicas"] == 2
    assert ensured == ["http://gitlab-1.test", "http://gitlab-2.test"]
    assert document["tier3_fixtures"] == {
        "gitlab": {
            "repository_content": {
                "scope": "worldsim_disposable",
                "file_path_prefix": "worldsim-fixtures",
                "projects": [{"project_path": "byteblaze/worldsim-tier3-fixture-01"}],
            }
        }
    }


def test_provision_document_can_declare_verified_delete_fixture(monkeypatch) -> None:
    provision = _load_module()
    ensured: list[list[str]] = []

    def fake_ensure(instance, *, project_paths, create_missing):
        ensured.append(list(project_paths))
        assert create_missing is True

    def fake_verify_repo(instance):
        assert "repository_content" in instance["tier3_fixtures"]["gitlab"]
        return [object()], {"status": "ready", "verified_projects": 1}

    def fake_verify_delete(instance):
        delete_config = instance["tier3_fixtures"]["gitlab"]["delete_project"]
        assert delete_config["scope"] == "worldsim_disposable"
        assert delete_config["cleanup_strategy"] == "fixture_reset"
        assert delete_config["reset_verified"] is True
        return [object()], {"status": "ready", "verified_projects": 1}

    monkeypatch.setattr(provision, "_ensure_projects_on_instance", fake_ensure)
    monkeypatch.setattr(provision, "verify_gitlab_repository_fixture_pool", fake_verify_repo)
    monkeypatch.setattr(provision, "verify_gitlab_delete_project_fixture_pool", fake_verify_delete)

    document = _document()
    report = provision.provision_document(
        document,
        project_paths=["byteblaze/worldsim-tier3-fixture-01"],
        delete_project_paths=["byteblaze/worldsim-tier3-delete-fixture-01"],
        delete_cleanup_strategy="fixture_reset",
        delete_reset_verified=True,
    )

    assert report["status"] == "ready"
    assert report["delete_project_paths"] == ["byteblaze/worldsim-tier3-delete-fixture-01"]
    assert ensured == [
        [
            "byteblaze/worldsim-tier3-fixture-01",
            "byteblaze/worldsim-tier3-delete-fixture-01",
        ],
        [
            "byteblaze/worldsim-tier3-fixture-01",
            "byteblaze/worldsim-tier3-delete-fixture-01",
        ],
    ]
    assert document["tier3_fixtures"]["gitlab"]["delete_project"] == {
        "scope": "worldsim_disposable",
        "cleanup_strategy": "fixture_reset",
        "reset_verified": True,
        "projects": [{"project_path": "byteblaze/worldsim-tier3-delete-fixture-01"}],
    }


def test_cli_rejects_delete_fixture_without_reset_proof(tmp_path) -> None:
    provision = _load_module()
    path = tmp_path / "instances.scale.json"
    path.write_text('{"instances": []}', encoding="utf-8")

    rc = provision.main(
        [
            "--instances",
            str(path),
            "--include-delete-project",
            "--dry-run",
        ]
    )

    assert rc == 1


def test_provision_document_fails_closed_when_any_replica_cannot_verify(monkeypatch) -> None:
    provision = _load_module()
    calls = 0

    def fake_ensure(instance, *, project_paths, create_missing):
        return None

    def fake_verify(instance):
        nonlocal calls
        calls += 1
        if calls == 2:
            return [], {"status": "failed", "reason": "no_verified_fixture_projects"}
        return [object()], {"status": "ready", "verified_projects": 1}

    monkeypatch.setattr(provision, "_ensure_projects_on_instance", fake_ensure)
    monkeypatch.setattr(provision, "verify_gitlab_repository_fixture_pool", fake_verify)

    document = _document()
    try:
        provision.provision_document(
            document,
            project_paths=["byteblaze/worldsim-tier3-fixture-01"],
        )
    except provision.ProvisionError as exc:
        assert "fixture canary verification failed" in str(exc)
    else:
        raise AssertionError("expected provisioning to fail closed")

    assert "tier3_fixtures" not in document
