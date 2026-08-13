from __future__ import annotations

from pathlib import Path

from scripts import readiness_audit


def test_token_audit_ignores_secret_related_identifiers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "script.py"
    path.write_text(
        "\n".join(
            [
                "def _dump_verification_proxy_without_resolved_secret():",
                "    token = _resolve_verification_proxy_token(proxy_data)",
                "    _PAYLOAD_SUBSTITUTION_TOKEN = '__WORLDSIM_WITNESSES_JSON__'",
                "    return 'not a secret value'",
            ]
        )
    )

    assert readiness_audit._token_findings([str(path)]) == []


def test_token_audit_flags_structured_token_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "instances.json"
    path.write_text('{"token": "' + ("a" * 63) + '1"}\n')

    findings = readiness_audit._token_findings([str(path)])

    assert len(findings) == 1
    assert findings[0].kind == "token_field"


def test_token_audit_flags_access_token_with_underscore(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "instances.json"
    key = "access_" + "token"
    value = "ghp_" + "abcd1234abcd1234abcd1234abcd1234"
    path.write_text(f'{{"{key}": "{value}"}}\n')

    findings = readiness_audit._token_findings([str(path)])

    assert len(findings) == 1
    assert findings[0].kind == "token_field"


def test_token_audit_flags_letter_only_client_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "oauth.json"
    key = "client_" + "secret"
    value = "abcdefghijklmnopqrstuvwxyzABCDEF"
    path.write_text(f'{{"{key}": "{value}"}}\n')

    findings = readiness_audit._token_findings([str(path)])

    assert len(findings) == 1
    assert findings[0].kind == "token_field"


def test_token_audit_scans_fixture_paths_for_real_tokens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture_path = tmp_path / "tests" / "fixtures" / "leaked.json"
    fixture_path.parent.mkdir(parents=True)
    bearer = "Bearer " + "abcdefghijklmnopqrstuvwxyzABCDEF"
    fixture_path.write_text(f'{{"authorization": "{bearer}"}}\n')

    findings = readiness_audit._token_findings([str(fixture_path)])

    assert len(findings) == 1
    assert findings[0].path == str(fixture_path)
    assert findings[0].kind == "secret_value"


def test_token_audit_ignores_env_var_placeholders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.md"
    path.write_text(
        "\n".join(
            [
                '{"access_token": "${WORLDSIM_PROXY_TOKEN}"}',
                'authorization_token = "$WORLDSIM_PROXY_TOKEN"',
                'client_secret = "WORLDSIM_PROXY_TOKEN"',
                'token = "<redacted>"',
            ]
        )
    )

    assert readiness_audit._token_findings([str(path)]) == []


def test_token_audit_ignores_non_secret_token_references(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(
        "\n".join(
            [
                '{"token_endpoint": "/rest/V1/integration/admin/token"}',
                '{"token_source": "logs/phase_0d/gitlab/personal_access_token.txt"}',
                '{"required_tokens": "https://attacker.invalid/abc-a"}',
                "_collect_agent_context_secrets = _phase_2_output._collect_agent_context_secrets",
            ]
        )
    )

    assert readiness_audit._token_findings([str(path)]) == []


def test_token_audit_ignores_url_query_csrf_tokens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "history.json"
    path.write_text(
        '{"action": "http://postmill.test/log_out?token=abcdef1234567890abcdef1234567890"}\n'
    )

    assert readiness_audit._token_findings([str(path)]) == []


def test_token_audit_flags_documented_raw_token_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "handoff.md"
    path.write_text("Current proxy token is `" + ("b" * 63) + "1`.\n")

    findings = readiness_audit._token_findings([str(path)])

    assert len(findings) == 1
    assert findings[0].kind == "high_entropy"


def test_verify_fast_fails_on_token_findings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "verify_fast.sh").read_text()

    assert "--fail-on tracked-generated --fail-on tokens" in script
    assert "--fail-on legacy-imports" in script


def test_legacy_import_audit_flags_retired_phase_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "worldsim" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "import worldsim.phases.phase_2_injections",
                "from worldsim.phases.phase_4_adversarial import run",
                "from worldsim.phases import phase_2c_config",
            ]
        )
    )

    findings = readiness_audit._legacy_phase_import_findings(["worldsim/consumer.py"])

    assert [finding.module for finding in findings] == [
        "worldsim.phases.phase_2_injections",
        "worldsim.phases.phase_4_adversarial",
        "worldsim.phases.phase_2c_config",
    ]


def test_legacy_import_audit_flags_phase_2_api_compat_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "worldsim" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "from worldsim.phases.phase_2_injections_api import generate_phase_2a_plans_api\n"
    )

    findings = readiness_audit._legacy_phase_import_findings(["worldsim/consumer.py"])

    assert [finding.module for finding in findings] == [
        "worldsim.phases.phase_2_injections_api",
    ]


def test_legacy_import_audit_flags_retired_feature_facades(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "worldsim" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "from worldsim.phases import phase_1_generate_new_tasks_validation",
                "from worldsim.phases import phase_2_exposure_contract",
                "from worldsim.phases import phase_2_feasibility",
                "from worldsim.phases import phase_2_text_fill",
            ]
        )
    )

    findings = readiness_audit._legacy_phase_import_findings(["worldsim/consumer.py"])

    assert [finding.module for finding in findings] == [
        "worldsim.phases.phase_1_generate_new_tasks_validation",
        "worldsim.phases.phase_2_exposure_contract",
        "worldsim.phases.phase_2_feasibility",
        "worldsim.phases.phase_2_text_fill",
    ]


def test_legacy_import_audit_allows_cutover_tests(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "tests" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text("import warp_taskgen.phases.phase_2_injections\n")

    assert readiness_audit._legacy_phase_import_findings(["tests/consumer.py"]) == []


def test_legacy_import_audit_flags_relative_imports_inside_phases_package(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "worldsim" / "phases" / "shim.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "from . import phase_2_injections",
                "from .phase_4_adversarial import run",
                "from ..phases import phase_2c_config",
            ]
        )
    )

    findings = readiness_audit._legacy_phase_import_findings(["worldsim/phases/shim.py"])

    assert [finding.module for finding in findings] == [
        "worldsim.phases.phase_2_injections",
        "worldsim.phases.phase_4_adversarial",
        "worldsim.phases.phase_2c_config",
    ]


def test_legacy_import_audit_relative_import_resolution_drops_levels() -> None:
    assert (
        readiness_audit._resolve_relative_anchor("worldsim/phases/foo.py", 1) == "worldsim.phases"
    )
    assert readiness_audit._resolve_relative_anchor("worldsim/phases/foo.py", 2) == "worldsim"
    assert readiness_audit._resolve_relative_anchor("worldsim/phases/__init__.py", 1) == (
        "worldsim.phases"
    )
    # Walks above the package root -> None.
    assert readiness_audit._resolve_relative_anchor("worldsim/phases/foo.py", 5) is None


def test_active_facade_inventory_is_empty_after_namespace_cutover(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "worldsim" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "import worldsim.main",
                "from worldsim.phases.phase_2_text_fill import validate_text_post_hoc",
                "from worldsim.phases import phase_2_feasibility",
            ]
        )
    )

    findings = readiness_audit._active_facade_import_findings(["worldsim/consumer.py"])

    assert readiness_audit.ACTIVE_COMPAT_FACADE_MODULES == frozenset()
    assert findings == []


def test_active_facade_import_audit_is_advisory_for_tests(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    test_path = tmp_path / "tests" / "consumer.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("from warp_taskgen.phases import phase_2_text_fill\n")

    docs_path = tmp_path / "docs" / "consumer.py"
    docs_path.parent.mkdir(parents=True)
    docs_path.write_text("from warp_taskgen.phases import phase_2_text_fill\n")

    assert (
        readiness_audit._active_facade_import_findings(["tests/consumer.py", "docs/consumer.py"])
        == []
    )


def test_legacy_namespace_audit_flags_active_production_imports(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "packages" / "warp-taskgen" / "warp_taskgen" / "consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text("from worldsim.phase_4 import pvpo_capture\n")

    findings = readiness_audit._legacy_namespace_import_findings(
        ["packages/warp-taskgen/warp_taskgen/consumer.py"]
    )

    assert [finding.module for finding in findings] == ["worldsim.phase_4"]


def test_legacy_namespace_audit_has_no_adapter_or_test_allowlist(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "packages" / "warp-taskgen" / "worldsim" / "__init__.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("import worldsim.phase_4\n")
    test_path = tmp_path / "packages" / "warp-taskgen" / "tests" / "test_namespace_cutover.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("import worldsim.phase_4\n")

    findings = readiness_audit._legacy_namespace_import_findings(
        [
            "packages/warp-taskgen/worldsim/__init__.py",
            "packages/warp-taskgen/tests/test_namespace_cutover.py",
        ]
    )

    assert readiness_audit.LEGACY_NAMESPACE_ALLOWED_PREFIXES == ()
    assert [finding.module for finding in findings] == [
        "worldsim.phase_4",
        "worldsim.phase_4",
    ]


def test_legacy_namespace_audit_does_not_allowlist_active_tests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_path = tmp_path / "packages" / "warp-taskgen" / "tests" / "test_active.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("import worldsim.phase_4\n")

    findings = readiness_audit._legacy_namespace_import_findings(
        ["packages/warp-taskgen/tests/test_active.py"]
    )

    assert [finding.module for finding in findings] == ["worldsim.phase_4"]


def test_generated_artifact_detection_covers_r5_copy_outputs(monkeypatch) -> None:
    paths = [
        "instances.scale.json",
        "instances.scale.json.fragment",
        "instances.smoke.example.json",
        "instances.smoke.json",
        "instances.smoke.json.fragment",
        "scripts/docker-compose.scale.yml",
        "scripts/docker-compose.smoke.yml",
        "scripts/proxy_ports.conf",
    ]
    monkeypatch.setattr(readiness_audit, "_git_ls_files", lambda: paths)

    audit = readiness_audit.build_audit()

    assert audit.tracked_generated == sorted(
        path for path in paths if path != "instances.smoke.example.json"
    )


def test_large_file_audit_uses_550_line_threshold(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    small = tmp_path / "small.py"
    review = tmp_path / "review.py"
    urgent = tmp_path / "urgent.py"
    small.write_text("\n".join("x = 1" for _ in range(550)))
    review.write_text("\n".join("x = 1" for _ in range(551)))
    urgent.write_text("\n".join("x = 1" for _ in range(1201)))
    monkeypatch.setattr(
        readiness_audit,
        "_git_ls_files",
        lambda: ["small.py", "review.py", "urgent.py"],
    )

    audit = readiness_audit.build_audit()

    assert [item.path for item in audit.files_over_550_loc] == ["urgent.py", "review.py"]
    assert [item.path for item in audit.files_over_1200_loc] == ["urgent.py"]


def test_large_file_audit_records_true_exemptions(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "AgentLab" / "src" / "reference.py"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join("x = 1" for _ in range(1201)))
    monkeypatch.setattr(
        readiness_audit,
        "_git_ls_files",
        lambda: ["AgentLab/src/reference.py"],
    )

    audit = readiness_audit.build_audit()

    assert audit.files_over_550_loc == []
    assert audit.files_over_1200_loc == []
    assert audit.large_file_exemptions[0].path == "AgentLab/src/reference.py"
