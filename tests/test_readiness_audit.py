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


def test_generated_artifact_detection_covers_r5_copy_outputs(monkeypatch) -> None:
    paths = [
        "instances.scale.json",
        "instances.scale.json.fragment",
        "instances.smoke.json",
        "instances.smoke.json.fragment",
        "scripts/docker-compose.scale.yml",
        "scripts/docker-compose.smoke.yml",
        "scripts/proxy_ports.conf",
    ]
    monkeypatch.setattr(readiness_audit, "_git_ls_files", lambda: paths)

    audit = readiness_audit.build_audit()

    assert audit.tracked_generated == sorted(paths)


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
