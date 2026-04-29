from __future__ import annotations

from pathlib import Path

from scripts import readiness_audit


def test_token_audit_ignores_secret_related_identifiers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "script.py"
    path.write_text(
        "def _dump_verification_proxy_without_resolved_secret():\n    return 'not a secret value'\n"
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
