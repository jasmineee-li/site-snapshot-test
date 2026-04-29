from __future__ import annotations

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


def test_token_audit_flags_documented_raw_token_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "handoff.md"
    path.write_text("Current proxy token is `" + ("b" * 63) + "1`.\n")

    findings = readiness_audit._token_findings([str(path)])

    assert len(findings) == 1
    assert findings[0].kind == "high_entropy"
