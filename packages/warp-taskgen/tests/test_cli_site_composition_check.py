from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from warp_taskgen.cli import site_composition_check
from warp_taskgen.cli._impl import main


class _FakeReport:
    def __init__(self, static_status: str) -> None:
        self.static_status = static_status

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "warp-site-composition-check-v1",
            "scope": "static_site_composition_only",
            "site": "gitlab",
            "benchmark": "webarena_verified",
            "use_case": "phase_2_feasibility",
            "carrier": None,
            "action_kind": None,
            "static_status": self.static_status,
            "site_composition_digest": None
            if self.static_status == "invalid"
            else "sha256:" + "0" * 64,
            "readiness_status": "blocked",
            "readiness_blockers": [
                "active_policy_not_checked",
                "live_evidence_not_checked",
            ],
            "active_policy_checked": False,
            "live_evidence_checked": False,
            "findings": [
                {
                    "capability": "static_closure",
                    "state": self.static_status,
                    "outcome": "pass" if self.static_status == "complete" else "failure",
                    "code": f"static_closure.{self.static_status}",
                    "detail": "deterministic finding",
                    "provenance": [],
                    "dependencies": [],
                }
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    site_composition_check.add_site_composition_parser(subparsers)
    return parser


def test_canonical_parser_requires_exact_site_benchmark_and_use_case() -> None:
    args = _parser().parse_args(
        [
            "site",
            "composition",
            "check",
            "gitlab",
            "--benchmark",
            "webarena_verified",
            "--use-case",
            "phase_2_feasibility",
            "--carrier",
            "issue.description",
            "--action-kind",
            "answer_opposite_binary_label",
        ]
    )

    assert args.command == "site"
    assert args.site_command == "composition"
    assert args.composition_command == "check"
    assert args.site == "gitlab"
    assert args.benchmark == "webarena_verified"
    assert args.use_case == "phase_2_feasibility"
    assert args.carrier == "issue.description"
    assert args.action_kind == "answer_opposite_binary_label"


def test_site_doctor_cli_is_a_direct_compatibility_alias() -> None:
    args = _parser().parse_args(
        [
            "site",
            "doctor",
            "gitlab",
            "--benchmark",
            "webarena_verified",
            "--use-case",
            "phase_2_feasibility",
        ]
    )

    assert args.site_command == "doctor"
    assert not hasattr(args, "composition_command")


@pytest.mark.parametrize(
    ("static_status", "expected_exit"),
    [("complete", 0), ("incomplete", 1), ("invalid", 2)],
)
def test_dispatch_exit_code_uses_static_status_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    static_status: str,
    expected_exit: int,
) -> None:
    observed: dict[str, object] = {}
    report = _FakeReport(static_status)

    def compile_default(**kwargs: object) -> _FakeReport:
        observed.update(kwargs)
        return report

    monkeypatch.setattr(site_composition_check, "_compile_default", compile_default)
    result = site_composition_check.dispatch_site_composition(
        SimpleNamespace(
            site_command="composition",
            composition_command="check",
            site="gitlab",
            benchmark="webarena_verified",
            use_case="phase_2_feasibility",
            carrier="issue.description",
            action_kind="answer_opposite_binary_label",
            json=False,
        )
    )

    assert result == expected_exit
    assert observed == {
        "site": "gitlab",
        "benchmark": "webarena_verified",
        "use_case": "phase_2_feasibility",
        "carrier": "issue.description",
        "action_kind": "answer_opposite_binary_label",
    }
    output = capsys.readouterr().out
    assert f"Static Site Composition status: {static_status}" in output
    assert "active policy and live evidence not checked" in output
    assert "Operational readiness: blocked" in output


def test_doctor_alias_dispatches_the_same_canonical_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _FakeReport("complete")
    monkeypatch.setattr(site_composition_check, "_compile_default", lambda **_: report)

    result = site_composition_check.dispatch_site_composition(
        SimpleNamespace(
            site_command="doctor",
            site="gitlab",
            benchmark="webarena_verified",
            use_case="phase_2_feasibility",
            carrier=None,
            action_kind=None,
            json=True,
        )
    )

    assert result == 0


def test_missing_packaged_composition_returns_structured_invalid_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_resource(**_: object) -> _FakeReport:
        raise ModuleNotFoundError("required Site Composition resource is absent")

    monkeypatch.setattr(site_composition_check, "_compile_default", missing_resource)
    result = site_composition_check.dispatch_site_composition(
        SimpleNamespace(
            site_command="composition",
            composition_command="check",
            site="classifieds",
            benchmark="visualwebarena",
            use_case="public_reply",
            carrier="listing_reply.body",
            action_kind="answer_opposite_binary_label",
            json=True,
        )
    )

    assert result == 2
    report = json.loads(capsys.readouterr().out)
    assert report["static_status"] == "invalid"
    assert report["readiness_status"] == "blocked"
    assert report["error"] == "ModuleNotFoundError"


def test_json_output_is_the_report_projection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _FakeReport("complete")
    monkeypatch.setattr(site_composition_check, "_compile_default", lambda **_: report)

    result = site_composition_check.dispatch_site_composition(
        SimpleNamespace(
            site_command="composition",
            composition_command="check",
            site="gitlab",
            benchmark="webarena_verified",
            use_case="phase_2_feasibility",
            carrier=None,
            action_kind=None,
            json=True,
        )
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == report.to_dict()


def test_canonical_cli_dispatches_default_static_diagnostic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "site",
            "composition",
            "check",
            "gitlab",
            "--benchmark",
            "webarena_verified",
            "--use-case",
            "phase_2_feasibility",
            "--json",
        ]
    )

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["static_status"] == "complete"
    assert report["scope"] == "static_site_composition_only"
    assert report["readiness_status"] == "blocked"
    assert report["active_policy_checked"] is False
    assert report["live_evidence_checked"] is False


def test_invalid_identity_returns_two_without_echoing_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "site",
            "composition",
            "check",
            "https://private.invalid/secret",
            "--benchmark",
            "webarena_verified",
            "--use-case",
            "phase_2_feasibility",
            "--json",
        ]
    )

    assert result == 2
    output = capsys.readouterr().out
    assert "private.invalid" not in output
    assert json.loads(output)["static_status"] == "invalid"


def test_unknown_benchmark_does_not_echo_secret_like_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "site",
            "composition",
            "check",
            "gitlab",
            "--benchmark",
            "SECRET_TOKEN_123",
            "--use-case",
            "phase_2_feasibility",
            "--json",
        ]
    )

    assert result == 2
    output = capsys.readouterr().out
    assert "SECRET_TOKEN_123" not in output
    assert "secret_token_123" not in output
