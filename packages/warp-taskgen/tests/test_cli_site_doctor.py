from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from warp_taskgen.cli import site_doctor
from warp_taskgen.cli._impl import build_parser, main


class _FakeReport:
    def __init__(self, static_status: str) -> None:
        self.static_status = static_status
        self.status = "blocked"

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": "webarena_verified",
            "site": "gitlab",
            "use_case": "read_only",
            "static_status": self.static_status,
            "status": self.status,
            "findings": [
                {
                    "category": "static_readiness",
                    "status": self.static_status,
                    "message": "deterministic finding",
                }
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    site_doctor.add_site_doctor_parser(subparsers)
    return parser


def _install_fake_composition(
    monkeypatch: pytest.MonkeyPatch,
    report: _FakeReport,
    observed: dict[str, object] | None = None,
) -> None:
    observed = observed if observed is not None else {}

    class FakePolicy:
        def __init__(self) -> None:
            observed["policy"] = self

    class FakeEvidence:
        def __init__(self, **kwargs: str) -> None:
            observed["evidence_object"] = self
            observed["evidence"] = kwargs

    class FakeRequest:
        def __init__(self, **kwargs: str) -> None:
            observed["request"] = kwargs

    def fake_compile(
        definitions: tuple[object, ...],
        request: object,
        *,
        active_policy: object,
        operational_evidence: object,
    ) -> _FakeReport:
        observed["definitions"] = definitions
        observed["request_object"] = request
        observed["active_policy"] = active_policy
        observed["operational_evidence"] = operational_evidence
        return report

    monkeypatch.setitem(
        sys.modules,
        "warp_taskgen.site_composition",
        SimpleNamespace(
            ActiveSitePolicy=FakePolicy,
            OperationalEvidence=FakeEvidence,
            SiteDoctorRequest=FakeRequest,
            compile_site_definitions=fake_compile,
            default_site_definitions=lambda: ("gitlab-definition",),
        ),
    )


def test_site_doctor_parser_requires_benchmark_and_use_case() -> None:
    args = _parser().parse_args(
        [
            "site",
            "doctor",
            "gitlab",
            "--benchmark",
            "webarena_verified",
            "--use-case",
            "read_only",
        ]
    )

    assert args.command == "site"
    assert args.site_command == "doctor"
    assert args.site == "gitlab"
    assert args.benchmark == "webarena_verified"
    assert args.use_case == "read_only"
    assert args.json is False


@pytest.mark.parametrize(
    ("static_status", "expected_exit"),
    [("complete", 0), ("incomplete", 1), ("invalid", 2)],
)
def test_site_doctor_exit_code_uses_static_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    static_status: str,
    expected_exit: int,
) -> None:
    observed: dict[str, object] = {}
    _install_fake_composition(monkeypatch, _FakeReport(static_status), observed)

    result = site_doctor.dispatch_site_doctor(
        SimpleNamespace(
            site_command="doctor",
            site="gitlab",
            benchmark="webarena_verified",
            use_case="read_only",
            json=False,
        )
    )

    assert result == expected_exit
    assert observed["request"] == {
        "site": "gitlab",
        "benchmark": "webarena_verified",
        "use_case": "read_only",
    }
    output = capsys.readouterr().out
    assert "Static status: " + static_status in output
    assert "Overall status: blocked" in output


def test_site_doctor_uses_deny_all_policy_and_missing_operational_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _FakeReport("complete")
    observed: dict[str, object] = {}
    _install_fake_composition(monkeypatch, report, observed)

    assert (
        site_doctor.dispatch_site_doctor(
            SimpleNamespace(
                site_command="doctor",
                site="gitlab",
                benchmark="webarena_verified",
                use_case="read_only",
                json=True,
            )
        )
        == 0
    )

    assert observed["request"] == {
        "site": "gitlab",
        "benchmark": "webarena_verified",
        "use_case": "read_only",
    }
    assert observed["definitions"] == ("gitlab-definition",)
    assert observed["active_policy"] is observed["policy"]
    assert observed["operational_evidence"] is observed["evidence_object"]
    assert observed["evidence"] == {
        "configured_host": "missing",
        "admission": "missing",
        "execution": "missing",
        "scoring": "missing",
    }


def test_site_doctor_json_is_the_report_projection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _FakeReport("complete")
    _install_fake_composition(monkeypatch, report)

    result = site_doctor.dispatch_site_doctor(
        SimpleNamespace(
            site_command="doctor",
            site="gitlab",
            benchmark="webarena_verified",
            use_case="read_only",
            json=True,
        )
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == report.to_dict()


def test_canonical_cli_dispatches_complete_static_diagnostic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "site",
        "doctor",
        "gitlab",
        "--benchmark",
        "webarena_verified",
        "--use-case",
        "phase_2_feasibility",
        "--json",
    ]

    parsed = build_parser().parse_args(argv)
    assert (parsed.command, parsed.site_command) == ("site", "doctor")
    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["static_status"] == "complete"
    assert report["status"] == "blocked"
    assert (
        next(finding for finding in report["findings"] if finding["capability"] == "active_policy")[
            "state"
        ]
        == "missing"
    )


def test_site_doctor_invalid_identity_returns_two_without_echoing_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "site",
            "doctor",
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


def test_site_doctor_unknown_benchmark_does_not_echo_secret_like_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "site",
            "doctor",
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
    assert json.loads(output)["static_status"] == "invalid"
