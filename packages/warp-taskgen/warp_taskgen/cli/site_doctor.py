"""Thin, read-only CLI adapter for the Site capability doctor.

The composition compiler owns Site/Benchmark validation.  This module only
parses one requested use case, supplies the explicit diagnostic defaults, and
projects the compiler's report for a human or machine caller.  It deliberately
does not inspect hosts, resolve credentials, launch a browser, or mutate any
catalog or Site state.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from typing import Any

_EXIT_CODES = {"complete": 0, "incomplete": 1, "invalid": 2}


def add_site_doctor_parser(subparsers: Any) -> None:
    """Add ``site doctor`` to an existing top-level subparser collection."""

    site_parser = subparsers.add_parser(
        "site",
        help="Inspect static Site capability composition.",
    )
    site_subparsers = site_parser.add_subparsers(
        dest="site_command",
        required=True,
    )
    doctor_parser = site_subparsers.add_parser(
        "doctor",
        help="Diagnose one Site/Benchmark use case without contacting a host.",
        description=(
            "Compile the requested Site capability closure. This command is "
            "static and read-only: it does not use credentials, contact a "
            "Benchmark Instance, launch a browser, or change admission."
        ),
    )
    doctor_parser.add_argument("site", help="Canonical Site name.")
    doctor_parser.add_argument(
        "--benchmark",
        required=True,
        help="Canonical Benchmark name or an accepted alias.",
    )
    doctor_parser.add_argument(
        "--use-case",
        required=True,
        dest="use_case",
        help="Requested capability use case, for example phase_2_feasibility.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the deterministic report JSON instead of human-readable text.",
    )


def _compile_default(*, site: str, benchmark: str, use_case: str) -> Any:
    """Compile one request against the explicit diagnostic definitions.

    Empty policy/evidence values are intentional. They keep static contract
    composition separate from active policy and configured-host proof while
    allowing the report to show those later states as distinct findings.
    """

    from warp_taskgen.site_composition import (
        ActiveSitePolicy,
        OperationalEvidence,
        SiteDoctorRequest,
        compile_site_definitions,
        default_site_definitions,
    )

    request = SiteDoctorRequest(site=site, benchmark=benchmark, use_case=use_case)
    return compile_site_definitions(
        tuple(default_site_definitions()),
        request,
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(
            configured_host="missing",
            admission="missing",
            execution="missing",
            scoring="missing",
        ),
    )


def render_json(report: Any) -> str:
    """Return the compiler-owned deterministic JSON projection."""

    to_json = getattr(report, "to_json", None)
    if not callable(to_json):
        raise TypeError("site doctor report does not expose to_json()")
    return str(to_json())


def _stable_value(value: Any) -> str:
    """Render safe report values without depending on mapping insertion order."""

    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _finding_lines(findings: Any) -> Iterable[str]:
    if not isinstance(findings, (list, tuple)):
        return ()
    lines: list[str] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            lines.append(f"- {finding}")
            continue
        capability = finding.get("capability", "unknown")
        state = finding.get("state", "unknown")
        code = finding.get("code", "unknown")
        detail = finding.get("detail", "")
        line = f"- {capability}: {state} [{code}]"
        if detail:
            line += f" {detail}"
        provenance = finding.get("provenance")
        if provenance:
            line += f" provenance={_stable_value(provenance)}"
        lines.append(line)
    return tuple(lines)


def render_human(report: Any) -> str:
    """Render every report finding in a stable operator-readable form."""

    to_dict = getattr(report, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("site doctor report does not expose to_dict()")
    data = to_dict()
    if not isinstance(data, Mapping):
        raise TypeError("site doctor report to_dict() must return an object")

    static_status = data.get("static_status", getattr(report, "static_status", "invalid"))
    lines = [
        f"WARP Taskgen site doctor: site={data.get('site', 'unknown')} "
        f"benchmark={data.get('benchmark', 'unknown')} "
        f"use_case={data.get('use_case', 'unknown')}",
        f"Static status: {static_status}",
    ]
    if "status" in data:
        lines.append(f"Overall status: {data['status']}")
    if data.get("definition_digest"):
        lines.append(f"Definition digest: {data['definition_digest']}")
    lines.extend(_finding_lines(data.get("findings")))
    return "\n".join(lines)


def _exit_code(report: Any) -> int:
    status = getattr(report, "static_status", None)
    if status is None:
        try:
            status = report.to_dict().get("static_status")
        except (AttributeError, TypeError):
            status = None
    return _EXIT_CODES.get(str(status), 2)


def dispatch_site_doctor(args: argparse.Namespace) -> int:
    """Dispatch the parsed ``site doctor`` command and return its exit code."""

    if getattr(args, "site_command", None) != "doctor":
        return 2
    try:
        report = _compile_default(
            site=args.site,
            benchmark=args.benchmark,
            use_case=args.use_case,
        )
    except (TypeError, ValueError) as exc:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "schema": "warp-site-doctor-experimental-v1",
                        "static_status": "invalid",
                        "status": "invalid",
                        "error": exc.__class__.__name__,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(f"WARP Taskgen site doctor: invalid input ({exc.__class__.__name__})")
        return 2
    rendered = render_json(report) if getattr(args, "json", False) else render_human(report)
    print(rendered)
    return _exit_code(report)


__all__ = [
    "add_site_doctor_parser",
    "dispatch_site_doctor",
    "render_human",
    "render_json",
]
