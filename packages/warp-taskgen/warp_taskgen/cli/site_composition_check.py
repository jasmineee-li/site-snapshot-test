"""CLI adapter for the canonical ``site composition check`` command."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from typing import Any

from warp_taskgen.site_composition import (
    SiteCompositionCheckRequest,
    check_site_composition,
    default_site_compositions,
)

_EXIT_CODES = {"complete": 0, "incomplete": 1, "invalid": 2}


def _add_check_arguments(check_parser: Any) -> None:
    check_parser.add_argument("site", help="Canonical Site name.")
    check_parser.add_argument(
        "--benchmark",
        required=True,
        help="Canonical Benchmark name or an accepted alias.",
    )
    check_parser.add_argument(
        "--use-case",
        required=True,
        dest="use_case",
        help="Host-owned Site Composition use case.",
    )
    check_parser.add_argument(
        "--carrier",
        help="Exact carrier identity required by the use case.",
    )
    check_parser.add_argument(
        "--action-kind",
        dest="action_kind",
        help="Exact action kind required by the use case.",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Print deterministic report JSON instead of human-readable text.",
    )


def _add_check_parser(parent: Any, *, help_text: str) -> None:
    check_parser = parent.add_parser(
        "check",
        help=help_text,
        description=(
            "Check one static Site Composition. Active policy and live evidence are not checked."
        ),
    )
    _add_check_arguments(check_parser)


def add_site_composition_parser(subparsers: Any) -> None:
    """Add the canonical nested ``site composition check`` parser."""

    site_parser = subparsers.add_parser(
        "site",
        help="Inspect static Site Composition declarations.",
    )
    site_subparsers = site_parser.add_subparsers(dest="site_command", required=True)
    composition_parser = site_subparsers.add_parser(
        "composition",
        help="Inspect static Site Composition declarations.",
    )
    composition_subparsers = composition_parser.add_subparsers(
        dest="composition_command",
        required=True,
    )
    _add_check_parser(
        composition_subparsers,
        help_text="Check one Site Composition without contacting a host.",
    )


def _compile_default(
    *, site: str, benchmark: str, use_case: str, carrier: str | None, action_kind: str | None
) -> Any:
    request = SiteCompositionCheckRequest(
        site=site,
        benchmark=benchmark,
        use_case=use_case,
        carrier=carrier,
        action_kind=action_kind,
    )
    return check_site_composition(default_site_compositions(), request)


def render_json(report: Any) -> str:
    to_json = getattr(report, "to_json", None)
    if not callable(to_json):
        raise TypeError("Site Composition report does not expose to_json()")
    return str(to_json())


def _stable_value(value: Any) -> str:
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
        line = (
            f"- {finding.get('capability', 'unknown')}: "
            f"{finding.get('state', 'unknown')} [{finding.get('code', 'unknown')}]"
        )
        detail = finding.get("detail")
        if detail:
            line += f" {detail}"
        provenance = finding.get("provenance")
        if provenance:
            line += f" provenance={_stable_value(provenance)}"
        lines.append(line)
    return tuple(lines)


def render_human(report: Any) -> str:
    to_dict = getattr(report, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("Site Composition report does not expose to_dict()")
    data = to_dict()
    if not isinstance(data, Mapping):
        raise TypeError("Site Composition report to_dict() must return an object")
    lines = [
        f"WARP Taskgen site composition check: site={data.get('site', 'unknown')} "
        f"benchmark={data.get('benchmark', 'unknown')} "
        f"use_case={data.get('use_case', 'unknown')}",
        f"Static Site Composition status: {data.get('static_status', 'invalid')}",
        "Scope: static Site Composition only; active policy and live evidence not checked.",
        f"Operational readiness: {data.get('readiness_status', 'blocked')}",
    ]
    if data.get("carrier"):
        lines.append(f"Carrier: {data['carrier']}")
    if data.get("action_kind"):
        lines.append(f"Action kind: {data['action_kind']}")
    if data.get("site_composition_digest"):
        lines.append(f"Site Composition digest: {data['site_composition_digest']}")
    lines.extend(_finding_lines(data.get("findings")))
    return "\n".join(lines)


def _exit_code(report: Any) -> int:
    try:
        status = report.to_dict().get("static_status")
    except (AttributeError, TypeError):
        status = getattr(report, "static_status", None)
    return _EXIT_CODES.get(str(status), 2)


def dispatch_site_composition(args: argparse.Namespace) -> int:
    """Compile and render one explicit static Site Composition request."""

    if (
        getattr(args, "site_command", None) != "composition"
        or getattr(args, "composition_command", None) != "check"
    ):
        return 2
    try:
        report = _compile_default(
            site=args.site,
            benchmark=args.benchmark,
            use_case=args.use_case,
            carrier=getattr(args, "carrier", None),
            action_kind=getattr(args, "action_kind", None),
        )
    except (ImportError, TypeError, ValueError) as exc:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "schema": "warp-site-composition-check-v1",
                        "scope": "static_site_composition_only",
                        "static_status": "invalid",
                        "site_composition_digest": None,
                        "source_package": None,
                        "source_package_version": None,
                        "source_provenance": [],
                        "readiness_status": "blocked",
                        "readiness_blockers": [
                            "active_policy_not_checked",
                            "live_evidence_not_checked",
                        ],
                        "active_policy_checked": False,
                        "live_evidence_checked": False,
                        "error": exc.__class__.__name__,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(f"WARP Taskgen site composition check: invalid input ({exc.__class__.__name__})")
        return 2
    rendered = render_json(report) if getattr(args, "json", False) else render_human(report)
    print(rendered)
    return _exit_code(report)


__all__ = [
    "add_site_composition_parser",
    "dispatch_site_composition",
    "render_human",
    "render_json",
]
