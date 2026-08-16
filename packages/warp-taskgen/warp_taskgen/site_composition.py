"""Pure static Site Composition checking.

The compiler consumes immutable declarations only.  It never imports or calls
Site, editor, feasibility, readback, evaluator, browser, or host behavior.
Active policy and live operational evidence are intentionally outside this
seam.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from warp_taskgen.benchmark_capabilities import DEFAULT_BENCHMARK_CATALOG
from warp_taskgen.site_composition_contracts import (
    SITE_COMPOSITION_CONTRACT_VERSION,
    SITE_OWNER_ROLE_ORDER,
    SiteBenchmarkComposition,
    SiteComposition,
    SiteCompositionCheckReport,
    SiteCompositionCheckRequest,
    SiteCompositionDigest,
    SiteCompositionFinding,
    SiteCompositionStatus,
    SiteCompositionUseCase,
    SiteCompositionUseCaseCatalog,
    SiteOwnerDeclaration,
    StaticCapabilityState,
)

_FINDING_ORDER = (
    "registration",
    *SITE_OWNER_ROLE_ORDER,
    "benchmark_capability",
    "static_closure",
)


def _finding(
    capability: str,
    state: StaticCapabilityState,
    *,
    detail: str,
    provenance: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
) -> SiteCompositionFinding:
    outcome = "pass" if state in {"supported", "not_applicable"} else "failure"
    return SiteCompositionFinding(
        capability=capability,
        state=state,
        outcome=outcome,
        code=f"{capability}.{state}",
        detail=detail,
        provenance=provenance,
        dependencies=dependencies,
    )


def _invalid_report(
    request: SiteCompositionCheckRequest,
    detail: str,
    *,
    provenance: tuple[str, ...] = (),
) -> SiteCompositionCheckReport:
    findings = (
        _finding(
            "registration",
            "unsupported",
            detail=detail,
            provenance=provenance,
        ),
        _finding(
            "static_closure",
            "unsupported",
            detail="static Site Composition cannot be evaluated",
            dependencies=("registration",),
        ),
    )
    return SiteCompositionCheckReport(
        site="invalid",
        benchmark="invalid",
        use_case="invalid",
        carrier=None,
        action_kind=None,
        static_status="invalid",
        site_composition_digest=None,
        findings=findings,
    )


def _declaration_payload(declaration: SiteOwnerDeclaration) -> dict[str, object]:
    return {
        "state": declaration.state,
        "owner_id": declaration.owner_id,
        "contract_version": declaration.contract_version,
        "provenance": list(declaration.provenance),
    }


def _composition_payload(
    composition: SiteComposition,
    use_case_catalog: SiteCompositionUseCaseCatalog,
) -> dict[str, object]:
    benchmarks = []
    for benchmark in composition.benchmark_compositions:
        benchmarks.append(
            {
                "benchmark": benchmark.benchmark,
                "owners": {
                    role: _declaration_payload(benchmark.owner(role))
                    for role in SITE_OWNER_ROLE_ORDER
                },
                "supported_carriers": list(benchmark.supported_carriers),
                "supported_action_kinds": list(benchmark.supported_action_kinds),
                "provenance": list(benchmark.provenance),
            }
        )
    return {
        "schema": "warp-site-composition-digest-v1",
        "contract_version": SITE_COMPOSITION_CONTRACT_VERSION,
        "site": composition.site,
        "benchmarks": benchmarks,
        "provenance": list(composition.provenance),
        "use_cases": [
            {
                "id": entry.id,
                "scope": entry.scope,
                "required_owner_roles": list(entry.required_owner_roles),
                "requires_carrier": entry.requires_carrier,
                "requires_action_kind": entry.requires_action_kind,
            }
            for entry in use_case_catalog.entries
        ],
    }


def site_composition_digest(
    composition: SiteComposition,
    *,
    use_case_catalog: SiteCompositionUseCaseCatalog | None = None,
) -> SiteCompositionDigest:
    """Return the deterministic digest of declarations for one Site only."""

    if not isinstance(composition, SiteComposition):
        raise TypeError("site_composition_digest requires a SiteComposition")
    catalog = use_case_catalog or SiteCompositionUseCaseCatalog.default()
    if not isinstance(catalog, SiteCompositionUseCaseCatalog):
        raise TypeError("use_case_catalog must be a SiteCompositionUseCaseCatalog")
    payload = _composition_payload(composition, catalog)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _check_request_requirements(
    benchmark: SiteBenchmarkComposition,
    use_case: SiteCompositionUseCase,
    request: SiteCompositionCheckRequest,
) -> dict[str, tuple[StaticCapabilityState, str]]:
    """Return declaration overrides for exact carrier/action requirements."""

    overrides: dict[str, tuple[StaticCapabilityState, str]] = {}
    if use_case.requires_carrier:
        if request.carrier is None:
            overrides["profile"] = ("missing", "public use case requires an exact carrier")
            overrides["editor_specification"] = (
                "missing",
                "public use case requires an exact carrier",
            )
        elif request.carrier not in benchmark.supported_carriers:
            overrides["profile"] = (
                "missing",
                "requested carrier is not declared by the Site Composition",
            )
            overrides["editor_specification"] = (
                "missing",
                "requested carrier is not declared by the Site Composition",
            )
    if use_case.requires_action_kind:
        if request.action_kind is None:
            overrides["action_cards"] = (
                "missing",
                "public use case requires an exact action kind",
            )
        elif request.action_kind not in benchmark.supported_action_kinds:
            overrides["action_cards"] = (
                "missing",
                "requested action kind is not declared by the Site Composition",
            )
    return overrides


def _check_one(
    composition: SiteComposition,
    benchmark: SiteBenchmarkComposition,
    request: SiteCompositionCheckRequest,
    use_case: SiteCompositionUseCase,
    use_case_catalog: SiteCompositionUseCaseCatalog,
) -> SiteCompositionCheckReport:
    required = set(use_case.required_owner_roles)
    overrides = _check_request_requirements(benchmark, use_case, request)
    findings: list[SiteCompositionFinding] = [
        _finding(
            "registration",
            "supported",
            detail="Site and Benchmark declarations are explicit",
            provenance=(*composition.provenance, *benchmark.provenance),
        )
    ]
    failures: list[str] = []
    for role in SITE_OWNER_ROLE_ORDER:
        declaration = benchmark.owner(role)
        if role not in required:
            findings.append(
                _finding(
                    role,
                    "not_applicable",
                    detail=f"{role} is not required for {use_case.id}",
                    provenance=declaration.provenance,
                )
            )
            continue
        state, detail = overrides.get(
            role,
            (declaration.state, f"{role} declaration is {declaration.state}"),
        )
        if state == "supported":
            detail = f"{role} owner declaration is structurally available"
        findings.append(
            _finding(
                role,
                state,
                detail=detail,
                provenance=declaration.provenance,
            )
        )
        if state != "supported":
            failures.append(role)

    benchmark_capability: StaticCapabilityState
    if use_case.scope == "static_diagnostic":
        benchmark_capability = "not_applicable"
        benchmark_detail = "static diagnostic use case does not grant Benchmark capability"
    elif request.use_case in _benchmark_contract_capabilities(request.benchmark):
        benchmark_capability = "supported"
        benchmark_detail = f"Benchmark declares {request.use_case}"
    else:
        benchmark_capability = "missing"
        benchmark_detail = "Benchmark does not declare the requested capability"
        failures.append("benchmark_capability")
    findings.append(
        _finding(
            "benchmark_capability",
            benchmark_capability,
            detail=benchmark_detail,
            provenance=(f"benchmark:{request.benchmark}",),
        )
    )
    status: SiteCompositionStatus = "incomplete" if failures else "complete"
    findings.append(
        _finding(
            "static_closure",
            "missing" if failures else "supported",
            detail=(
                f"missing static closure: {', '.join(failures)}"
                if failures
                else "all required static Site owner declarations close"
            ),
            dependencies=tuple(use_case.required_owner_roles),
        )
    )
    order = {name: index for index, name in enumerate(_FINDING_ORDER)}
    ordered = tuple(sorted(findings, key=lambda item: order[item.capability]))
    return SiteCompositionCheckReport(
        site=request.site,
        benchmark=request.benchmark,
        use_case=request.use_case,
        carrier=request.carrier,
        action_kind=request.action_kind,
        static_status=status,
        site_composition_digest=site_composition_digest(
            composition,
            use_case_catalog=use_case_catalog,
        ),
        findings=ordered,
    )


def _benchmark_contract_capabilities(benchmark: str) -> frozenset[str]:
    """Read Benchmark metadata without touching Site behavior."""

    return frozenset(DEFAULT_BENCHMARK_CATALOG.resolve(benchmark).capabilities)


def check_site_composition(
    compositions: Iterable[SiteComposition],
    request: SiteCompositionCheckRequest,
    *,
    use_case_catalog: SiteCompositionUseCaseCatalog | None = None,
) -> SiteCompositionCheckReport:
    """Check one explicit static Site Composition request.

    The only inputs are immutable declaration records.  No owner methods are
    looked up or invoked and no active policy/live evidence is accepted.
    """

    if not isinstance(request, SiteCompositionCheckRequest):
        raise TypeError("Site Composition request must be typed")
    catalog = use_case_catalog or SiteCompositionUseCaseCatalog.default()
    if not isinstance(catalog, SiteCompositionUseCaseCatalog):
        raise TypeError("use_case_catalog must be a SiteCompositionUseCaseCatalog")
    items = tuple(compositions)
    if any(not isinstance(item, SiteComposition) for item in items):
        return _invalid_report(request, "compositions must contain typed SiteComposition values")
    by_site: dict[str, SiteComposition] = {}
    for item in items:
        if item.site in by_site:
            return _invalid_report(
                request,
                "duplicate Site Composition identity",
            )
        by_site[item.site] = item
    try:
        benchmark_contract = DEFAULT_BENCHMARK_CATALOG.resolve(request.benchmark)
    except ValueError:
        return _invalid_report(request, "Benchmark identity is not registered")
    if benchmark_contract.is_comparison_only:
        return _invalid_report(
            request,
            "comparison-only Benchmark cannot enter a static Site use case",
        )
    use_case = catalog.resolve(request.use_case)
    if use_case is None:
        return _invalid_report(request, "Site Composition use case is not registered")
    composition = by_site.get(request.site)
    if composition is None:
        return _invalid_report(request, "requested Site Composition is not registered")
    benchmark = composition.benchmark(request.benchmark)
    if benchmark is None:
        return _invalid_report(
            request,
            "Site Composition has no projection for the requested Benchmark",
            provenance=composition.provenance,
        )
    return _check_one(composition, benchmark, request, use_case, catalog)


def default_site_compositions() -> tuple[SiteComposition, ...]:
    """Load the explicit built-in static projections lazily."""

    from warp_taskgen.site_composition_defaults import default_site_compositions as build

    return build()


__all__ = [
    "SITE_OWNER_ROLE_ORDER",
    "SiteBenchmarkComposition",
    "SiteComposition",
    "SiteCompositionCheckReport",
    "SiteCompositionCheckRequest",
    "SiteCompositionFinding",
    "SiteCompositionUseCase",
    "SiteCompositionUseCaseCatalog",
    "SiteOwnerDeclaration",
    "StaticCapabilityState",
    "check_site_composition",
    "default_site_compositions",
    "site_composition_digest",
]
