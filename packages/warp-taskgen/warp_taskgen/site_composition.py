"""Static composition diagnostics for one Site/Benchmark use case.

The records in this module reference existing behavior owners.  They do not
construct runtime catalogs, execute editors or evaluators, inspect a Benchmark
Instance, or grant admission.  ``compile_site_definitions`` only validates
that the referenced contracts close for the requested static use case.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any

from warp_taskgen.benchmark_capabilities import DEFAULT_BENCHMARK_CATALOG
from warp_taskgen.site_composition_contracts import (
    EDGE_ORDER,
    ActiveSitePolicy,
    CapabilityFinding,
    CapabilityReference,
    CapabilityState,
    FindingOutcome,
    OperationalEvidence,
    SiteBenchmarkBinding,
    SiteDefinition,
    SiteDoctorReport,
    SiteDoctorRequest,
    StaticStatus,
)

_EDGE_ORDER = EDGE_ORDER
_FINDING_ORDER = (
    "registration",
    *_EDGE_ORDER,
    "benchmark_capability",
    "static_readiness",
    "active_policy",
    "configured_host_feasibility",
    "admission",
    "execution",
    "scoring",
)
_USE_CASE_REQUIREMENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "phase_1_generation": ("targeting", "profile", "editor_specs", "action_cards"),
        "phase_2_generation": (
            "targeting",
            "profile",
            "editor_specs",
            "seed",
            "read_surface",
            "readback",
        ),
        "phase_2_feasibility": (
            "targeting",
            "profile",
            "editor_specs",
            "seed",
            "feasibility",
            "read_surface",
            "readback",
        ),
        "phase_4_execution": (
            "targeting",
            "profile",
            "editor_specs",
            "seed",
            "feasibility",
            "read_surface",
            "readback",
            "final_state",
            "action_cards",
        ),
        "warp_evaluation": ("targeting", "profile", "final_state"),
        # The research POC's complete fake parent/body/reply chain.
        "ugc_reply": (
            "targeting",
            "profile",
            "editor_specs",
            "seed",
            "feasibility",
            "read_surface",
            "readback",
            "final_state",
            "action_cards",
        ),
    }
)
_DIAGNOSTIC_USE_CASES = frozenset({"ugc_reply"})


def _finding(
    capability: str,
    state: CapabilityState,
    *,
    detail: str,
    provenance: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    blocked: bool = False,
) -> CapabilityFinding:
    outcome: FindingOutcome
    if blocked:
        outcome = "blocked"
    elif state in {"supported", "not_applicable"}:
        outcome = "pass"
    else:
        outcome = "failure"
    return CapabilityFinding(
        capability=capability,
        state=state,
        outcome=outcome,
        code=f"{capability}.{state}",
        detail=detail,
        provenance=provenance,
        dependencies=dependencies,
    )


def _invalid_report(
    request: SiteDoctorRequest,
    detail: str,
    *,
    provenance: tuple[str, ...] = (),
) -> SiteDoctorReport:
    request_identity = ":".join((request.benchmark, request.site, request.use_case))
    request_digest = hashlib.sha256(request_identity.encode("utf-8")).hexdigest()
    request_provenance = (
        "site_composition.request",
        f"request_sha256:{request_digest}",
        *provenance,
    )
    findings = (
        _finding(
            "registration",
            "unsupported",
            detail=detail,
            provenance=request_provenance,
        ),
        _finding(
            "static_readiness",
            "unsupported",
            detail="static closure cannot be evaluated",
            provenance=request_provenance,
            dependencies=("registration",),
            blocked=True,
        ),
    )
    return SiteDoctorReport(
        site="invalid",
        benchmark="invalid",
        use_case="invalid",
        static_status="invalid",
        status="invalid",
        definition_digest="",
        findings=findings,
    )


def _digest_value(value: object) -> Any:
    """Project contract metadata without retaining executable owner state."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _digest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        projected = [_digest_value(item) for item in value]
        return sorted(projected, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [_digest_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        owner_type = type(value)
        return {
            "type": f"{owner_type.__module__}.{owner_type.__qualname__}",
            "fields": {
                field.name: _digest_value(getattr(value, field.name)) for field in fields(value)
            },
        }
    if callable(value):
        return {
            "callable": (
                f"{getattr(value, '__module__', type(value).__module__)}."
                f"{getattr(value, '__qualname__', type(value).__qualname__)}"
            )
        }
    owner_type = type(value)
    return {"type": f"{owner_type.__module__}.{owner_type.__qualname__}"}


def _owner_projection(
    name: str,
    owner: object | None,
    *,
    definition: SiteDefinition,
    binding: SiteBenchmarkBinding,
) -> Any:
    if owner is None:
        return None
    if name in {"editor_specs", "action_cards"} and isinstance(owner, tuple):
        projected = [_digest_value(item) for item in owner]
        if name == "action_cards":
            for item in projected:
                if isinstance(item, dict):
                    projected_fields = item.get("fields")
                    if isinstance(projected_fields, dict) and isinstance(
                        projected_fields.get("route_ids"), list
                    ):
                        projected_fields["route_ids"] = sorted(
                            projected_fields["route_ids"],
                            key=lambda value: json.dumps(value, sort_keys=True),
                        )
        return sorted(projected, key=lambda item: json.dumps(item, sort_keys=True))
    owner_type = type(owner)
    projection: dict[str, Any] = {
        "type": f"{owner_type.__module__}.{owner_type.__qualname__}",
        "contract": _digest_value(owner),
    }
    for field in ("benchmark", "site", "key"):
        value = getattr(owner, field, None)
        if isinstance(value, (str, tuple)):
            projection[field] = value
    if name == "targeting" and callable(getattr(owner, "routes", None)):
        from warp_taskgen.sites import TargetingContext

        try:
            routes = owner.routes(
                TargetingContext(benchmark=binding.benchmark, site=definition.site)
            )
            projection["routes"] = [route.as_dict() for route in sorted(routes, key=lambda r: r.id)]
        except Exception as exc:
            projection["routes_error"] = exc.__class__.__name__
    return projection


def _definition_digest(definition: SiteDefinition, binding: SiteBenchmarkBinding) -> str:
    projection = {
        "site": definition.site,
        "benchmark": binding.benchmark,
        "provenance": list(definition.provenance),
        "capabilities": {
            name: {
                "state": reference.state,
                "owner": _owner_projection(
                    name,
                    reference.owner,
                    definition=definition,
                    binding=binding,
                ),
                "provenance": list(reference.provenance),
            }
            for name in _EDGE_ORDER
            for reference in (getattr(binding, name),)
        },
    }
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_reference(
    name: str,
    reference: CapabilityReference[Any],
    *,
    definition: SiteDefinition,
    binding: SiteBenchmarkBinding,
    request: SiteDoctorRequest,
) -> tuple[CapabilityState, str]:
    if reference.state != "supported":
        return reference.state, f"{name} is declared {reference.state} for this binding"
    owner = reference.owner
    if name == "targeting":
        required = ("validate", "validate_task", "routes", "match", "reconstruct")
        if getattr(owner, "site", None) != definition.site:
            return "unsupported", "targeting owner Site identity does not match definition"
        if binding.benchmark not in getattr(owner, "supported_benchmarks", frozenset()):
            return "unsupported", "targeting owner does not support the Benchmark"
        if any(not callable(getattr(owner, method, None)) for method in required):
            return "unsupported", "targeting owner is missing required deterministic methods"
        try:
            from warp_taskgen.sites import CanonicalRoute, TargetingContext

            owner.validate()
            routes = tuple(
                owner.routes(TargetingContext(benchmark=binding.benchmark, site=definition.site))
            )
        except Exception as exc:
            return (
                "unsupported",
                f"targeting owner failed static validation ({exc.__class__.__name__})",
            )
        if (
            not routes
            or any(
                not isinstance(route, CanonicalRoute)
                or route.site != definition.site
                or not isinstance(route.id, str)
                or not route.id.strip()
                or not isinstance(route.kind, str)
                or not route.kind.strip()
                or (
                    route.compatibility_kind is not None
                    and not isinstance(route.compatibility_kind, str)
                )
                for route in routes
            )
            or len({route.id for route in routes}) != len(routes)
        ):
            return "unsupported", "targeting owner has invalid or duplicate canonical routes"
    elif name == "profile":
        if getattr(owner, "site", None) != definition.site or binding.benchmark not in getattr(
            owner, "supported_benchmarks", frozenset()
        ):
            return "unsupported", "profile owner identity does not match the binding"
        required = (
            "canonicalize_surface_id",
            "resolve_profile_surface",
            "route_contract_facts",
        )
        if any(not callable(getattr(owner, method, None)) for method in required):
            return "unsupported", "Site has no complete profile-route capability"
    elif name == "editor_specs":
        from warp_taskgen.editors._registry import EditorMethodSpec

        if not isinstance(owner, tuple) or not owner:
            return "missing", "editor metadata snapshot is empty"
        if any(
            not isinstance(spec, EditorMethodSpec)
            or spec.benchmark != binding.benchmark
            or spec.site != definition.site
            for spec in owner
        ):
            return "unsupported", "editor metadata identity does not match the binding"
        if any(
            not isinstance(spec.method, str)
            or not spec.method.strip()
            or not isinstance(spec.http, tuple)
            or len(spec.http) != 2
            or any(not isinstance(part, str) or not part.strip() for part in spec.http)
            or not isinstance(spec.kinds, frozenset)
            or any(not isinstance(kind, str) or not kind.strip() for kind in spec.kinds)
            or not isinstance(spec.surface_id_per_kind, Mapping)
            or set(spec.surface_id_per_kind) != set(spec.kinds)
            or any(
                not isinstance(surface, str) or not surface.strip()
                for surface in spec.surface_id_per_kind.values()
            )
            for spec in owner
        ):
            return "unsupported", "editor metadata contains a malformed method contract"
        editor_keys = [(spec.benchmark, spec.site, spec.method) for spec in owner]
        if len(set(editor_keys)) != len(editor_keys):
            return "unsupported", "editor metadata contains duplicate method identities"
        if request.carrier and not any(
            request.carrier in spec.surface_id_per_kind.values() for spec in owner
        ):
            return "missing", "no editor metadata binds the requested carrier"
    elif name == "seed":
        if getattr(owner, "key", None) != (binding.benchmark, definition.site):
            return "unsupported", "seed registration identity does not match the binding"
        if not callable(getattr(owner, "create", None)):
            return "unsupported", "seed registration has no editor factory seam"
    elif name == "feasibility":
        if (
            getattr(owner, "benchmark", None) != binding.benchmark
            or getattr(owner, "site", None) != definition.site
        ):
            return "unsupported", "feasibility policy identity does not match the binding"
        required = ("probe_targets", "classify_probe", "decide_source_data")
        if any(not callable(getattr(owner, method, None)) for method in required):
            return "unsupported", "feasibility policy is missing deterministic methods"
    elif name == "read_surface":
        if getattr(owner, "site", None) != definition.site or binding.benchmark not in getattr(
            owner, "supported_benchmarks", frozenset()
        ):
            return "unsupported", "read-surface owner identity does not match the binding"
        if not callable(getattr(owner, "build_read_surface_plan", None)):
            return "unsupported", "Site has no read-surface planning capability"
    elif name == "readback":
        if getattr(owner, "site", None) != definition.site or binding.benchmark not in getattr(
            owner, "supported_benchmarks", frozenset()
        ):
            return "unsupported", "readback owner identity does not match the binding"
        if not callable(getattr(owner, "interpret_readback", None)):
            return "unsupported", "Site has no pure readback interpreter"
    elif name == "final_state":
        if (
            getattr(owner, "benchmark", None) != binding.benchmark
            or getattr(owner, "site", None) != definition.site
        ):
            return "unsupported", "final-state evaluator identity does not match the binding"
        if not callable(getattr(owner, "evaluate", None)):
            return "unsupported", "final-state evaluator has no evaluation seam"
    elif name == "action_cards":
        from warp_taskgen.adversarial_actions.capability_adapters import CapabilityTaskAdapter
        from warp_taskgen.adversarial_actions.capability_contracts import (
            get_action_capability_contract,
        )
        from warp_taskgen.editors._registry import EditorMethodSpec

        if not isinstance(owner, tuple) or not owner:
            return "missing", "action-card snapshot is empty"
        if any(not isinstance(card, CapabilityTaskAdapter) for card in owner):
            return "unsupported", "action-card snapshot contains an invalid owner"
        if any(
            card.benchmark_family != binding.benchmark or card.site != definition.site
            for card in owner
        ):
            return "unsupported", "action-card identity does not match the binding"
        if any(
            not isinstance(card.id, str)
            or not card.id.strip()
            or not isinstance(card.route_ids, tuple)
            or not card.route_ids
            or any(not isinstance(route_id, str) or not route_id for route_id in card.route_ids)
            for card in owner
        ):
            return "unsupported", "action-card snapshot contains a malformed identity"
        card_ids = [card.id for card in owner]
        if len(set(card_ids)) != len(card_ids):
            return "unsupported", "action-card snapshot contains duplicate card identities"
        if any(len(set(card.route_ids)) != len(card.route_ids) for card in owner):
            return "unsupported", "action-card snapshot contains duplicate route identities"
        if request.action_kind and not any(
            card.action_kind == request.action_kind for card in owner
        ):
            return "missing", "no action card binds the requested action"
        try:
            for card in owner:
                card.validate()
        except Exception:
            return "unsupported", "action-card contract is invalid"
        for dependency in ("targeting", "profile", "editor_specs"):
            dependency_state, _ = _validate_reference(
                dependency,
                getattr(binding, dependency),
                definition=definition,
                binding=binding,
                request=request,
            )
            if dependency_state != "supported":
                return "unsupported", f"action-card {dependency} closure is unavailable"
        targeting = binding.targeting.owner if binding.targeting.state == "supported" else None
        profile = binding.profile.owner if binding.profile.state == "supported" else None
        editor_specs = (
            binding.editor_specs.owner if binding.editor_specs.state == "supported" else None
        )
        if targeting is None or profile is None or not isinstance(editor_specs, tuple):
            return (
                "unsupported",
                "action-card closure requires targeting, profile, and editor owners",
            )
        try:
            from warp_taskgen.sites import TargetingContext

            routes = targeting.routes(
                TargetingContext(benchmark=binding.benchmark, site=definition.site)
            )
            route_kinds = {
                kind for route in routes for kind in (route.kind, route.compatibility_kind) if kind
            }
        except Exception as exc:
            return (
                "unsupported",
                f"action-card route closure failed ({exc.__class__.__name__})",
            )
        canonicalize = getattr(profile, "canonicalize_surface_id", None)
        try:
            for card in owner:
                for route_id in card.route_ids:
                    parts = route_id.split(".")
                    if len(parts) != 4:
                        return "unsupported", "action-card route identity is malformed"
                    route_site, raw_surface, kind, method = parts
                    matching_specs = [
                        spec
                        for spec in editor_specs
                        if isinstance(spec, EditorMethodSpec)
                        and spec.method == method
                        and kind in spec.kinds
                    ]
                    if (
                        route_site != definition.site
                        or kind not in route_kinds
                        or not matching_specs
                    ):
                        return "unsupported", "action-card route has no targeting/editor closure"
                    route_surface = canonicalize(
                        benchmark=binding.benchmark,
                        raw_surface_id=raw_surface,
                    )
                    action_contract = get_action_capability_contract(card.action_kind)
                    if action_contract is None:
                        return "unsupported", "action-card capability contract is unavailable"
                    editor_surfaces = {
                        canonicalize(
                            benchmark=binding.benchmark,
                            raw_surface_id=spec.surface_id_per_kind.get(kind),
                        )
                        for spec in matching_specs
                    }
                    if not route_surface or route_surface not in editor_surfaces:
                        return "unsupported", "action-card carrier has no profile/editor closure"
                    raw_target = card.action_target_contract
                    target = (
                        raw_target.to_dict()
                        if callable(getattr(raw_target, "to_dict", None))
                        else raw_target
                    )
                    if target is None:
                        if (
                            method not in action_contract.compatible_editor_methods
                            or route_surface not in action_contract.compatible_carrier_surfaces
                        ):
                            return (
                                "unsupported",
                                "action-card method/carrier contract is incompatible",
                            )
                        continue
                    if not isinstance(target, Mapping):
                        return "unsupported", "action-card target contract is malformed"
                    target_surface = canonicalize(
                        benchmark=binding.benchmark,
                        raw_surface_id=target.get("target_surface_id"),
                    )
                    source_surface = canonicalize(
                        benchmark=binding.benchmark,
                        raw_surface_id=target.get("source_surface_id"),
                    )
                    target_method = target.get("target_editor_method")
                    target_specs = [
                        spec
                        for spec in editor_specs
                        if isinstance(spec, EditorMethodSpec) and spec.method == target_method
                    ]
                    target_editor_surfaces = {
                        canonicalize(
                            benchmark=binding.benchmark,
                            raw_surface_id=surface,
                        )
                        for spec in target_specs
                        for surface in spec.surface_id_per_kind.values()
                    }
                    if (
                        target.get("site") != definition.site
                        or target.get("action_kind") != card.action_kind
                        or target.get("source_editor_method") != method
                        or source_surface != route_surface
                        or target_method not in action_contract.compatible_editor_methods
                        or target_surface not in action_contract.compatible_carrier_surfaces
                        or target_surface not in target_editor_surfaces
                    ):
                        return "unsupported", "action-card target contract has no editor closure"
        except Exception as exc:
            return (
                "unsupported",
                f"action-card profile/editor closure failed ({exc.__class__.__name__})",
            )
    return "supported", f"{name} owner is structurally available"


def compile_site_definitions(
    definitions: Iterable[SiteDefinition],
    request: SiteDoctorRequest,
    *,
    active_policy: ActiveSitePolicy,
    operational_evidence: OperationalEvidence,
) -> SiteDoctorReport:
    """Validate static closure without executing any referenced owner."""

    if not isinstance(request, SiteDoctorRequest):
        raise TypeError("site doctor request must be typed")
    if not isinstance(active_policy, ActiveSitePolicy):
        raise TypeError("active policy must be typed")
    if not isinstance(operational_evidence, OperationalEvidence):
        raise TypeError("operational evidence must be typed")
    items = tuple(definitions)
    if any(not isinstance(item, SiteDefinition) for item in items):
        return _invalid_report(request, "definitions must contain typed SiteDefinition values")
    by_site: dict[str, SiteDefinition] = {}
    for definition in items:
        if definition.site in by_site:
            return _invalid_report(
                request,
                f"duplicate Site definition {definition.site!r}",
                provenance=(*by_site[definition.site].provenance, *definition.provenance),
            )
        by_site[definition.site] = definition

    try:
        benchmark_contract = DEFAULT_BENCHMARK_CATALOG.resolve(request.benchmark)
    except ValueError:
        return _invalid_report(request, "Benchmark identity is not registered")
    if benchmark_contract.is_comparison_only and request.use_case != "comparison_ingestion":
        return _invalid_report(
            request,
            "comparison-only Benchmark cannot enter the requested WARP use case",
        )
    required = _USE_CASE_REQUIREMENTS.get(request.use_case)
    if required is None:
        return _invalid_report(request, "Site doctor use case is not registered")
    if (
        request.use_case not in benchmark_contract.capabilities
        and request.use_case not in _DIAGNOSTIC_USE_CASES
    ):
        return _invalid_report(
            request,
            "Benchmark does not declare the requested capability",
        )
    definition = by_site.get(request.site)
    if definition is None:
        return _invalid_report(request, "requested Site definition is not registered")
    binding = next(
        (item for item in definition.bindings if item.benchmark == request.benchmark),
        None,
    )
    if binding is None:
        return _invalid_report(
            request,
            "Site has no binding for the requested Benchmark",
            provenance=definition.provenance,
        )

    findings: list[CapabilityFinding] = [
        _finding(
            "registration",
            "supported",
            detail="definition and Benchmark binding are explicit",
            provenance=definition.provenance,
        )
    ]
    required_failures: list[str] = []
    for name in _EDGE_ORDER:
        reference = getattr(binding, name)
        if name not in required:
            findings.append(
                _finding(
                    name,
                    "not_applicable",
                    detail=f"{name} is not required for {request.use_case}",
                    provenance=reference.provenance,
                )
            )
            continue
        state, detail = _validate_reference(
            name,
            reference,
            definition=definition,
            binding=binding,
            request=request,
        )
        findings.append(_finding(name, state, detail=detail, provenance=reference.provenance))
        if state != "supported":
            required_failures.append(name)

    diagnostic_use_case = request.use_case in _DIAGNOSTIC_USE_CASES
    benchmark_state: CapabilityState = "not_applicable" if diagnostic_use_case else "supported"
    benchmark_detail = (
        "test-only composition profile does not declare a Benchmark capability"
        if diagnostic_use_case
        else f"Benchmark declares {request.use_case}"
    )
    findings.append(
        _finding(
            "benchmark_capability",
            benchmark_state,
            detail=benchmark_detail,
            provenance=(f"benchmark:{request.benchmark}",),
        )
    )
    static_status: StaticStatus = "incomplete" if required_failures else "complete"
    findings.append(
        _finding(
            "static_readiness",
            "missing" if required_failures else "supported",
            detail=(
                f"missing static closure: {', '.join(required_failures)}"
                if required_failures
                else "all required static capability owners close"
            ),
            dependencies=tuple(required),
        )
    )

    policy_supported = active_policy.authorizes(request)
    findings.append(
        _finding(
            "active_policy",
            "supported" if policy_supported else "missing",
            detail=(
                "requested use case is explicitly authorized"
                if policy_supported
                else "static composition does not grant active policy"
            ),
            dependencies=("static_readiness",),
            blocked=not policy_supported,
        )
    )
    operational = (
        ("configured_host_feasibility", operational_evidence.configured_host),
        ("admission", operational_evidence.admission),
        ("execution", operational_evidence.execution),
        ("scoring", operational_evidence.scoring),
    )
    previous = "active_policy"
    for name, state in operational:
        is_supported = state == "supported"
        findings.append(
            _finding(
                name,
                state,
                detail=(
                    f"external {name} evidence is present"
                    if is_supported
                    else f"external {name} evidence is {state}; static doctor does not infer it"
                ),
                dependencies=(previous,),
                blocked=not is_supported,
            )
        )
        previous = name

    order = {name: index for index, name in enumerate(_FINDING_ORDER)}
    sorted_findings = tuple(sorted(findings, key=lambda item: order[item.capability]))
    ready = static_status == "complete" and all(
        finding.state == "supported"
        for finding in sorted_findings
        if finding.capability
        in {
            "active_policy",
            "configured_host_feasibility",
            "admission",
            "execution",
            "scoring",
        }
    )
    return SiteDoctorReport(
        site=request.site,
        benchmark=request.benchmark,
        use_case=request.use_case,
        static_status=static_status,
        status="ready" if ready else "blocked",
        definition_digest=_definition_digest(definition, binding),
        findings=sorted_findings,
    )


def default_site_definitions() -> tuple[SiteDefinition, ...]:
    """Load the explicit built-in diagnostic projection lazily."""

    from warp_taskgen.site_composition_defaults import default_site_definitions as build

    return build()


__all__ = [
    "ActiveSitePolicy",
    "CapabilityFinding",
    "CapabilityReference",
    "CapabilityState",
    "OperationalEvidence",
    "SiteBenchmarkBinding",
    "SiteDefinition",
    "SiteDoctorReport",
    "SiteDoctorRequest",
    "compile_site_definitions",
    "default_site_definitions",
]
