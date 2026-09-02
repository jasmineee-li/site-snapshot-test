"""Opt-in Phase 2 admission for the Rocket.Chat decision slice.

This is deliberately a feature-local admission predicate.  It does not alter
the TAC Benchmark capability catalog (which remains source-only until live
proof exists) and it does not provide a generic auth/workflow registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlsplit

from warp_taskgen.benchmark_capabilities import get_benchmark_capabilities
from warp_taskgen.phase_1.rocket_chat_contracts import (
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_EVALUATOR_NAME,
    ROCKET_CHAT_SITE,
    RocketChatContractError,
)
from warp_taskgen.phase_1.rocket_chat_task_envelope import (
    ROCKET_CHAT_CONTRACT_FIELD,
    project_rocket_chat_static_contract,
    validate_rocket_chat_benign_task,
)
from warp_taskgen.runtime_composition import Phase2RuntimeAdmission
from warp_taskgen.sites.catalog import SiteCatalog


def _failure(reason: str, *checks: str) -> Phase2RuntimeAdmission:
    return Phase2RuntimeAdmission(False, reason, checks)


def _valid_reset_endpoint(instance: Mapping[str, object]) -> bool:
    endpoint = instance.get("reset_endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        return False
    parsed = urlsplit(endpoint.strip())
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.netloc
        and parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def _task_benchmark(task: Mapping[str, object]) -> str | None:
    raw = task.get("benchmark") or task.get("benchmark_name") or task.get("benchmark_adapter")
    if raw in (None, ""):
        return None
    try:
        return get_benchmark_capabilities(raw).canonical_name
    except ValueError:
        return None


def rocket_chat_phase2_admission(
    tasks: Sequence[Mapping[str, object]],
    instances: Sequence[Mapping[str, object]],
    *,
    site_catalog: SiteCatalog,
    reader_preflight: Callable[[Mapping[str, object]], object],
) -> Phase2RuntimeAdmission:
    """Check every live prerequisite before a Rocket.Chat seed can mutate.

    The predicate is intentionally strict: one malformed task/instance or one
    missing capability rejects the complete explicit run.  This prevents a
    writer-session success or body-text-only render from being promoted to
    Phase 2 admission.
    """

    if not isinstance(site_catalog, SiteCatalog):
        return _failure("invalid_site_catalog")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
        return _failure("no_tasks")
    if not isinstance(instances, Sequence) or isinstance(instances, (str, bytes)) or not instances:
        return _failure("no_instances")
    try:
        capabilities = get_benchmark_capabilities(ROCKET_CHAT_BENCHMARK)
    except ValueError:
        return _failure("benchmark_unregistered")
    if capabilities.canonical_name != ROCKET_CHAT_BENCHMARK:
        return _failure("benchmark_identity_mismatch")

    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            return _failure("malformed_task", f"task_{index}")
        if _task_benchmark(task) != ROCKET_CHAT_BENCHMARK:
            return _failure("task_benchmark_mismatch", f"task_{index}")
        if str(task.get("site") or "").strip().lower() != ROCKET_CHAT_SITE:
            return _failure("task_site_mismatch", f"task_{index}")
        if task.get("task_id") not in (None, ""):
            return _failure("native_task_id_forbidden", f"task_{index}")
        try:
            static_contract = project_rocket_chat_static_contract(task)
            if ROCKET_CHAT_CONTRACT_FIELD in task:
                validate_rocket_chat_benign_task(task)
        except (RocketChatContractError, TypeError, ValueError):
            return _failure("task_contract_invalid", f"task_{index}")
        reward = static_contract.get("reward_function")
        evaluations = reward.get("eval") if isinstance(reward, Mapping) else None
        if (
            not isinstance(evaluations, list)
            or len(evaluations) != 1
            or not isinstance(evaluations[0], Mapping)
            or evaluations[0].get("evaluator") != ROCKET_CHAT_EVALUATOR_NAME
        ):
            return _failure("unsupported_evaluator", f"task_{index}")
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            return _failure("malformed_instance", f"instance_{index}")
        if str(instance.get("site_name") or "").strip().lower() != ROCKET_CHAT_SITE:
            return _failure("instance_site_mismatch", f"instance_{index}")
        if _task_benchmark(instance) != ROCKET_CHAT_BENCHMARK:
            return _failure("instance_benchmark_mismatch", f"instance_{index}")
        if not _valid_reset_endpoint(instance):
            return _failure("reset_endpoint_missing", f"instance_{index}")
        try:
            reader_result = reader_preflight(instance)
        except Exception:
            return _failure("reader_preflight_error", f"instance_{index}")
        if not getattr(reader_result, "ok", False):
            return _failure(
                "reader_preflight_failed",
                f"instance_{index}:{getattr(reader_result, 'reason', 'unknown')}",
            )
        site_url = instance.get("site_url")
        if not isinstance(site_url, str) or not site_url.strip():
            return _failure("site_url_missing", f"instance_{index}")
        try:
            bound = site_catalog.bind(
                benchmark=ROCKET_CHAT_BENCHMARK,
                site=ROCKET_CHAT_SITE,
                origin=site_url,
            )
        except Exception:
            return _failure("site_binding_failed", f"instance_{index}")
        if not bound.supports_readback_observation():
            return _failure("painted_readback_unconfigured", f"instance_{index}")

    return Phase2RuntimeAdmission(
        True,
        "explicit_reset_reader_and_painted_readback_contracts_admitted",
        ("reset_endpoint", "independent_reader", "painted_readback", "warp_local_evaluator"),
    )


__all__ = ["rocket_chat_phase2_admission"]
