"""Phase 2 target-resolution input preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.auth_tokens import acquire_tokens_for_instances
from warp_taskgen.phase_2.exposure_contract import (
    signature_hash as exposure_contract_signature_hash,
)
from warp_taskgen.phase_2.phase_2c.config import _extract_instances_list
from warp_taskgen.phase_2.target_resolution.constants import (
    PHASE_2A_SYNTHETIC_PLACEHOLDERS as _PHASE_2A_SYNTHETIC_PLACEHOLDERS,
)
from warp_taskgen.phase_2.target_resolution.runner import derive_benign_target_resource
from warp_taskgen.sites import default_catalog

logger = logging.getLogger(__name__)


def _l1_l2_resources_dict(
    site_tasks: list[dict],
    *,
    benchmark: str = "webarena_verified",
) -> dict[str, dict[str, Any]]:
    return {
        str(task.get("id")): derive_benign_target_resource(
            task,
            _PHASE_2A_SYNTHETIC_PLACEHOLDERS,
            benchmark=benchmark,
        )
        for task in site_tasks
    }


def _phase_2a_resolution_signature(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint the live inputs that affect Phase 2a L3/L4 output."""
    instances_arg = getattr(args, "feasibility_instances", None)
    signature: dict[str, Any] = {
        "no_l3_l4": bool(getattr(args, "no_l3_l4", False)),
        "instances_path": str(instances_arg) if instances_arg else None,
        "instances_sha256": None,
        "exposure_contract_signature": exposure_contract_signature_hash(),
    }
    if not instances_arg:
        return signature
    path = Path(instances_arg)
    try:
        payload = path.read_text()
    except OSError:
        signature["instances_missing"] = True
        return signature
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        signature["instances_unparseable"] = True
        signature["instances_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return signature
    projected = _project_phase_2a_resolution_inputs(raw)
    canonical = json.dumps(projected, sort_keys=True, separators=(",", ":"))
    signature["instances_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return signature


def _project_phase_2a_resolution_inputs(payload: Any) -> list[dict[str, Any]]:
    """Keep only the benign-probe inputs that can change L3/L4 output."""
    effective_by_site: dict[str, dict[str, Any]] = {}
    for instance in _extract_instances_list(payload):
        site_name = str(instance.get("site_name", "")).strip().lower()
        if not site_name:
            continue
        entry: dict[str, Any] = {
            "site_name": site_name,
            "site_url": str(instance.get("site_url", "")).strip(),
            "probe_auth_mode": (
                "api_auth_only" if _instance_lacks_benign_probe_auth(instance) else "benign_auth"
            ),
        }
        auth = instance.get("auth")
        if isinstance(auth, dict):
            entry["auth"] = _phase_2a_auth_identity(auth)
        effective_by_site[site_name] = entry
    projected = list(effective_by_site.values())
    projected.sort(key=lambda item: item["site_name"])
    return projected


def _phase_2a_auth_identity(auth: Mapping[str, Any]) -> dict[str, Any]:
    auth_type = str(auth.get("type", "")).strip()
    identity: dict[str, Any] = {"type": auth_type}
    if auth_type == "http_headers":
        headers = auth.get("headers")
        if isinstance(headers, dict):
            normalized: dict[str, Any] = {}
            for key, value in sorted(headers.items()):
                key_str = str(key)
                if isinstance(value, str):
                    normalized[key_str] = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
                    continue
                if isinstance(value, dict) and isinstance(value.get("from_env"), str):
                    env_name = value["from_env"].strip()
                    resolved = os.environ.get(env_name, "")
                    normalized[key_str] = {
                        "from_env": env_name,
                        "value_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12],
                    }
            identity["headers"] = normalized
        return identity
    if auth_type == "web_login":
        credentials = auth.get("credentials")
        if isinstance(credentials, dict):
            identity["credentials_sha256"] = hashlib.sha256(
                json.dumps(credentials, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
        login_url = auth.get("login_url")
        if isinstance(login_url, str) and login_url.strip():
            identity["login_url"] = login_url.strip()
        return identity
    if auth_type == "bearer_token":
        from warp_taskgen.auth_tokens import _cache_identity

        return _cache_identity(dict(auth))
    return identity


def _load_phase_2a_instance_by_site(
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]] | None:
    """Build a ``{site_name: instance}`` map for Phase 2a L3/L4 enrichment.

    Reuses Phase 2c's ``--feasibility-instances`` flag — a single source
    of truth for "which live benchmark are we hitting" across the two
    stages. Returns ``None`` when the flag is absent, the file doesn't
    exist, ``--no-l3-l4`` is set, or the wrapper file carries no
    instances — in every such case Phase 2a falls back to the legacy
    L1/L2-only synchronous derive_benign_target_resource path.

    This helper is read-only and cheap; token acquisition for L3/L4
    probes is deferred to the call site in commit 4 (see
    :func:`warp_taskgen.auth_tokens.acquire_tokens_for_instances`).
    """
    if getattr(args, "no_l3_l4", False):
        return None
    instances_arg = getattr(args, "feasibility_instances", None)
    if not instances_arg:
        return None
    instances_path = Path(instances_arg)
    if not instances_path.exists():
        return None
    try:
        raw = json.loads(instances_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Phase 2a: could not parse %s for L3/L4 enrichment: %s", instances_path, exc)
        return None
    instances = _extract_instances_list(raw)
    if not instances:
        return None
    by_site: dict[str, dict[str, Any]] = {}
    for inst in instances:
        name = str(inst.get("site_name", "")).strip().lower()
        if not name:
            continue
        by_site[name] = inst
    return by_site or None


def _instance_bearer_tokens_ready(instance: Mapping[str, Any] | None) -> bool:
    if instance is None:
        return True
    auth = instance.get("auth")
    if isinstance(auth, dict) and str(auth.get("type", "")).strip() == "bearer_token":
        token = auth.get("token")
        if not isinstance(token, str) or not token.strip():
            return False
    return True


def _warm_phase_2a_instance_tokens(instance_by_site: Mapping[str, Any] | None) -> None:
    if not instance_by_site:
        return
    pending = [
        instance
        for instance in instance_by_site.values()
        if isinstance(instance, dict) and not _instance_bearer_tokens_ready(instance)
    ]
    if not pending:
        return
    errors = acquire_tokens_for_instances(pending, auth_fields=("auth",))
    if errors:
        logger.warning(
            "Phase 2a: token warmup failed for %d site(s): %s",
            len(errors),
            "; ".join(errors),
        )


def _instance_lacks_benign_probe_auth(instance: Mapping[str, Any] | None) -> bool:
    if instance is None:
        return False
    auth = instance.get("auth")
    api_auth = instance.get("api_auth")
    return isinstance(api_auth, dict) and not isinstance(auth, dict)


def _mark_probe_dependent_resources_unresolved(
    resources: dict[str, dict[str, Any]],
    *,
    reason: str,
    benchmark: str = "webarena_verified",
) -> dict[str, dict[str, Any]]:
    for task_id, record in resources.items():
        kind = record.get("kind")
        if record.get("pending_layer") == "L3":
            resources[task_id] = {
                "kind": None,
                "anchors": dict(record.get("anchors") or {}),
                "start_url_resolved": record.get("start_url_resolved"),
                "attach_surfaces": [],
                "encounter_requirements": record.get("encounter_requirements")
                or {"viewport_budget_chars": 600},
                "layer": record.get("layer"),
                "pending_layer": "L3",
                "reason": reason,
            }
            continue
        if isinstance(kind, str) and default_catalog().is_expandable_listing_kind(
            kind,
            benchmark=benchmark,
        ):
            resources[task_id] = {
                "kind": None,
                "anchors": dict(record.get("anchors") or {}),
                "start_url_resolved": record.get("start_url_resolved"),
                "attach_surfaces": [],
                "encounter_requirements": record.get("encounter_requirements")
                or {"viewport_budget_chars": 600},
                "layer": record.get("layer"),
                "pending_layer": "L4",
                "reason": reason,
            }
    return resources


def _l1_l2_resources_with_probe_fail_closed(
    site_tasks: list[dict[str, Any]],
    *,
    reason: str,
    benchmark: str = "webarena_verified",
) -> dict[str, dict[str, Any]]:
    return _mark_probe_dependent_resources_unresolved(
        _l1_l2_resources_dict(site_tasks, benchmark=benchmark),
        reason=reason,
        benchmark=benchmark,
    )
