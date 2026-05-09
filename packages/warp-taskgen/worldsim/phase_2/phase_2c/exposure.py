"""Phase 2c exposure and reachability projection behavior."""

from __future__ import annotations

from typing import Any


def _phase4_exposure_inadmissible_reason(contract: Any) -> str | None:
    if not isinstance(contract, dict):
        return None
    capability = contract.get("phase4_exposure")
    if not isinstance(capability, dict):
        return None
    if capability.get("admissible") is True:
        return None
    reason = capability.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return "inadmissible"


def _reachability_resource_for_task(
    task: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project an exposure contract into the existing reachability probe shape."""
    resource = task.get("benign_target_resource")
    projected = dict(resource) if isinstance(resource, dict) else {}
    contract = task.get("exposure_contract")
    if not isinstance(contract, dict):
        return projected or None
    verification = contract.get("verification")
    verification_url = (
        verification.get("url")
        if isinstance(verification, dict)
        else contract.get("benign_read_url")
    )
    if isinstance(verification_url, str) and verification_url.strip():
        projected["start_url_resolved"] = verification_url
    target_url = _verification_target_url(contract, metadata or {})
    if target_url:
        projected["exposure_target_url"] = target_url
    kind = contract.get("kind")
    if isinstance(kind, str) and kind.strip():
        projected["kind"] = kind
    if "anchors" not in projected and isinstance(contract.get("anchors"), dict):
        projected["anchors"] = dict(contract["anchors"])
    projected["exposure_contract_id"] = contract.get("contract_id")
    projected["exposure_mode"] = contract.get("mode")
    return projected


def _verification_target_url(contract: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    verification = contract.get("verification")
    if not isinstance(verification, dict):
        return None
    target = verification.get("target")
    if not isinstance(target, dict):
        return None
    direct = target.get("url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    source = target.get("url_source")
    if not isinstance(source, str) or not source.startswith("seed_metadata."):
        return None
    key = source.removeprefix("seed_metadata.")
    value = _metadata_path_value(metadata, key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _metadata_path_value(metadata: dict[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list):
            selected: Any = None
            if part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    selected = current[index]
            else:
                for item in current:
                    if isinstance(item, dict) and item.get("role") == part:
                        selected = item
                        break
            current = selected
            continue
        return None
    return current


def _required_url_token(task: dict[str, Any]) -> str | None:
    """Return the first URL value from task.required_tokens, or None."""
    tokens = task.get("required_tokens")
    if not isinstance(tokens, list):
        return None
    for entry in tokens:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "url":
            continue
        value = entry.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_rendered_payload(seed: dict[str, Any]) -> str | None:
    """Extract the longest rendered payload string from a seed's editor_calls.

    Used as the source pool for the reachability probe's second witness.
    Prefers the longest arg value across all calls so an unsubstituted
    short ``{benign_*}`` token does not win over the actual body text.
    """
    if not isinstance(seed, dict):
        return None
    calls = seed.get("editor_calls")
    if not isinstance(calls, list):
        return None
    best: str | None = None
    for call in calls:
        if not isinstance(call, dict):
            continue
        args = call.get("args")
        if not isinstance(args, dict):
            continue
        for value in args.values():
            if isinstance(value, str) and len(value) >= 20:
                if best is None or len(value) > len(best):
                    best = value
    return best


def _selected_rendered_payload(task: dict[str, Any]) -> str | None:
    """Return the selected Phase 2b rendered payload when present."""
    payloads = task.get("payload_texts")
    if not isinstance(payloads, list) or not payloads:
        return None
    raw_index = task.get("selected_payload_index", 0)
    try:
        selected_index = int(raw_index)
    except (TypeError, ValueError):
        selected_index = 0
    candidates: list[Any] = []
    if 0 <= selected_index < len(payloads):
        candidates.append(payloads[selected_index])
    candidates.extend(payloads)
    for payload in candidates:
        if not isinstance(payload, dict):
            continue
        for key in ("rendered_payload", "raw_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None

__all__ = [
    "_first_rendered_payload",
    "_metadata_path_value",
    "_phase4_exposure_inadmissible_reason",
    "_reachability_resource_for_task",
    "_required_url_token",
    "_selected_rendered_payload",
    "_verification_target_url",
]
