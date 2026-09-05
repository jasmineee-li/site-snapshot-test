#!/usr/bin/env python3
"""Run the two genuine Phase 1 provider-boundary micro-canaries serially."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from warp_taskgen.modal_sandbox import (
    LEGACY_NAMED_SECRET_ENV_VAR,
    NAMED_SECRET_ENV_VAR,
    preflight_sandbox_environment,
    upload_to_volume,
)
from warp_taskgen.phases.phase_1_generate_new_tasks import (
    EligibleSiteProfile,
    SiteGenerateNewTasksResult,
    _use_contract_bound_action_api,
    compute_generate_new_tasks_shared_inputs_fingerprint,
    compute_site_cache_fingerprint,
    generate_new_tasks_for_site,
)
from warp_taskgen.phases.phase_1_task_cards import (
    load_task_card_plan,
    task_card_plan_digest,
    validate_task_card_plan,
)
from warp_taskgen.profile_validation import load_and_validate_profile
from warp_taskgen.state import bind_state_paths

MODEL = "claude-sonnet-4-6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_CURRENT_KEY_URL = f"{OPENROUTER_BASE_URL}/v1/key"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_LOGS_ROOT = PACKAGE_ROOT / "logs"


@dataclass(frozen=True)
class SourceInputs:
    source_run: Path
    benchmark_root: Path
    manifest: dict[str, Any]
    site: EligibleSiteProfile
    task_card_plan: dict[str, Any]


class MicrocanaryError(RuntimeError):
    """A fail-closed provider micro-canary invariant failed."""

    def __init__(self, message: str, *, code: str = "microcanary_invariant_failed") -> None:
        super().__init__(message)
        self.code = code


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MicrocanaryError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise MicrocanaryError(f"{label} must contain a JSON object: {path}")
    return payload


def _definition_inputs(state: dict[str, Any]) -> dict[str, Any]:
    envelope = state.get("run_definition")
    if not isinstance(envelope, dict):
        raise MicrocanaryError("source Run has no persisted Run Definition")
    contributions = envelope.get("contributions")
    if not isinstance(contributions, dict):
        raise MicrocanaryError("source Run Definition has no contributions")
    projected: dict[str, Any] = {}
    for values in contributions.values():
        if isinstance(values, dict):
            projected.update(values)
    return projected


def _required_path(inputs: dict[str, Any], field: str, *, kind: str) -> Path:
    value = inputs.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MicrocanaryError(f"source Run Definition is missing {field}")
    path = Path(value).expanduser().resolve()
    exists = path.is_dir() if kind == "directory" else path.is_file()
    if not exists:
        raise MicrocanaryError(f"source Run {field} is not a readable {kind}: {path}")
    return path


def load_source_inputs(source_run: Path) -> SourceInputs:
    source_run = source_run.expanduser().resolve()
    if not source_run.is_dir():
        raise MicrocanaryError(f"source Run is not a directory: {source_run}")
    state = _load_json_object(source_run / "pipeline_state.json", label="pipeline state")
    inputs = _definition_inputs(state)
    if inputs.get("sandbox_model") != MODEL:
        raise MicrocanaryError(f"source Run sandbox model must be {MODEL}")

    benchmark_root = _required_path(inputs, "benchmark_path", kind="directory")
    manifest_path = _required_path(inputs, "manifest_path", kind="file")
    task_card_plan_path = _required_path(inputs, "task_card_plan_path", kind="file")
    manifest = _load_json_object(manifest_path, label="benchmark manifest")
    task_card_plan = load_task_card_plan(task_card_plan_path)
    if task_card_plan is None:
        raise MicrocanaryError("source Run task-card plan is missing")
    observed_digest = task_card_plan_digest(task_card_plan)
    expected_digest = state.get("task_card_plan_digest")
    if not isinstance(expected_digest, str) or observed_digest != expected_digest:
        raise MicrocanaryError("source Run task-card plan digest does not match pipeline state")

    profile_path = source_run / "phase_0c" / "BENCHMARK_PROFILE_gitlab.json"
    context_path = profile_path.with_name("AGENT_CONTEXT_gitlab.json")
    if not profile_path.is_file():
        raise MicrocanaryError(f"source Run GitLab profile is missing: {profile_path}")
    _load_json_object(context_path, label="GitLab agent context")
    eval_types = manifest.get("evaluation", {}).get("eval_types", [])
    if not isinstance(eval_types, list):
        raise MicrocanaryError("benchmark manifest evaluation.eval_types must be an array")
    profile = load_and_validate_profile(
        "gitlab",
        profile_path,
        manifest_eval_types=eval_types,
    )
    return SourceInputs(
        source_run=source_run,
        benchmark_root=benchmark_root,
        manifest=manifest,
        site=EligibleSiteProfile("gitlab", profile_path, profile),
        task_card_plan=task_card_plan,
    )


def select_one_card(plan: dict[str, Any], card_id: str) -> dict[str, Any]:
    matches = [
        card
        for card in plan.get("task_cards", [])
        if isinstance(card, dict)
        and card.get("id") == card_id
        and str(card.get("status", "active")) == "active"
    ]
    if len(matches) != 1:
        raise MicrocanaryError(f"expected exactly one active task card named {card_id!r}")
    card = copy.deepcopy(matches[0])
    if card.get("site") != "gitlab":
        raise MicrocanaryError(f"task card {card_id!r} must target gitlab")
    card["generation_count"] = 1
    selected = {key: copy.deepcopy(value) for key, value in plan.items() if key != "task_cards"}
    selected["task_cards"] = [card]
    validate_task_card_plan(selected)
    return selected


def validate_frozen_route(env: dict[str, str] | os._Environ[str]) -> str:
    normalized_base = env.get("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if normalized_base != OPENROUTER_BASE_URL:
        raise MicrocanaryError(
            f"ANTHROPIC_BASE_URL must normalize to {OPENROUTER_BASE_URL}",
            code="route_configuration_invalid",
        )
    token = env.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    if not token:
        raise MicrocanaryError(
            "ANTHROPIC_AUTH_TOKEN must be present",
            code="route_configuration_invalid",
        )
    backend_opt_in = env.get("WORLDSIM_PHASE1_CONTRACT_BOUND_API", "").strip().lower()
    if backend_opt_in not in {"1", "true", "yes", "on"}:
        raise MicrocanaryError(
            "WORLDSIM_PHASE1_CONTRACT_BOUND_API must enable the retained Run route",
            code="route_configuration_invalid",
        )
    forbidden = [
        name
        for name in (
            "CLAUDE_CODE_OAUTH_TOKEN",
            NAMED_SECRET_ENV_VAR,
            LEGACY_NAMED_SECRET_ENV_VAR,
        )
        if env.get(name, "").strip()
    ]
    if forbidden:
        raise MicrocanaryError(
            "higher-precedence OAuth or named Modal secret must be absent: " + ", ".join(forbidden),
            code="route_configuration_invalid",
        )
    return token


def check_openrouter_capacity(token: str) -> None:
    request = urllib.request.Request(
        OPENROUTER_CURRENT_KEY_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = getattr(response, "status", None)
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise MicrocanaryError(
            f"OpenRouter current-key preflight returned HTTP {exc.code}",
            code="capacity_http_error",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise MicrocanaryError(
            "OpenRouter current-key preflight returned no usable response",
            code="capacity_response_invalid",
        ) from exc
    if status != 200 or not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise MicrocanaryError(
            "OpenRouter current-key preflight returned an unknown response",
            code="capacity_response_invalid",
        )
    remaining = payload["data"].get("limit_remaining")
    if (
        isinstance(remaining, bool)
        or not isinstance(remaining, (int, float))
        or not math.isfinite(remaining)
    ):
        raise MicrocanaryError(
            "OpenRouter current-key preflight omitted numeric remaining capacity",
            code="capacity_response_invalid",
        )
    if remaining <= 0:
        raise MicrocanaryError(
            "OpenRouter current-key preflight reports no usable capacity",
            code="capacity_unavailable",
        )


def validate_fresh_output_root(output_root: Path) -> Path:
    root = output_root.expanduser().resolve()
    logs_root = PACKAGE_LOGS_ROOT.resolve()
    if root == logs_root or logs_root not in root.parents:
        raise MicrocanaryError(
            f"output root must be a child of ignored logs: {logs_root}",
            code="output_root_invalid",
        )
    if root.exists():
        raise MicrocanaryError(
            f"output root must be fresh: {root}",
            code="output_root_invalid",
        )
    return root


def _require_one_row(
    result: SiteGenerateNewTasksResult,
    *,
    boundary: str,
    output_dir: Path,
) -> None:
    if result.errors:
        raise MicrocanaryError(
            f"{boundary} provider boundary returned {len(result.errors)} error(s); "
            f"inspect {output_dir}",
            code=f"{boundary}_generation_failed",
        )
    if len(result.benign_tasks) != 1:
        raise MicrocanaryError(
            f"{boundary} provider boundary returned {len(result.benign_tasks)} rows; expected 1",
            code=f"{boundary}_output_invalid",
        )


def _bounded_text(value: object, *, limit: int = 240) -> str:
    return " ".join(str(value).split())[:limit]


def _diagnostic(
    *,
    boundary: str,
    status: str,
    path: Path | None = None,
    error_code: str | None = None,
    stream: TextIO | None = None,
) -> None:
    fields = {
        "timestamp": datetime.now(UTC).isoformat(),
        "route": "openrouter-anthropic-compatible",
        "model": MODEL,
        "card": _bounded_text(boundary),
        "status": status,
    }
    if path is not None:
        fields["path"] = _bounded_text(path)
    if error_code is not None:
        fields["error_code"] = error_code
    print(json.dumps(fields, sort_keys=True), file=stream, flush=True)


async def run_microcanaries(
    *,
    source_run: Path,
    output_root: Path,
    direct_card_id: str,
    sandbox_card_id: str,
) -> None:
    root = validate_fresh_output_root(output_root)
    source = load_source_inputs(source_run)
    direct_plan = select_one_card(source.task_card_plan, direct_card_id)
    sandbox_plan = select_one_card(source.task_card_plan, sandbox_card_id)
    if not _use_contract_bound_action_api(direct_plan):
        raise MicrocanaryError(
            "direct card does not resolve to the production contract-bound backend",
            code="direct_route_invalid",
        )
    if _use_contract_bound_action_api(sandbox_plan):
        raise MicrocanaryError(
            "sandbox card resolves to the production contract-bound backend",
            code="sandbox_route_invalid",
        )
    token = validate_frozen_route(os.environ)
    await asyncio.to_thread(check_openrouter_capacity, token)
    _diagnostic(boundary="capacity", status="usable")

    direct_output = root / "phase_1" / "direct"
    sandbox_output = root / "phase_1" / "sandbox"
    direct_output.mkdir(parents=True)
    with bind_state_paths(root, resume_pointer=root / "last_run_state.json"):
        direct_shared = compute_generate_new_tasks_shared_inputs_fingerprint(
            benchmark_root=source.benchmark_root,
            manifest=source.manifest,
            sandbox_model=MODEL,
            task_card_plan=direct_plan,
        )
        direct_result = await generate_new_tasks_for_site(
            site=source.site,
            benchmark_volume=None,
            output_dir=direct_output,
            cache_fingerprint=compute_site_cache_fingerprint(
                shared_inputs_fingerprint=direct_shared,
                site=source.site,
                novel_tasks_per_site=1,
                task_card_plan=direct_plan,
            ),
            sandbox_model=MODEL,
            novel_tasks_per_site=1,
            task_card_plan=direct_plan,
            _allow_task_card_slicing=False,
        )
        _require_one_row(direct_result, boundary="direct", output_dir=direct_output)
        _diagnostic(boundary=direct_card_id, status="succeeded", path=direct_output)

        sandbox_output.mkdir(parents=True)
        await preflight_sandbox_environment()
        benchmark_volume = await upload_to_volume(source.benchmark_root)
        sandbox_shared = compute_generate_new_tasks_shared_inputs_fingerprint(
            benchmark_root=source.benchmark_root,
            manifest=source.manifest,
            sandbox_model=MODEL,
            task_card_plan=sandbox_plan,
        )
        sandbox_result = await generate_new_tasks_for_site(
            site=source.site,
            benchmark_volume=benchmark_volume,
            output_dir=sandbox_output,
            cache_fingerprint=compute_site_cache_fingerprint(
                shared_inputs_fingerprint=sandbox_shared,
                site=source.site,
                novel_tasks_per_site=1,
                task_card_plan=sandbox_plan,
            ),
            sandbox_model=MODEL,
            novel_tasks_per_site=1,
            task_card_plan=sandbox_plan,
            _allow_task_card_slicing=False,
        )
        _require_one_row(sandbox_result, boundary="sandbox", output_dir=sandbox_output)
        _diagnostic(boundary=sandbox_card_id, status="succeeded", path=sandbox_output)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--direct-card-id", required=True)
    parser.add_argument("--sandbox-card-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        asyncio.run(
            run_microcanaries(
                source_run=args.source_run,
                output_root=args.output_root,
                direct_card_id=args.direct_card_id,
                sandbox_card_id=args.sandbox_card_id,
            )
        )
    except Exception as exc:
        error_code = (
            exc.code if isinstance(exc, MicrocanaryError) else "unexpected_microcanary_failure"
        )
        _diagnostic(
            boundary="all",
            status="failed",
            path=args.output_root.resolve(),
            error_code=error_code,
            stream=sys.stderr,
        )
        return 1
    _diagnostic(boundary="all", status="succeeded", path=args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
