"""Phase 0c: staging benchmark files into a tier sandbox and running the tier.

Owns the staging directory, the Modal Sandbox call for one tier, the correction
block rendered from validation feedback, the per-tier success publisher, and the
bounded correction retry loop.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.modal_sandbox import run_claude_in_sandbox
from warp_taskgen.phases.phase_0c_artifacts import (
    Phase0cTraceWriter,
    publish_tier_output,
    redact_json_secrets,
)
from warp_taskgen.phases.phase_0c_profile_reuse import _render_tier_prompt
from warp_taskgen.prompt_corrections import render_validation_feedback

logger = logging.getLogger(__name__)


# Maximum number of correction retries for profile validation (initial attempt + this many).
PROFILE_FIX_MAX_ITERATIONS = 2


def _stage_benchmark_files(
    file_list: list[str],
    benchmark_root: Path,
    site_name: str,
) -> tuple[Path, Path]:
    """Stage benchmark files into a temp dir for sandbox mounting.

    Returns ``(staging_root, staging_dir)`` where staging_dir is the inner
    "benchmark" directory suitable for mounting at ``/workspace/benchmark``.
    Caller is responsible for cleanup via ``shutil.rmtree(staging_root)``.
    """
    staging_root = Path(tempfile.mkdtemp(prefix=f"worldsim_0c_{site_name}_"))
    staging_dir = staging_root / "benchmark"
    staging_dir.mkdir()
    for local_path in file_list:
        rel = os.path.relpath(local_path, benchmark_root)
        staged = staging_dir / rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, staged)
    return staging_root, staging_dir


async def _run_tier_sandbox(
    *,
    site_name: str,
    site_files: dict[str, str],
    prompt: str,
    output_paths: list[str],
    timeout: int,
    label: str,
    sandbox_model: str,
    extra_inputs: dict[str, str] | None = None,
    volumes: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    """Run a single profiling tier sandbox with the standard pattern.

    Loads the prompt, appends validation footer, runs the sandbox, records
    cost, and returns raw outputs.
    """
    all_files = dict(site_files)
    if extra_inputs:
        all_files.update(extra_inputs)

    outputs = await run_claude_in_sandbox(
        site_files=all_files,
        prompt=prompt,
        output_paths=output_paths,
        timeout=timeout,
        model=sandbox_model,
        volumes=volumes,
        label=label,
    )
    cost_tracker.record("phase_0c", outputs.get("_summary"), site=site_name)
    return outputs


def _render_correction_block(
    *,
    site_name: str,
    artifact_name: str,
    errors: list[str],
    extra_guidance: str | None = None,
) -> str:
    """Return a reusable prompt suffix for retrying a failed tier output."""
    return render_validation_feedback(
        artifact_name=artifact_name,
        errors=[
            {
                "code": "VALIDATION_ERROR",
                "path": "$",
                "message": error,
            }
            for error in errors
        ],
        summary=f"{artifact_name} for site {site_name} failed validation.",
        instruction=(
            "Rewrite the output file completely so it satisfies the schema and all "
            "cross-reference checks. Do not include markdown or commentary."
        ),
        extra_guidance=extra_guidance,
    )


def _tier_success_publisher(
    *,
    output_dir: Path,
    site_name: str,
    tier_name: str,
    artifact_stem: str,
    output_path: str,
    metadata: dict[str, Any],
    sidecar_outputs: dict[str, str] | None = None,
    redact_values: tuple[str, ...] = (),
) -> Callable[[dict[str, str | None]], None]:
    def publish(outputs: dict[str, str | None]) -> None:
        raw = outputs.get(output_path)
        if not raw:
            return
        payload = json.loads(raw)
        sidecars: dict[str, object] = {}
        for side_output_path, sidecar_stem in (sidecar_outputs or {}).items():
            side_raw = outputs.get(side_output_path)
            if not side_raw:
                raise ValueError(f"{Path(side_output_path).name} was not produced")
            try:
                sidecars[sidecar_stem] = json.loads(side_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{Path(side_output_path).name} contained invalid JSON: {exc}"
                ) from exc
        publish_tier_output(
            output_dir=output_dir,
            site_name=site_name,
            tier_name=tier_name,
            artifact_stem=artifact_stem,
            payload=payload,
            metadata=metadata,
            sandbox_outputs=outputs,
            sidecars=sidecars,
            redact_values=redact_values,
        )

    return publish


async def _run_tier_json_with_retries(
    *,
    site_name: str,
    site_files: dict[str, str],
    prompt_name: str,
    validation_command: str,
    output_path: str,
    timeout: int,
    label: str,
    sandbox_model: str,
    validate_parsed: Callable[[object], list[str]],
    extra_inputs: dict[str, str] | None = None,
    correction_guidance: str | None = None,
    volumes: dict[str, Any] | None = None,
    side_output_paths: list[str] | None = None,
    on_success_outputs: Callable[[dict[str, str | None]], None] | None = None,
    redact_values: tuple[str, ...] = (),
    trace_writer: Phase0cTraceWriter | None = None,
    trace_context: dict[str, Any] | None = None,
) -> Any:
    """Run one profiling tier, retrying semantic validation failures in-place."""
    artifact_name = Path(output_path).name
    base_prompt = _render_tier_prompt(
        prompt_name=prompt_name,
        validation_command=validation_command,
        site_name=site_name,
    )
    prompt = base_prompt
    last_errors: list[str] = []

    for attempt in range(1 + PROFILE_FIX_MAX_ITERATIONS):
        attempt_label = label if attempt == 0 else f"{label}-retry{attempt}"
        if trace_writer is not None:
            trace_writer.record(
                "tier_attempt_started",
                site_name=site_name,
                label=attempt_label,
                attempt=attempt,
                output_path=output_path,
                **(trace_context or {}),
            )
        outputs = await _run_tier_sandbox(
            site_name=site_name,
            site_files=site_files,
            prompt=prompt,
            output_paths=[output_path, *list(side_output_paths or [])],
            timeout=timeout,
            label=attempt_label,
            sandbox_model=sandbox_model,
            extra_inputs=extra_inputs,
            volumes=volumes,
        )
        if trace_writer is not None:
            telemetry = outputs.get("_telemetry")
            trace_writer.record(
                "tier_attempt_finished",
                site_name=site_name,
                label=attempt_label,
                attempt=attempt,
                output_path=output_path,
                telemetry=telemetry,
                **(trace_context or {}),
            )

        raw = outputs.get(output_path)
        parsed: object | None = None
        errors: list[str] = []
        if not raw:
            errors.append(f"{artifact_name} was not produced")
        else:
            try:
                parsed = redact_json_secrets(
                    json.loads(raw),
                    redact_values=redact_values,
                )
            except json.JSONDecodeError as exc:
                errors.append(f"{artifact_name} contained invalid JSON: {exc}")

        for side_output_path in side_output_paths or []:
            side_name = Path(side_output_path).name
            side_raw = outputs.get(side_output_path)
            if not side_raw:
                errors.append(f"{side_name} was not produced")
                continue
            try:
                json.loads(side_raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{side_name} contained invalid JSON: {exc}")

        if not errors and parsed is not None:
            errors.extend(validate_parsed(parsed))

        if not errors:
            if on_success_outputs is not None:
                try:
                    on_success_outputs(outputs)
                except ValueError as exc:
                    errors.append(str(exc))
            if not errors:
                if trace_writer is not None:
                    trace_writer.record(
                        "tier_generated",
                        site_name=site_name,
                        label=attempt_label,
                        attempt=attempt,
                        output_path=output_path,
                        **(trace_context or {}),
                    )
                return parsed

        last_errors = errors
        if trace_writer is not None:
            trace_writer.record(
                "tier_validation_failed",
                site_name=site_name,
                label=attempt_label,
                attempt=attempt,
                output_path=output_path,
                errors=errors,
                **(trace_context or {}),
            )
        if attempt < PROFILE_FIX_MAX_ITERATIONS:
            logger.warning(
                "Phase 0c: site %r %s failed validation, retrying (%d/%d): %s",
                site_name,
                artifact_name,
                attempt + 1,
                PROFILE_FIX_MAX_ITERATIONS,
                "; ".join(errors),
            )
            prompt = base_prompt + _render_correction_block(
                site_name=site_name,
                artifact_name=artifact_name,
                errors=errors,
                extra_guidance=correction_guidance,
            )

    if trace_writer is not None:
        trace_writer.record(
            "tier_failed",
            site_name=site_name,
            label=label,
            output_path=output_path,
            errors=last_errors,
            **(trace_context or {}),
        )
    raise RuntimeError(
        f"{artifact_name} for site {site_name} failed validation:\n"
        + "\n".join(f"  - {error}" for error in last_errors)
    )
