from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from warp_taskgen.phase_2.text_fill.api import (
    _call_text_fill_api,
    _parse_text_fill_response,
    is_refusal,
)
from warp_taskgen.phase_2.text_fill.api_errors import TextFillAPIError
from warp_taskgen.phase_2.text_fill.constants import TEXT_FILL_PROMPT_VARIANTS
from warp_taskgen.phase_2.text_fill.prompt import render_fill_prompt
from warp_taskgen.phase_2.text_fill.seed import materialize_adversarial_seed
from warp_taskgen.phase_2.text_fill.validation import validate_text_post_hoc
from warp_taskgen.phase_2.text_fill.voice import (
    _select_exemplars,
    load_voice_registry,
    resolve_site_kind,
)


async def fill_texts_for_tasks(
    tasks: list[dict[str, Any]],
    *,
    texts_per_plan: int,
    concurrency: int,
    model: str,
    registry_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not tasks:
        return ([], [])
    registry = load_voice_registry(registry_path)
    limiter = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(task: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        async with limiter:
            return await _fill_one_task(
                task,
                registry=registry,
                texts_per_plan=texts_per_plan,
                model=model,
            )

    results = await asyncio.gather(*(_run_one(task) for task in tasks))
    filled: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for task, diag in results:
        diagnostics.append(diag)
        if task is not None:
            filled.append(task)
    return (filled, diagnostics)


async def _fill_one_task(
    task: dict[str, Any],
    *,
    registry: dict[str, Any],
    texts_per_plan: int,
    model: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    existing = task.get("payload_texts")
    payloads: list[dict[str, Any]]
    attempts: list[dict[str, Any]] = []
    if isinstance(existing, list) and len(existing) >= texts_per_plan:
        payloads = [json.loads(json.dumps(item)) for item in existing[:texts_per_plan]]
        cache_errors = _validate_cached_payloads(task, payloads)
        if not cache_errors:
            diag = {
                "task_id": task.get("id"),
                "site": task.get("site"),
                "status": "reused_existing",
                "attempts": attempts,
                "texts_generated": len(payloads),
            }
            finalized = _finalize_task_payloads(task, payloads, diag)
            return (finalized, diag)
        attempts.extend(cache_errors)

    payloads = []
    for ordinal in range(texts_per_plan):
        payload, diag = await _generate_single_payload(
            task,
            registry=registry,
            model=model,
            ordinal=ordinal,
        )
        attempts.append(diag)
        if payload is None:
            summary = {
                "task_id": task.get("id"),
                "site": task.get("site"),
                "status": "text_unrecoverable",
                "attempts": attempts,
                "texts_generated": len(payloads),
            }
            return (None, summary)
        payloads.append(payload)

    summary = {
        "task_id": task.get("id"),
        "site": task.get("site"),
        "status": "ok",
        "attempts": attempts,
        "texts_generated": len(payloads),
    }
    finalized = _finalize_task_payloads(task, payloads, summary)
    return (finalized, summary)


async def _generate_single_payload(
    task: dict[str, Any],
    *,
    registry: dict[str, Any],
    model: str,
    ordinal: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    site_kind = resolve_site_kind(
        registry,
        site=str(task.get("site", "")),
        target_surface_id=str(task.get("target_surface_id", "")),
        source_field=task.get("source_field"),
    )
    exemplars = _select_exemplars(
        registry,
        site_kind=site_kind,
        framing=str(task.get("framing", "")),
        seed_material=f"{task.get('id', '')}:{ordinal}",
    )
    prompts = [
        (
            variant,
            render_fill_prompt(task, exemplars=exemplars, site_kind=site_kind, variant=variant),
        )
        for variant in TEXT_FILL_PROMPT_VARIANTS
    ]
    errors: list[dict[str, Any]] = []

    for prompt_variant, prompt in prompts:
        try:
            call_result = await _call_text_fill_api(prompt, model, task=task)
            if len(call_result) == 3:
                raw_or_parsed, auth_path, api_diagnostics = call_result
            else:
                raw_or_parsed, auth_path = call_result
                api_diagnostics = None
        except TextFillAPIError as exc:
            error: dict[str, Any] = {
                "variant": prompt_variant,
                "auth_path": "instructor_anthropic",
                "error": str(exc),
            }
            if exc.diagnostics:
                error["api_diagnostics"] = exc.diagnostics
            errors.append(error)
            continue
        except Exception as exc:  # pragma: no cover - network path exercised via mocks
            errors.append({"variant": prompt_variant, "auth_path": "shared_api", "error": str(exc)})
            continue
        if isinstance(raw_or_parsed, dict):
            parsed = raw_or_parsed
            parse_error = None
        else:
            raw = str(raw_or_parsed)
            if is_refusal(raw):
                errors.append(
                    {"variant": prompt_variant, "auth_path": auth_path, "error": "refused"}
                )
                continue
            parsed, parse_error = _parse_text_fill_response(raw)
        if parse_error is None:
            validation_errors = validate_text_post_hoc(parsed, task)
            if not validation_errors:
                parsed["auth_path"] = auth_path
                parsed["attempt"] = prompt_variant
                diag: dict[str, Any] = {"status": "ok", "errors": errors}
                if api_diagnostics is not None:
                    diag["api_diagnostics"] = api_diagnostics
                return parsed, diag
            errors.append(
                {
                    "variant": prompt_variant,
                    "auth_path": auth_path,
                    "error": "post_hoc_failed",
                    "details": validation_errors,
                }
            )
        else:
            errors.append(
                {
                    "variant": prompt_variant,
                    "auth_path": auth_path,
                    "error": "parse_failed",
                    "details": parse_error,
                }
            )

    return (None, {"status": "failed", "errors": errors})


def _validate_cached_payloads(
    task: dict[str, Any],
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for ordinal, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            errors.append(
                {
                    "variant": "cached",
                    "ordinal": ordinal,
                    "error": "cached_payload_invalid",
                    "details": ["cached payload must be an object"],
                }
            )
            continue
        validation_errors = validate_text_post_hoc(payload, task)
        if validation_errors:
            errors.append(
                {
                    "variant": "cached",
                    "ordinal": ordinal,
                    "error": "cached_payload_invalid",
                    "details": validation_errors,
                }
            )
    return errors


def _finalize_task_payloads(
    task: dict[str, Any],
    payloads: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    finalized = json.loads(json.dumps(task))
    finalized["payload_texts"] = payloads
    finalized["selected_payload_index"] = 0
    if diagnostics.get("status") == "reused_existing":
        # Preserve original generation evidence (or its historical absence).
        finalized["payload_text_reuse_diagnostics"] = diagnostics
    else:
        finalized["payload_text_diagnostics"] = diagnostics
    selected = payloads[0]
    finalized["adversarial_data_seed"] = materialize_adversarial_seed(
        finalized["seed_template"],
        str(selected["rendered_payload"]),
    )
    return finalized
