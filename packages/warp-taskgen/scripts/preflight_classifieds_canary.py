#!/usr/bin/env python3
"""Fail-closed, host-owned preflight for the Classifieds live canary.

This is deliberately an evidence checker, not another Taskgen/runtime layer.
It reads the artifacts produced by preparation, Phase 2c, Phase 3, and the
anonymous reset probe, then writes a secret-free ``preflight.json`` only when
the exact one-task canary contract is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from warp_taskgen.classifieds_canary import (
    CLASSIFIEDS_COMPOSE_PROJECT,
    CLASSIFIEDS_DB_IMAGE,
    CLASSIFIEDS_DB_MANIFEST_DIGEST,
    CLASSIFIEDS_SOURCE_COMMIT,
    CLASSIFIEDS_WEB_IMAGE,
    CLASSIFIEDS_WEB_MANIFEST_DIGEST,
    build_compose_overlay_from_values,
    canary_body_for_run,
    validate_classifieds_loopback_origin,
)
from warp_taskgen.phase_2.phase_2c.checkpoints import (
    CheckpointValidationError,
    Phase2cCheckpointContext,
    task_fingerprint,
    validate_checkpoint_payload,
)
from warp_taskgen.phase_2.phase_2c.fingerprints import _host_fingerprint, _task_content_hash
from warp_taskgen.phase_4.exposure_admission import exposure_admission_error
from warp_taskgen.profile_validation import load_and_validate_profile, profile_requires_agent_auth
from warp_taskgen.run_definition import define_run
from warp_taskgen.sites.classifieds_reply_html import normalize_reply_body

EXPECTED_BENCHMARK = "visualwebarena"
EXPECTED_SITE = "classifieds"
EXPECTED_SURFACE = "listing_reply.body"
EXPECTED_PROFILE = "classifieds_listing_reply_poc"
EXPECTED_VARIANT_SYSTEM = "eval-awareness-iterator"
EXPECTED_AGENT_PROVIDER = "anthropic"
EXPECTED_AGENT_MODEL = "claude-sonnet-4-6"
EXPECTED_RUNNER = "browser_use"
EXPECTED_WEB_IMAGE = "ghcr.io/bgrins/vwa_classifieds_web"
EXPECTED_DB_IMAGE = "ghcr.io/bgrins/vwa_classifieds_db"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class CanaryPreflightError(ValueError):
    """Raised when a required canary admission/evidence check is absent."""


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryPreflightError(f"{label} is missing or malformed") from exc


def _mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} is not an object")
        return None
    return value


def _string(value: Any, label: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} is missing")
        return None
    return value.strip()


def _load_array(path: Path, label: str) -> list[Mapping[str, Any]]:
    raw = _read_json(path, label)
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise CanaryPreflightError(f"{label} must be an array of objects")
    return list(raw)


def _expected_image_ref(image: str, digest: str, label: str, errors: list[str]) -> str:
    if image not in {EXPECTED_WEB_IMAGE, EXPECTED_DB_IMAGE}:
        errors.append(f"{label} image name is not the canonical Classifieds image")
    if not _DIGEST_RE.fullmatch(digest):
        errors.append(f"{label} image digest is not an immutable sha256 reference")
    return f"{image}@{digest}"


def _check_run_identity(
    state: Mapping[str, Any],
    *,
    expected_instances_path: Path,
    expected_run_dir: Path,
    errors: list[str],
) -> dict[str, Any]:
    definition = _mapping(state.get("run_definition"), "pipeline_state.run_definition", errors)
    if definition is None:
        return {}
    try:
        canonical = define_run({"run_definition": dict(definition)})
    except (TypeError, ValueError) as exc:
        errors.append(f"persisted Run Definition is invalid: {exc}")
        canonical = None
    if definition.get("legacy") is not False:
        errors.append("current run definition is legacy or lacks an explicit false legacy marker")
    run_id = _string(definition.get("run_id"), "run definition run_id", errors)
    if run_id is not None and not _RUN_ID_RE.fullmatch(run_id):
        errors.append("run definition run_id is not a safe opaque identifier")
    digest = _string(definition.get("definition_digest"), "run definition digest", errors)
    if digest is not None and not _HEX_DIGEST_RE.fullmatch(digest):
        errors.append("run definition digest is not a SHA-256 hex digest")
    if canonical is not None and canonical.definition_digest != digest:
        errors.append("persisted Run Definition digest does not match its contributions")
    for field in ("run_id", "definition_digest"):
        if field in state and state.get(field) != definition.get(field):
            errors.append(f"pipeline_state.{field} disagrees with the run definition")
    if state.get("step") != "phase_3" or state.get("status") != "complete":
        errors.append("current WARP state is not a completed Phase 3 state")
    contributions = _mapping(
        definition.get("contributions"), "run definition contributions", errors
    )
    canonical_instances_path = str(expected_instances_path.expanduser().resolve(strict=False))
    expected_pipeline = {
        "manifest_path": str((expected_run_dir / "phase_0a" / "BENCHMARK_MANIFEST.json").resolve()),
        "instances_path": canonical_instances_path,
        "sites": [EXPECTED_SITE],
        "task_origin": "all",
        "max_tasks_per_site": 1,
    }
    expected_phase_1 = {
        "sandbox_model": EXPECTED_AGENT_MODEL,
        "generate_novel": False,
        "novel_tasks_per_site": 30,
        "task_capability_profile": EXPECTED_PROFILE,
    }
    expected_phase_2 = {
        "phase_2b_texts_per_plan": 1,
        "phase_2_text_model": f"anthropic/{EXPECTED_AGENT_MODEL}",
        "skip_feasibility": False,
        "feasibility_only": True,
        "feasibility_instances": canonical_instances_path,
        "feasibility_retry_count": 0,
        "force_reverify": True,
        "no_l3_l4": False,
        "runtime_composition": EXPECTED_PROFILE,
    }
    expected_phase_4 = {
        "agent_model": EXPECTED_AGENT_MODEL,
        "agent_runner": EXPECTED_RUNNER,
        "agent_provider": EXPECTED_AGENT_PROVIDER,
        "agent_llm_timeout": 240,
        "agent_step_timeout": 300,
        "agent_task_timeout": 900,
        "phase_4_variant_budget": "adaptive-3-3-1",
        "phase_4_variant_system": EXPECTED_VARIANT_SYSTEM,
        "phase_4_eval_awareness_max_iterations": 1,
        "phase_4_task_id": "classifieds_listing_reply_canary_semantic",
        "skip_intermediate_asr": True,
        "allow_unknown_auth": False,
        "skip_host_bound_storage_state_auth": False,
        "runtime_composition": EXPECTED_PROFILE,
    }
    phase_2 = _mapping(
        contributions.get("phase_2") if contributions else None, "phase 2 definition", errors
    )
    if phase_2 is not None and dict(phase_2) != expected_phase_2:
        errors.append("run definition Phase 2 contribution does not match the canary tuple")
    pipeline = _mapping(
        contributions.get("pipeline") if contributions else None, "pipeline definition", errors
    )
    if pipeline is not None and dict(pipeline) != expected_pipeline:
        errors.append("run definition pipeline contribution does not match the canary tuple")
    phase_1 = _mapping(
        contributions.get("phase_1") if contributions else None, "phase 1 definition", errors
    )
    if phase_1 is not None and dict(phase_1) != expected_phase_1:
        errors.append("run definition Phase 1 contribution does not match the canary tuple")
    phase_4 = _mapping(
        contributions.get("phase_4") if contributions else None, "phase 4 definition", errors
    )
    if phase_4 is not None and dict(phase_4) != expected_phase_4:
        errors.append("run definition Phase 4 contribution does not match the canary tuple")
    return {"id": run_id, "definition_digest": digest, "legacy": False}


def _check_instances(
    instances: Mapping[str, Any],
    *,
    expected_site_url: str,
    expected_writer_storage_state: str,
    errors: list[str],
) -> dict[str, Any]:
    expected_top_level_keys = {
        "benchmark_name",
        "benchmark_codebase",
        "url_placeholders",
        "instances",
    }
    if set(instances) != expected_top_level_keys:
        errors.append("instance topology contains non-canonical top-level fields")
    if instances.get("benchmark_name") != EXPECTED_BENCHMARK:
        errors.append("instance topology benchmark is not visualwebarena")
    if instances.get("benchmark_codebase") != str(Path.cwd().resolve()):
        errors.append("instance topology benchmark_codebase drifted")
    rows = instances.get("instances")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        errors.append("instance topology must contain exactly one instance")
        return {}
    row = rows[0]
    expected_row_keys = {
        "benchmark_name",
        "site_name",
        "site_url",
        "replica_index",
        "replica_name",
        "url_placeholders",
        "auth",
        "agent_auth",
        "reader_auth",
    }
    if set(row) != expected_row_keys:
        errors.append("instance topology contains non-canonical instance fields")
    if row.get("benchmark_name") != EXPECTED_BENCHMARK or row.get("site_name") != EXPECTED_SITE:
        errors.append("the selected instance is not the visualwebarena Classifieds instance")
    if row.get("site_url") != expected_site_url:
        errors.append("the selected instance site URL does not match the canary URL")
    if row.get("reader_auth") != {"type": "none"}:
        errors.append("reader_auth is not exactly anonymous ({type: none})")
    writer_auth = _mapping(row.get("auth"), "instance writer auth", errors)
    if writer_auth is not None and set(writer_auth) != {"type", "storage_state"}:
        errors.append("writer auth contains non-canonical fields")
    storage = _mapping(
        writer_auth.get("storage_state") if writer_auth else None,
        "writer storage_state",
        errors,
    )
    storage_path = storage.get("path") if storage else None
    if storage is not None and set(storage) != {"path"}:
        errors.append("writer storage_state contains non-canonical fields")
    if writer_auth is not None and writer_auth.get("type") != "storage_state":
        errors.append("writer auth is not storage_state")
    if not isinstance(storage_path, str) or not storage_path.strip():
        errors.append("writer storage-state path is missing")
    elif storage_path != expected_writer_storage_state:
        errors.append("writer storage-state path drifted from the configured participant route")
    elif not Path(storage_path).is_file():
        errors.append("writer storage-state artifact is not present")
    if row.get("agent_auth") != {"type": "none"}:
        errors.append("browser agent auth is not explicitly anonymous")
    if instances.get("url_placeholders") != {"__CLASSIFIEDS__": expected_site_url}:
        errors.append("instance topology URL placeholder drifted")
    if row.get("url_placeholders") != {"__CLASSIFIEDS__": expected_site_url}:
        errors.append("selected instance URL placeholder drifted")
    if row.get("replica_index") != 0 or row.get("replica_name") != "classifieds_canary_0":
        errors.append("selected instance replica identity drifted")
    return {
        "benchmark": EXPECTED_BENCHMARK,
        "site": EXPECTED_SITE,
        "site_url": expected_site_url,
        "instance_count": 1,
        "reader_auth": "anonymous",
        "writer_auth": "storage_state_present",
        "agent_auth": "anonymous",
    }


def _check_phase2(
    tasks: list[Mapping[str, Any]],
    *,
    expected_task_id: str,
    expected_listing_id: str,
    expected_site_url: str,
    expected_body: str,
    expected_task: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    if len(tasks) != 1:
        errors.append("Phase 2 admitted task set does not contain exactly one task")
        return {}
    task = tasks[0]
    if task.get("id") != expected_task_id:
        errors.append("Phase 2 admitted task id does not match the expected canary task")
    for key, expected_value in expected_task.items():
        if task.get(key) != expected_value:
            errors.append(f"Phase 2 admitted task canonical field drifted: {key}")
    expected_keys = set(expected_task) | {
        "feasibility",
        "read_surface_urls",
        "read_surface_provenance",
    }
    if set(task) != expected_keys:
        errors.append("Phase 2 admitted task contains non-canonical fields")
    expected_read_url = f"{expected_site_url}/index.php?page=item&id={expected_listing_id}"
    if task.get("read_surface_urls") != [expected_read_url]:
        errors.append("Phase 2 admitted task read surface drifted")
    provenance = task.get("read_surface_provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != {"source", "editor_method", "captured_at"}
        or provenance.get("source") != "classifieds.regular_participant"
        or provenance.get("editor_method") != ["classifieds.create_listing_reply"]
        or not isinstance(provenance.get("captured_at"), str)
        or not provenance.get("captured_at")
    ):
        errors.append("Phase 2 admitted task read-surface provenance drifted")
    if task.get("benchmark") != EXPECTED_BENCHMARK or task.get("site") != EXPECTED_SITE:
        errors.append("Phase 2 admitted task is not the visualwebarena Classifieds task")
    feasibility = _mapping(task.get("feasibility"), "Phase 2 feasibility", errors)
    if feasibility is None or feasibility.get("status") != "verified":
        errors.append("Phase 2 admitted task does not have verified feasibility")
    render_evidence = _mapping(
        feasibility.get("render_evidence") if feasibility else None,
        "Phase 2 render evidence",
        errors,
    )
    diagnostics = _mapping(
        render_evidence.get("diagnostics") if render_evidence else None,
        "Phase 2 render diagnostics",
        errors,
    )
    readback = _mapping(
        diagnostics.get("site_readback") if diagnostics else None,
        "Phase 2 exact site readback",
        errors,
    )
    identity = _mapping(
        readback.get("identity_tokens") if readback else None,
        "Phase 2 readback identity",
        errors,
    )
    if readback is not None and readback.get("verified") is not True:
        errors.append("Phase 2 exact site readback is not verified")
    if identity is not None:
        if str(identity.get("listing_id")) != expected_listing_id:
            errors.append("Phase 2 readback listing identity drifted")
        reply_id = identity.get("reply_id")
        if not isinstance(reply_id, str) or not reply_id.isdigit() or int(reply_id) <= 0:
            errors.append("Phase 2 readback reply identity is not a positive stable id")
        actor = identity.get("actor_name")
        if not isinstance(actor, str) or not actor.strip():
            errors.append("Phase 2 readback actor attribution is missing")
        body_digest = identity.get("reply_body_sha256")
        if not isinstance(body_digest, str) or not _HEX_DIGEST_RE.fullmatch(body_digest):
            errors.append("Phase 2 readback body digest is missing or malformed")
    visibility = _mapping(
        readback.get("visibility") if readback else None,
        "Phase 2 exact readback visibility",
        errors,
    )
    if visibility is not None and visibility.get("ok") is not True:
        errors.append("Phase 2 exact reply body is not painted")
    contract = _mapping(task.get("exposure_contract"), "Phase 2 exposure contract", errors)
    if contract is not None:
        if contract.get("target_surface_id") != EXPECTED_SURFACE:
            errors.append("Phase 2 exposure surface is not listing_reply.body")
        if contract.get("editor_method") != "create_listing_reply":
            errors.append("Phase 2 exposure editor is not create_listing_reply")
        eligibility = _mapping(contract.get("eligibility"), "exposure eligibility", errors)
        if eligibility is None or eligibility.get("status") != "eligible":
            errors.append("Phase 2 exposure contract is not eligible")
        phase4 = _mapping(contract.get("phase4_exposure"), "Phase 4 exposure", errors)
        if phase4 is None or phase4.get("admissible") is not True:
            errors.append("Phase 2 exposure contract is not Phase 4-admissible")
        args = _mapping(contract.get("editor_args_template"), "editor args template", errors)
        if args is not None and str(args.get("listing_id")) != expected_listing_id:
            errors.append("Phase 2 exposure contract listing anchor drifted")
    if task.get("target_surface_id") not in {None, EXPECTED_SURFACE}:
        errors.append("Phase 2 task target surface is not listing_reply.body")
    seed = _mapping(task.get("adversarial_data_seed"), "Phase 2 adversarial seed", errors)
    editor_calls = seed.get("editor_calls") if seed is not None else None
    if not isinstance(editor_calls, list) or not editor_calls:
        errors.append("Phase 2 adversarial seed lacks editor calls")
    elif contract is not None and feasibility is not None:
        fingerprint = _mapping(
            feasibility.get("host_fingerprint"), "Phase 2 host fingerprint", errors
        )
        expected_content_hash = _task_content_hash(
            editor_calls,
            exposure_contract=dict(contract),
        )
        if (
            fingerprint is not None
            and fingerprint.get("task_content_hash") != expected_content_hash
        ):
            errors.append("Phase 2 task content does not match its verified fingerprint")
        if len(editor_calls) != 1 or not isinstance(editor_calls[0], Mapping):
            errors.append("Phase 2 canary must contain exactly one editor call")
        else:
            call = editor_calls[0]
            args = call.get("args")
            if (
                call.get("benchmark") != EXPECTED_BENCHMARK
                or call.get("site") != EXPECTED_SITE
                or call.get("method") != "create_listing_reply"
                or not isinstance(args, Mapping)
            ):
                errors.append("Phase 2 editor call is not the exact Classifieds writer call")
            else:
                body = args.get("body")
                if str(args.get("listing_id")) != expected_listing_id:
                    errors.append("Phase 2 editor call listing identity drifted")
                if not isinstance(body, str) or not body.strip():
                    errors.append("Phase 2 editor call body is missing")
                elif body != expected_body:
                    errors.append("Phase 2 editor call body is not the deterministic canary body")
                elif identity is not None:
                    body_digest = hashlib.sha256(
                        normalize_reply_body(body).encode("utf-8")
                    ).hexdigest()
                    if identity.get("reply_body_sha256") != body_digest:
                        errors.append("Phase 2 readback body digest does not match the seeded body")
    action = _mapping(task.get("adversarial_action"), "adversarial action", errors)
    if action is None or action.get("kind") != "answer_opposite_binary_label":
        errors.append("Phase 2 task does not use the named semantic Classifieds action")
    admission_error = exposure_admission_error(dict(task))
    if admission_error is not None:
        errors.append(f"Phase 2 task is not Phase 4-admissible: {admission_error}")
    return {
        "task_id": expected_task_id,
        "feasibility": "verified",
        "surface": EXPECTED_SURFACE,
        "listing_id": str(identity.get("listing_id") or "") if identity else "",
        "reply_id": str(identity.get("reply_id") or "") if identity else "",
        "actor_name": str(identity.get("actor_name") or "").strip() if identity else "",
        "reply_body_sha256": (str(identity.get("reply_body_sha256") or "") if identity else ""),
        # The named runtime composition admitted this readback only through a
        # fresh anonymous browser context; preserve that bounded contract in
        # the final preflight evidence without persisting browser state.
        "reader_route": "fresh_anonymous_context",
    }


def _check_phase2_checkpoint(
    run_dir: Path,
    *,
    task: Mapping[str, Any] | None,
    run: Mapping[str, Any],
    expected_topology: Mapping[str, str] | None,
    expected_task_id: str,
    errors: list[str],
) -> dict[str, Any]:
    paths = sorted((run_dir / "phase_2" / "feasibility_checkpoints").glob("*.json"))
    if len(paths) != 1:
        errors.append("Phase 2c must have exactly one task checkpoint")
        return {}
    raw = _mapping(_read_json(paths[0], "Phase 2c checkpoint"), "Phase 2c checkpoint", errors)
    if raw is None or task is None:
        return {}
    if expected_topology is None:
        errors.append("Phase 2c checkpoint topology cannot be derived from the instance topology")
        return {}
    seed = task.get("adversarial_data_seed")
    exposure = task.get("exposure_contract")
    editor_calls = seed.get("editor_calls") if isinstance(seed, Mapping) else None
    if not isinstance(editor_calls, list) or not isinstance(exposure, Mapping):
        errors.append("Phase 2c checkpoint cannot bind malformed task content")
        return {}
    task_content_hash = _task_content_hash(editor_calls, exposure_contract=dict(exposure))
    feasibility = task.get("feasibility")
    observed_host = (
        feasibility.get("host_fingerprint") if isinstance(feasibility, Mapping) else None
    )
    expected_host = {**expected_topology, "task_content_hash": task_content_hash}
    if observed_host != expected_host:
        errors.append("Phase 2 task host fingerprint does not match the selected topology")
    context = Phase2cCheckpointContext(
        run_id=str(run.get("id") or ""),
        definition_digest=str(run.get("definition_digest") or ""),
        task_id=expected_task_id,
        task_content_hash=task_content_hash,
        task_fingerprint=task_fingerprint(task),
        topology_fingerprint=dict(expected_topology),
    )
    try:
        checkpoint_result = validate_checkpoint_payload(raw, context=context)
    except (CheckpointValidationError, TypeError, ValueError) as exc:
        errors.append(f"Phase 2c checkpoint is not bound to current evidence: {exc}")
        return {}
    if checkpoint_result != dict(task):
        errors.append("Phase 2 promoted task does not match its validated checkpoint result")
    work_unit = raw.get("work_unit")
    if not isinstance(work_unit, Mapping) or any(
        work_unit.get(key) is not True
        for key in (
            "seed_applied",
            "render_completed",
            "reachability_completed",
            "cleanup_completed",
        )
    ):
        errors.append("Phase 2c checkpoint does not prove the complete atomic work unit")
    if raw.get("cleanup_completed") is not True or raw.get("cleanup_warnings") not in ([], ()):
        errors.append("Phase 2c checkpoint cleanup is incomplete or warned")
    if not isinstance(work_unit, Mapping) or work_unit.get("outcome") != "verified":
        errors.append("Phase 2c checkpoint outcome is not verified")
    return {"task_id": expected_task_id, "atomic_work_unit": "verified"}


def _check_phase3(
    contracts: list[Mapping[str, Any]],
    *,
    expected_benign_task_id: str,
    expected_task: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    if len(contracts) != 1:
        errors.append("Phase 3 must contain exactly one canonical benign contract")
    matches = [item for item in contracts if item.get("id") == expected_benign_task_id]
    if len(matches) != 1:
        errors.append("matching Phase 3 benign contract is missing or duplicated")
        return {}
    contract = matches[0]
    if set(contract) != {"id", "origin", "validity_status", "validity_errors", "task"}:
        errors.append("Phase 3 benign contract contains non-canonical fields")
    if contract.get("origin") != "existing_task":
        errors.append("matching Phase 3 contract origin is not existing_task")
    if contract.get("validity_status") != "valid":
        errors.append("matching Phase 3 contract is not valid")
    if contract.get("validity_errors") not in ([], None):
        errors.append("matching Phase 3 contract contains validity errors")
    task = _mapping(contract.get("task"), "Phase 3 contract task", errors)
    if task is None or task.get("id") != expected_benign_task_id:
        errors.append("Phase 3 contract task identity does not match the expected benign task")
    elif dict(task) != dict(expected_task):
        errors.append("Phase 3 contract task does not match the canonical benign task")
    return {"task_id": expected_benign_task_id, "validity": "valid"}


def _check_prepare_and_images(
    prepare: Mapping[str, Any],
    images: Mapping[str, Any],
    *,
    expected_web_ref: str,
    expected_db_ref: str,
    expected_source_commit: str,
    expected_task_id: str,
    expected_benign_task_id: str,
    errors: list[str],
) -> dict[str, Any]:
    provenance = _mapping(prepare.get("provenance"), "prepare provenance", errors)
    if provenance is None:
        return {}
    if provenance.get("benchmark") != EXPECTED_BENCHMARK or provenance.get("site") != EXPECTED_SITE:
        errors.append("prepare manifest benchmark/site does not match the canary")
    if (
        provenance.get("web_image") != expected_web_ref
        or provenance.get("db_image") != expected_db_ref
    ):
        errors.append("prepare manifest image refs do not match the pinned refs")
    if provenance.get("source_commit") != expected_source_commit:
        errors.append("prepare manifest source commit does not match the pinned source")
    if provenance.get("task_ids") != [expected_benign_task_id, expected_task_id]:
        errors.append("prepare manifest task ids are not the exact one-task canary pair")
    recorded_digest = provenance.get("prepare_digest")
    digest_input = {key: value for key, value in provenance.items() if key != "prepare_digest"}
    computed_digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if recorded_digest != computed_digest:
        errors.append("prepare manifest provenance digest does not match its contents")

    for key, expected_ref in (("web", expected_web_ref), ("db", expected_db_ref)):
        row = _mapping(images.get(key), f"images.{key}", errors)
        if row is None:
            continue
        if row.get("ref") != expected_ref:
            errors.append(f"images.{key}.ref does not match the pinned ref")
        repo_digests = row.get("repo_digests")
        if not isinstance(repo_digests, list) or expected_ref not in repo_digests:
            errors.append(f"images.{key} does not prove the pinned RepoDigest")
        if not isinstance(row.get("id"), str) or not row.get("id"):
            errors.append(f"images.{key} is missing the resolved image id")
        if row.get("os") != "linux" or row.get("architecture") != "amd64":
            errors.append(f"images.{key} is not the inspected linux/amd64 image")
    return {
        "platform": "linux/amd64",
        "web_ref": expected_web_ref,
        "db_ref": expected_db_ref,
        "source_commit": expected_source_commit,
    }


def _check_profile(
    run_dir: Path,
    *,
    prepare: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    from scripts.prepare_classifieds_canary import _profile_document

    path = run_dir / "phase_0c" / "BENCHMARK_PROFILE_classifieds.json"
    try:
        profile = load_and_validate_profile(EXPECTED_SITE, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Classifieds Phase 4 profile is missing or invalid: {exc}")
        return {}
    provenance = prepare.get("provenance")
    expected_digest = provenance.get("profile_sha256") if isinstance(provenance, Mapping) else None
    if expected_digest != digest:
        errors.append("Classifieds profile digest is not bound to prepare provenance")
    surfaces = profile.get("injection_surface")
    if not isinstance(surfaces, list) or len(surfaces) != 1:
        errors.append("Classifieds profile must contain exactly one injection surface")
    else:
        surface = surfaces[0]
        if not isinstance(surface, Mapping) or surface.get("id") != EXPECTED_SURFACE:
            errors.append("Classifieds profile surface is not listing_reply.body")
    if profile_requires_agent_auth(profile):
        errors.append("Classifieds profile must keep the browser agent anonymous")
    if profile != _profile_document():
        errors.append(
            "Classifieds profile does not match the canonical nonprivileged form contract"
        )
    return {"path": str(path), "sha256": digest, "surface": EXPECTED_SURFACE}


def _check_overlay(
    path: Path,
    *,
    site_url: str,
    network: str,
    web_port: int,
    web_ref: str,
    db_ref: str,
    app_env_file: str,
    project_name: str,
    errors: list[str],
) -> dict[str, Any]:
    if project_name != CLASSIFIEDS_COMPOSE_PROJECT:
        errors.append("Compose project is not the dedicated Classifieds canary project")
    expected = build_compose_overlay_from_values(
        site_url=site_url,
        network=network,
        web_port=web_port,
        web_image_ref=web_ref,
        db_image_ref=db_ref,
        app_env_file=app_env_file,
    )
    try:
        observed = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"Classifieds Compose overlay is missing: {exc}")
        return {}
    if observed != expected:
        errors.append("Classifieds Compose overlay drifted from the pinned additive topology")
    return {
        "path": str(path),
        "project": CLASSIFIEDS_COMPOSE_PROJECT,
        "sha256": hashlib.sha256(observed.encode()).hexdigest(),
    }


def _check_reset_probe(
    probe: Mapping[str, Any],
    *,
    expected_listing_id: str,
    expected_body: str,
    errors: list[str],
) -> dict[str, Any]:
    if str(probe.get("listing_id")) != expected_listing_id:
        errors.append("reset absence evidence targets a different listing")
    reset = _mapping(probe.get("reset_observation"), "reset observation", errors)
    if reset is None:
        return {}
    precondition = _mapping(probe.get("precondition"), "initial probe precondition", errors)
    if precondition != {"independent": True, "anonymous": True, "canary_absent": True}:
        errors.append("initial anonymous canary-absence precondition is missing")
    for key in ("independent", "anonymous", "reply_absent"):
        if reset.get(key) is not True:
            errors.append(f"reset absence evidence is not true for {key}")
    reply_id = str(probe.get("reply_id") or "")
    if not reply_id.isdigit() or reset.get("checked_reply_id") != reply_id:
        errors.append("reset absence evidence is not bound to its exact reply identity")
    actor = str(probe.get("actor_name") or "").strip()
    expected_digest = hashlib.sha256(
        normalize_reply_body(expected_body).encode("utf-8")
    ).hexdigest()
    if (
        not actor
        or probe.get("body") != expected_body
        or probe.get("body_sha256") != expected_digest
        or probe.get("writer") != {"regular_participant": True}
        or probe.get("reader") != {"independent": True, "anonymous": True, "exact_reply_id": True}
    ):
        errors.append("reset evidence is not the exact writer/reader canary witness")
    return {
        "listing_id": expected_listing_id,
        "reply_id": reply_id,
        "actor_name": actor,
        "reply_body_sha256": expected_digest,
        "independent": True,
        "anonymous": True,
        "reply_absent": True,
    }


def _check_command_contract(
    *,
    task_count: int,
    worker_count: int,
    max_iterations: int,
    variant_system: str,
    runner: str,
    agent_provider: str,
    agent_model: str,
    sandbox_model: str,
    errors: list[str],
) -> dict[str, Any]:
    expected = {
        "task_count": 1,
        "worker_count": 1,
        "max_iterations": 1,
        "variant_system": EXPECTED_VARIANT_SYSTEM,
        "runner": EXPECTED_RUNNER,
        "agent_provider": EXPECTED_AGENT_PROVIDER,
        "agent_model": EXPECTED_AGENT_MODEL,
        "sandbox_model": EXPECTED_AGENT_MODEL,
    }
    observed = locals()
    for key, value in expected.items():
        if observed[key] != value:
            errors.append(f"Phase 4 command contract drifted for {key}")
    return {**expected, "max_one_iterator": True}


def validate_preflight(
    *,
    run_dir: Path,
    instances_path: Path,
    prepare_path: Path,
    images_path: Path,
    probe_path: Path,
    expected_site_url: str,
    expected_writer_storage_state: str,
    expected_overlay_path: Path,
    expected_project_name: str,
    expected_network: str,
    expected_web_port: int,
    expected_app_env_file: str,
    expected_listing_id: str,
    expected_task_id: str,
    expected_benign_task_id: str,
    expected_web_ref: str,
    expected_db_ref: str,
    expected_source_commit: str,
    task_count: int = 1,
    worker_count: int = 1,
    max_iterations: int = 1,
    variant_system: str = EXPECTED_VARIANT_SYSTEM,
    runner: str = EXPECTED_RUNNER,
    agent_provider: str = EXPECTED_AGENT_PROVIDER,
    agent_model: str = EXPECTED_AGENT_MODEL,
    sandbox_model: str = EXPECTED_AGENT_MODEL,
) -> dict[str, Any]:
    """Validate the host canary boundary and return secret-free evidence."""

    errors: list[str] = []
    try:
        expected_site_url = validate_classifieds_loopback_origin(expected_site_url)
    except ValueError as exc:
        errors.append(f"expected site URL is not the loopback canary origin: {exc}")
    state = _mapping(
        _read_json(run_dir / "pipeline_state.json", "pipeline state"), "pipeline state", errors
    )
    if state is None:
        state = {}
    run = _check_run_identity(
        state,
        expected_instances_path=instances_path,
        expected_run_dir=run_dir,
        errors=errors,
    )
    instances = _mapping(
        _read_json(instances_path, "instance topology"), "instance topology", errors
    )
    topology = _check_instances(
        instances or {},
        expected_site_url=expected_site_url,
        expected_writer_storage_state=expected_writer_storage_state,
        errors=errors,
    )
    instance_rows = instances.get("instances") if instances is not None else None
    expected_checkpoint_topology = (
        _host_fingerprint(
            instances_path.name,
            [{**dict(instance_rows[0]), "benchmark": EXPECTED_BENCHMARK}],
        )
        if isinstance(instance_rows, list)
        and len(instance_rows) == 1
        and isinstance(instance_rows[0], Mapping)
        else None
    )
    phase2_tasks = _load_array(
        run_dir / "phase_2" / "adversarial_tasks.json", "Phase 2 admitted tasks"
    )
    from scripts.prepare_classifieds_canary import _task_pair

    canonical_benign, canonical_adversarial, _ = _task_pair(
        site_url=expected_site_url,
        listing_id=expected_listing_id,
        run_dir=run_dir.as_posix(),
    )
    phase2 = _check_phase2(
        phase2_tasks,
        expected_task_id=expected_task_id,
        expected_listing_id=expected_listing_id,
        expected_site_url=expected_site_url,
        expected_body=canary_body_for_run(run_dir.as_posix()),
        expected_task=canonical_adversarial,
        errors=errors,
    )
    phase2_checkpoint = _check_phase2_checkpoint(
        run_dir,
        task=phase2_tasks[0] if len(phase2_tasks) == 1 else None,
        run=run,
        expected_topology=expected_checkpoint_topology,
        expected_task_id=expected_task_id,
        errors=errors,
    )
    phase3 = _check_phase3(
        _load_array(run_dir / "phase_3" / "contracts.json", "Phase 3 contracts"),
        expected_benign_task_id=expected_benign_task_id,
        expected_task=canonical_benign,
        errors=errors,
    )
    expected_web_ref = expected_web_ref.strip()
    expected_db_ref = expected_db_ref.strip()
    if expected_web_ref != f"{CLASSIFIEDS_WEB_IMAGE}@{CLASSIFIEDS_WEB_MANIFEST_DIGEST}":
        errors.append("expected web image is not the canonical pinned Classifieds image")
    if expected_db_ref != f"{CLASSIFIEDS_DB_IMAGE}@{CLASSIFIEDS_DB_MANIFEST_DIGEST}":
        errors.append("expected DB image is not the canonical pinned Classifieds image")
    if expected_source_commit != CLASSIFIEDS_SOURCE_COMMIT:
        errors.append("expected source commit is not the canonical pinned Classifieds source")
    if "@" not in expected_web_ref or "@" not in expected_db_ref:
        errors.append("expected image refs must include immutable digests")
    else:
        web_image, web_digest = expected_web_ref.rsplit("@", 1)
        db_image, db_digest = expected_db_ref.rsplit("@", 1)
        _expected_image_ref(web_image, web_digest, "web", errors)
        _expected_image_ref(db_image, db_digest, "db", errors)
    images = _mapping(_read_json(images_path, "image evidence"), "image evidence", errors)
    prepare = _mapping(_read_json(prepare_path, "prepare manifest"), "prepare manifest", errors)
    image_evidence = _check_prepare_and_images(
        prepare or {},
        images or {},
        expected_web_ref=expected_web_ref,
        expected_db_ref=expected_db_ref,
        expected_source_commit=expected_source_commit,
        expected_task_id=expected_task_id,
        expected_benign_task_id=expected_benign_task_id,
        errors=errors,
    )
    profile_evidence = _check_profile(run_dir, prepare=prepare or {}, errors=errors)
    overlay_evidence = _check_overlay(
        expected_overlay_path,
        site_url=expected_site_url,
        network=expected_network,
        web_port=expected_web_port,
        web_ref=expected_web_ref,
        db_ref=expected_db_ref,
        app_env_file=expected_app_env_file,
        project_name=expected_project_name,
        errors=errors,
    )
    reset = _check_reset_probe(
        _mapping(_read_json(probe_path, "reset absence evidence"), "reset absence evidence", errors)
        or {},
        expected_listing_id=expected_listing_id,
        expected_body=canary_body_for_run(run_dir.as_posix()),
        errors=errors,
    )
    command = _check_command_contract(
        task_count=task_count,
        worker_count=worker_count,
        max_iterations=max_iterations,
        variant_system=variant_system,
        runner=runner,
        agent_provider=agent_provider,
        agent_model=agent_model,
        sandbox_model=sandbox_model,
        errors=errors,
    )
    direct_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    compatible_anthropic = bool(
        os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
        and os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    )
    if not direct_anthropic and not compatible_anthropic:
        errors.append("Anthropic provider route is not present")
    if errors:
        raise CanaryPreflightError("classifieds canary preflight failed: " + "; ".join(errors))
    return {
        "schema_version": 1,
        "status": "passed",
        "run": run,
        "topology": topology,
        "phase2": {**phase2, "checkpoint": phase2_checkpoint},
        "phase3": phase3,
        "images": image_evidence,
        "profile": profile_evidence,
        "overlay": overlay_evidence,
        "reset": reset,
        "command": command,
        "environment": {"anthropic_route": "direct" if direct_anthropic else "compatible_base_url"},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the prepared one-task Classifieds canary before Phase 4.",
        epilog=(
            "Inputs: prepared artifacts, the exact loopback topology, immutable image/source "
            "references, and the expected Run/task values. Output: secret-free preflight.json "
            "(or --output). Safety: this command only reads and validates artifacts; it does "
            "not start containers, mutate the listing, or call a reset endpoint."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Classifieds canary Run root containing the evidence files.",
    )
    parser.add_argument(
        "--instances",
        required=True,
        type=Path,
        help="Host-local instances document used by Phase 2c and Phase 4.",
    )
    parser.add_argument(
        "--prepare",
        type=Path,
        help="Preparation manifest; defaults to <run-dir>/prepare.json.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Image evidence; defaults to <run-dir>/images.json.",
    )
    parser.add_argument(
        "--probe-evidence",
        type=Path,
        help="Writer/reader/reset probe evidence; defaults to <run-dir>/classifieds_probe.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Preflight output path; defaults to <run-dir>/preflight.json.",
    )
    parser.add_argument("--site-url", required=True, help="Configured loopback Classifieds origin.")
    parser.add_argument(
        "--writer-storage-state",
        required=True,
        help="External regular-participant storage-state path.",
    )
    parser.add_argument(
        "--overlay",
        required=True,
        type=Path,
        help="Compose overlay produced by preparation.",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Dedicated compose project name; must match the canary contract.",
    )
    parser.add_argument("--network", required=True, help="Dedicated compose network name.")
    parser.add_argument(
        "--web-port",
        required=True,
        type=int,
        help="Loopback web port in the selected topology.",
    )
    parser.add_argument(
        "--app-env-file",
        required=True,
        help="External Classifieds application environment file reference.",
    )
    parser.add_argument("--listing-id", required=True, help="Exact seeded listing ID.")
    parser.add_argument("--expected-task-id", required=True, help="Expected adversarial task ID.")
    parser.add_argument(
        "--expected-benign-task-id",
        required=True,
        help="Expected benign task ID.",
    )
    parser.add_argument(
        "--web-image-ref",
        required=True,
        help="Pinned Classifieds web image reference.",
    )
    parser.add_argument(
        "--db-image-ref",
        required=True,
        help="Pinned Classifieds database image reference.",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Pinned Classifieds source commit.",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=1,
        help="Expected admitted task count (default: 1).",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="Expected worker count (default: 1).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum evaluation-awareness iterations (default: 1).",
    )
    parser.add_argument(
        "--variant-system",
        default=EXPECTED_VARIANT_SYSTEM,
        help=f"Expected Phase 4 variant system (default: {EXPECTED_VARIANT_SYSTEM}).",
    )
    parser.add_argument(
        "--runner",
        default=EXPECTED_RUNNER,
        help=f"Expected runner (default: {EXPECTED_RUNNER}).",
    )
    parser.add_argument(
        "--agent-provider",
        default=EXPECTED_AGENT_PROVIDER,
        help=f"Expected agent provider (default: {EXPECTED_AGENT_PROVIDER}).",
    )
    parser.add_argument(
        "--agent-model",
        default=EXPECTED_AGENT_MODEL,
        help=f"Expected agent model (default: {EXPECTED_AGENT_MODEL}).",
    )
    parser.add_argument(
        "--sandbox-model",
        default=EXPECTED_AGENT_MODEL,
        help=f"Expected sandbox model (default: {EXPECTED_AGENT_MODEL}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    try:
        payload = validate_preflight(
            run_dir=run_dir,
            instances_path=args.instances,
            prepare_path=args.prepare or run_dir / "prepare.json",
            images_path=args.images or run_dir / "images.json",
            probe_path=args.probe_evidence or run_dir / "classifieds_probe.json",
            expected_site_url=args.site_url.rstrip("/"),
            expected_writer_storage_state=args.writer_storage_state,
            expected_overlay_path=args.overlay,
            expected_project_name=args.project_name,
            expected_network=args.network,
            expected_web_port=args.web_port,
            expected_app_env_file=args.app_env_file,
            expected_listing_id=args.listing_id,
            expected_task_id=args.expected_task_id,
            expected_benign_task_id=args.expected_benign_task_id,
            expected_web_ref=args.web_image_ref,
            expected_db_ref=args.db_image_ref,
            expected_source_commit=args.source_commit,
            task_count=args.task_count,
            worker_count=args.worker_count,
            max_iterations=args.max_iterations,
            variant_system=args.variant_system,
            runner=args.runner,
            agent_provider=args.agent_provider,
            agent_model=args.agent_model,
            sandbox_model=args.sandbox_model,
        )
    except CanaryPreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output = args.output or run_dir / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "evidence": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
