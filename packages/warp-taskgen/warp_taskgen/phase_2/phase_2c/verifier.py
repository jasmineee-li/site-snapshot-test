"""Phase 2c per-task verifier: seed, render/readback, reachability, cleanup."""

from __future__ import annotations

import asyncio
from typing import Any

from warp_taskgen._async_utils import retrying
from warp_taskgen.editors import EditorError
from warp_taskgen.phase_1.gitlab_compare_decide import GitLabBindingError
from warp_taskgen.phase_2.phase_2c import checkpoints as _checkpoints
from warp_taskgen.phase_2.phase_2c.admission_guards import _answer_target_collision_reason
from warp_taskgen.phase_2.phase_2c.fingerprints import (
    _idempotency_decision,
    _task_content_hash,
)
from warp_taskgen.phase_2.phase_2c.gitlab_compare_join import join_gitlab_compare_decide_seed
from warp_taskgen.phase_2.phase_2c.outcomes import _infeasible_task, _now_iso, _safe_cleanup
from warp_taskgen.phase_2.phase_2c.probe_bundle import Phase2cProbeBundle
from warp_taskgen.phase_2.phase_2c.probes import (
    _RENDER_UNVERIFIED_KIND,
    _RENDER_UNVERIFIED_RETRY_DELAY_S,
)
from warp_taskgen.phase_2.phase_2c.reddit_attribution import (
    _attach_gitlab_issue_note_state_probe_anchors,
    _attach_reddit_comment_attribution_contract,
)
from warp_taskgen.phases.phase_2_reachability import ReachabilityOutcome
from warp_taskgen.phases.phase_2_render_check import RenderOutcome
from warp_taskgen.runtime_composition import RequiredSeedCleanupError, RuntimeComposition
from warp_taskgen.seeding import SeedCleanupHandle, UnboundTokenError


async def _verify_one(
    task: dict[str, Any],
    instance: dict[str, Any],
    *,
    retry_count: int,
    fingerprint_base: dict[str, str],
    ttl_hours: float | None,
    force_reverify: bool,
    cleanup_warnings: list[str],
    browser: Any = None,
    render_semaphore: asyncio.Semaphore | None = None,
    runtime_composition: RuntimeComposition,
    checkpoint_context: _checkpoints.Phase2cCheckpointContext | None = None,
    checkpoint_work_unit: dict[str, bool] | None = None,
    probes: Phase2cProbeBundle,
) -> dict[str, Any]:
    """Verify one task against one instance under the Run's composition.

    ``runtime_composition`` is the whole per-Run bundle: its seed registry and
    token scope bind the seed application, its Site catalog and planning
    strictness bind the read-surface plan, and its cleanup strictness and
    reader preflight bind the Atomic Work Unit.  A Run that named no
    composition resolves :meth:`RuntimeComposition.default` before it gets
    here.
    """

    strict_cleanup = runtime_composition.strict_seed_cleanup
    seed = task.get("adversarial_data_seed") or {}
    editor_calls = seed.get("editor_calls") if isinstance(seed, dict) else None

    content_hash = _task_content_hash(
        editor_calls if isinstance(editor_calls, list) else [],
        exposure_contract=task.get("exposure_contract"),
    )
    fingerprint = dict(fingerprint_base)
    fingerprint["task_content_hash"] = content_hash

    # Identified runs reuse only a complete, validated task checkpoint.  This
    # is stricter than the legacy feasibility stanza/TTL shortcut: a topology
    # drift or malformed checkpoint must rerun even if an older result is
    # recent.  Legacy direct callers retain the historical TTL/fingerprint
    # behavior until they acquire an explicit Run-bound checkpoint directory.
    if checkpoint_context is not None:
        decision, skip_reason = ("verify", None)
    else:
        decision, skip_reason = _idempotency_decision(
            task.get("feasibility"),
            current_fingerprint=fingerprint,
            ttl_hours=ttl_hours,
            force_reverify=force_reverify,
        )
    if decision == "skip":
        # Preserve the prior ``status="verified"`` record verbatim — Phase 4's
        # strict admission gate only admits ``status == "verified"``, so
        # overwriting to ``"skipped"`` would silently take prior verifications
        # offline on every idempotent re-run. We record the skip fact on a
        # sibling field so the report bucket still picks it out.
        result = dict(task)
        prior = dict(task.get("feasibility") or {})
        prior["last_reverify_skipped_at"] = _now_iso()
        prior["last_reverify_skip_reason"] = skip_reason or "fingerprint_match"
        result["feasibility"] = prior
        return result

    # A named composition may require a reader contract before it mutates the
    # benchmark.  Check that contract before applying the writer seed so a
    # missing/invalid anonymous-reader declaration fails closed without
    # leaving seeded state behind.  The render probe repeats the pure check
    # immediately before opening its reader context and records its metadata.
    if runtime_composition.reader_preflight is not None:
        try:
            reader_result = runtime_composition.reader_preflight(instance)
        except Exception as exc:
            return _infeasible_task(
                task,
                kind="auth_unusable",
                detail=(f"independent reader preflight raised {exc.__class__.__name__}: {exc}"),
                fingerprint=fingerprint,
                http_status=None,
                response_snippet=None,
                attempts=[],
                timestamp=_now_iso(),
            )
        if not getattr(reader_result, "ok", False):
            reason = str(getattr(reader_result, "reason", "reader_contract_failed"))
            detail = str(getattr(reader_result, "detail", "independent reader contract failed"))
            return _infeasible_task(
                task,
                kind="auth_missing" if reason == "missing_reader_auth" else "auth_unusable",
                detail=f"independent reader preflight failed: {reason}: {detail}",
                fingerprint=fingerprint,
                http_status=None,
                response_snippet=None,
                attempts=[],
                timestamp=_now_iso(),
            )

    if not isinstance(editor_calls, list):
        return _infeasible_task(
            task,
            kind="schema_mismatch",
            detail="adversarial_data_seed missing editor_calls list",
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=[],
            timestamp=_now_iso(),
        )

    answer_collision_reason = _answer_target_collision_reason(task)
    if answer_collision_reason is not None:
        return _infeasible_task(
            task,
            kind="answer_target_collision",
            detail=answer_collision_reason,
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=[],
            timestamp=_now_iso(),
        )

    bound_instance = dict(instance)
    bound_instance["seed_task"] = task

    attempts: list[dict[str, Any]] = []
    handle: SeedCleanupHandle | None = None
    metadata: dict[str, Any] = {}
    gitlab_compare_binding: Any = None

    async def _apply_and_keep_metadata() -> tuple[SeedCleanupHandle | None, dict[str, Any]]:
        apply_kwargs: dict[str, Any] = {
            "seed_registry": runtime_composition.seed_registry,
            "seed_token_scope": runtime_composition.seed_token_scope,
        }
        if strict_cleanup:
            apply_kwargs["strict_cleanup"] = True
        return await probes.apply_seed(seed, bound_instance, **apply_kwargs)

    try:
        handle, metadata = await retrying(
            _apply_and_keep_metadata,
            retries=retry_count,
            sleep=probes.retry_sleep,
            attempts_log=attempts,
        )
        gitlab_compare_binding = join_gitlab_compare_decide_seed(task, metadata)
    except EditorError as exc:
        _safe_cleanup(
            handle,
            cleanup_warnings,
            task.get("id"),
            raise_on_failure=strict_cleanup,
        )
        return _infeasible_task(
            task,
            kind=exc.kind,
            detail=exc.detail,
            fingerprint=fingerprint,
            http_status=exc.http_status,
            response_snippet=exc.response_snippet,
            attempts=attempts,
            timestamp=_now_iso(),
        )
    except UnboundTokenError as exc:
        # Phantom {benign_*} token — the seed referenced a token the
        # resolver's anchors don't support. Categorized separately from
        # schema_mismatch so dashboards can track the commit 4/6 fail-
        # loud contract hits distinct from shape violations.
        _safe_cleanup(
            handle,
            cleanup_warnings,
            task.get("id"),
            raise_on_failure=strict_cleanup,
        )
        return _infeasible_task(
            task,
            kind="contract_violation",
            detail=str(exc),
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
        )
    except GitLabBindingError as exc:
        # A generated comparison task may only be admitted with a complete
        # current-attempt map. Keep this distinct from ordinary seed schema
        # failures so the feasibility report explains the backpressure cause.
        _safe_cleanup(
            handle,
            cleanup_warnings,
            task.get("id"),
            raise_on_failure=strict_cleanup,
        )
        return _infeasible_task(
            task,
            kind="gitlab_binding_failed",
            detail=f"{exc.code}: {exc}",
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
        )
    except RequiredSeedCleanupError:
        # Required cleanup is a named composition terminal gate. It is a
        # RuntimeError subclass, so keep this branch ahead of the structural
        # ValueError/RuntimeError categorization below.
        raise
    except (ValueError, RuntimeError) as exc:
        # ValueError comes from validate_data_seed; RuntimeError comes from
        # ``_render_editor_seed_call`` when a template placeholder (e.g.
        # ``{submission_id}``) can't be resolved because the chain is
        # missing a producer call. Both are structural problems; neither is
        # a platform rejection.
        _safe_cleanup(
            handle,
            cleanup_warnings,
            task.get("id"),
            raise_on_failure=strict_cleanup,
        )
        return _infeasible_task(
            task,
            kind="schema_mismatch",
            detail=str(exc),
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
        )

    if handle is None:
        # Empty seed never registered a cleanup handle, so no cleanup needed.
        return _infeasible_task(
            task,
            kind="empty_seed",
            detail="adversarial_data_seed produced no editor calls",
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
        )

    if checkpoint_work_unit is not None:
        checkpoint_work_unit["seed_applied"] = True

    # Render check runs BEFORE cleanup because cleanup deletes the seeded
    # row. The 2026-04-21 Magento bug shipped because Phase 2c stamped
    # ``verified`` on HTTP 2xx alone — Layer 2 of the long-term fix closes
    # that contract gap. ``browser is None`` only when the operator opted
    # out via WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK=1; in that case the
    # ``verified`` stamp regresses to the pre-Layer-2 meaning ("API write
    # succeeded only").
    render_outcome: RenderOutcome | None = None
    reachability_outcome: ReachabilityOutcome | None = None
    try:
        if browser is not None:
            render_kwargs: dict[str, Any] = {
                "browser": browser,
                "render_semaphore": render_semaphore,
                "seed": seed,
                "metadata": metadata,
                "instance": instance,
                "verify_seed_renders": probes.verify_seed_renders,
                "site_catalog": runtime_composition.site_catalog,
                "strict_site_planning": runtime_composition.strict_site_planning,
            }
            if runtime_composition.reader_preflight is not None:
                render_kwargs["reader_preflight"] = runtime_composition.reader_preflight
            render_outcome = await probes.render_check(**render_kwargs)
            if checkpoint_work_unit is not None:
                checkpoint_work_unit["render_completed"] = True
            # Render-unverified means the seed wrote successfully but the
            # signature did not appear in any read-surface URL within the
            # body-poll window. On loaded GitLab hosts this is dominated
            # by sidekiq indexer + page-cache invalidation tail; the seed
            # IS visible a few seconds later. Give the platform one
            # 3-second breather and re-run the check so single-run jitter
            # (typically 1-3 tasks per Phase 2c) doesn't gate admission.
            # The exponential-backoff body poll already handles the fast
            # tail; this retry covers the slow tail (>20 s sidekiq).
            if (
                render_outcome is not None
                and not render_outcome.ok
                and render_outcome.kind == _RENDER_UNVERIFIED_KIND
            ):
                await probes.retry_sleep(_RENDER_UNVERIFIED_RETRY_DELAY_S)
                render_outcome = await probes.render_check(**render_kwargs)
            if render_outcome is not None and render_outcome.ok:
                # Option A reachability only applies to tasks whose benign
                # target resource is known — legacy datasets without the
                # field are skipped so this commit doesn't regress them.
                resource = task.get("benign_target_resource")
                exposure_contract = task.get("exposure_contract")
                if isinstance(exposure_contract, dict):
                    eligibility = exposure_contract.get("eligibility")
                    if not isinstance(eligibility, dict) or eligibility.get("status") != "eligible":
                        reachability_outcome = ReachabilityOutcome.unreachable(
                            kind="exposure_contract_ineligible",
                            detail="task exposure_contract is missing eligible status",
                            url=str(instance.get("site_url") or ""),
                        )
                    else:
                        reachability_outcome = await probes.reachability_check(
                            browser=browser,
                            render_semaphore=render_semaphore,
                            task=task,
                            seed=seed,
                            metadata=metadata,
                            instance=instance,
                            render_outcome=render_outcome,
                            verify_reachable=probes.verify_reachable,
                        )
                elif isinstance(resource, dict) and resource.get("kind") is not None:
                    reachability_outcome = await probes.reachability_check(
                        browser=browser,
                        render_semaphore=render_semaphore,
                        task=task,
                        seed=seed,
                        metadata=metadata,
                        instance=instance,
                        render_outcome=render_outcome,
                        verify_reachable=probes.verify_reachable,
                    )
                if checkpoint_work_unit is not None and reachability_outcome is not None:
                    checkpoint_work_unit["reachability_completed"] = True
    finally:
        _safe_cleanup(
            handle,
            cleanup_warnings,
            task.get("id"),
            raise_on_failure=strict_cleanup,
        )

    if render_outcome is not None and not render_outcome.ok:
        return _infeasible_task(
            task,
            kind=render_outcome.kind,
            detail=render_outcome.detail,
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
            render_evidence=render_outcome.evidence(),
        )

    if reachability_outcome is not None and reachability_outcome.reachability == "unreachable":
        return _infeasible_task(
            task,
            kind=f"reachability_{reachability_outcome.kind}" or "reachability_failed",
            detail=reachability_outcome.detail,
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
            render_evidence=(render_outcome.evidence() if render_outcome else None),
            reachability_evidence=reachability_outcome.evidence(),
        )

    result = dict(task)
    read_surface_urls = metadata.get("read_surface_urls") if isinstance(metadata, dict) else None
    if isinstance(read_surface_urls, list):
        cleaned_urls = [url for url in read_surface_urls if isinstance(url, str) and url.strip()]
        if cleaned_urls:
            result["read_surface_urls"] = cleaned_urls
    read_surface_provenance = (
        metadata.get("read_surface_provenance") if isinstance(metadata, dict) else None
    )
    if isinstance(read_surface_provenance, dict):
        result["read_surface_provenance"] = read_surface_provenance
    feasibility: dict[str, Any] = {
        "status": "verified",
        "verified_at": _now_iso(),
        "host_fingerprint": fingerprint,
        "attempts": attempts,
    }
    if gitlab_compare_binding is not None:
        feasibility["gitlab_compare_decide"] = gitlab_compare_binding.diagnostics
    if render_outcome is not None:
        feasibility["render_verified"] = True
        feasibility["render_evidence"] = render_outcome.evidence()
    if reachability_outcome is not None:
        feasibility["reachability"] = reachability_outcome.reachability
        feasibility["reachability_evidence"] = reachability_outcome.evidence()
        exposure_contract = task.get("exposure_contract")
        if isinstance(exposure_contract, dict):
            layout_probe = (
                render_outcome.layout_probe
                if render_outcome is not None and isinstance(render_outcome.layout_probe, dict)
                else None
            )
            feasibility["exposure"] = {
                "contract_id": exposure_contract.get("contract_id"),
                "reachable": reachability_outcome.reachability != "unreachable",
                "visual_reachable": reachability_outcome.visual_reachable is True,
                "layout_visible_at_entry": (
                    layout_probe.get("visible_at_entry") if layout_probe is not None else None
                ),
                "scroll_to_visible_px": (
                    layout_probe.get("scroll_to_visible_px") if layout_probe is not None else None
                ),
                "requires_expand": (
                    layout_probe.get("requires_expand") if layout_probe is not None else None
                ),
                "verification": exposure_contract.get("verification"),
            }
            diagnostics = (
                render_outcome.diagnostics
                if render_outcome is not None and isinstance(render_outcome.diagnostics, dict)
                else {}
            )
            reddit_comment_visibility = diagnostics.get("reddit_seed_comment_visibility")
            if isinstance(reddit_comment_visibility, dict):
                feasibility["exposure"]["seeded_comment_visibility"] = dict(
                    reddit_comment_visibility
                )
    _attach_gitlab_issue_note_state_probe_anchors(result, metadata)
    attribution_problem = _attach_reddit_comment_attribution_contract(result, metadata)
    if attribution_problem is not None:
        return _infeasible_task(
            task,
            kind="reddit_comment_attribution_unbound",
            detail=attribution_problem,
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
            render_evidence=(render_outcome.evidence() if render_outcome else None),
            reachability_evidence=(
                reachability_outcome.evidence() if reachability_outcome else None
            ),
        )
    result["feasibility"] = feasibility
    return result


__all__ = ["_verify_one"]
