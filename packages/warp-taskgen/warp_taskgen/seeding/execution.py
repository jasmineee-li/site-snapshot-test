"""Editor seed execution: dispatch, per-run editor binding, and cleanup."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import requests

from warp_taskgen.editors import EditorError
from warp_taskgen.seeding.context import (
    _build_seed_context,
    _editor_free_text_arg_names,
    _merge_seed_context,
    _render_editor_seed_call,
    _unresolved_structural_seed_placeholder_names,
)
from warp_taskgen.seeding.editor_args import (
    _editor_call_pre_delay_s,
    _filter_editor_method_args,
    _infer_editor_call_benchmark,
)
from warp_taskgen.seeding.results import (
    _dedupe_created_resources,
    _editor_call_result_record,
    _primary_created_resource,
)
from warp_taskgen.seeding.site_contracts import (
    CreatedResourceFact,
    EditorSeedResult,
    ReadSurfaceFact,
    SeedSiteRegistration,
    SeedSiteRegistry,
    default_seed_registry,
)
from warp_taskgen.seeding.tokens import _assert_benign_tokens_bound
from warp_taskgen.seeding.validation import validate_data_seed

logger = logging.getLogger(__name__)


class SeedCleanupHandle:
    def __init__(
        self,
        *,
        session: requests.Session,
        editor_instances: dict[tuple[str, str], Any],
    ) -> None:
        self._session = session
        self._editor_instances = editor_instances
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        failures: list[str] = []
        for editor in reversed(list(self._editor_instances.values())):
            try:
                editor.cleanup()
            except Exception as exc:
                logger.exception("seed editor cleanup failed")
                failures.append(str(exc) or exc.__class__.__name__)
        try:
            self._session.close()
        except Exception as exc:
            logger.exception("seed session cleanup failed")
            failures.append(str(exc) or exc.__class__.__name__)
        finally:
            self._cleaned = True
        if failures:
            raise RuntimeError("seed cleanup failed: " + "; ".join(failures))


class _EditorInstanceCache(dict[tuple[str, str], Any]):
    """Per-run editor instances plus the immutable registry they were bound to."""

    def __init__(self, seed_registry: SeedSiteRegistry) -> None:
        super().__init__()
        self.seed_registry = seed_registry


def _coerce_seed_registry(seed_registry: SeedSiteRegistry | None) -> SeedSiteRegistry:
    """Resolve the Run's Site editor binding for one seed application.

    A caller that carries a Runtime Composition passes its registry; a caller
    with none resolves the default binding, which is the same object
    :meth:`RuntimeComposition.default` builds.  Everything below this point
    reads the resolved registry unconditionally.
    """

    if seed_registry is None:
        return default_seed_registry()
    if not isinstance(seed_registry, SeedSiteRegistry):
        raise TypeError("seed_registry must be a SeedSiteRegistry")
    return seed_registry


def apply_data_seed(
    seed: dict[str, Any],
    instance: dict[str, Any],
    *,
    seed_registry: SeedSiteRegistry | None = None,
    seed_token_scope: str = "kind",
    strict_cleanup: bool = False,
) -> tuple[SeedCleanupHandle | None, dict[str, Any]]:
    """Apply a data seed to a running benchmark instance.

    Returns a ``(cleanup_handle, metadata)`` tuple where ``metadata`` carries
    the editor-emitted read-surface URLs (C1b signal, see
    ``docs/handoffs/codex-handoff-c1-read-surface.md`` §5.4)::

        {
          "read_surface_urls": [...],  # deduped, first-occurrence order
          "read_surface_provenance": {"source": ..., "editor_method": ...},
        }

    Args:
        seed: Seed spec with a ``mechanism`` field and mechanism-specific
            extras. See the v5 spec for the field schemas.
        instance: Benchmark instance dict with ``site_url`` and any
            mechanism-specific auth configuration.

    Raises:
        ValueError: If ``seed["mechanism"]`` is unknown.
    """
    from warp_taskgen.editors._read_surface import normalize_surface_urls

    if not isinstance(strict_cleanup, bool):
        raise TypeError("strict_cleanup must be a bool")

    resolved_registry = _coerce_seed_registry(seed_registry)
    validate_data_seed(seed, allow_none=True, seed_registry=resolved_registry)

    seed_context = _build_seed_context(seed, instance)
    run_registry = resolved_registry
    editor_instances: dict[tuple[str, str], Any] = _EditorInstanceCache(run_registry)
    session = requests.Session()
    cleanup_handle: SeedCleanupHandle | None = None
    read_surface_accumulator: list[str] = []
    read_surface_provenance: dict[str, Any] = {}
    created_resource_accumulator: list[dict[str, Any]] = []
    editor_call_result_accumulator: list[dict[str, Any]] = []
    try:
        for call_index, call in enumerate(seed.get("editor_calls", [])):
            call_kwargs = {
                "call_index": call_index,
                "seed_context": seed_context,
                "editor_instances": editor_instances,
                "read_surface_accumulator": read_surface_accumulator,
                "read_surface_provenance": read_surface_provenance,
                "created_resource_accumulator": created_resource_accumulator,
                "editor_call_result_accumulator": editor_call_result_accumulator,
                "seed_registry": resolved_registry,
                "seed_token_scope": seed_token_scope,
            }
            _apply_editor_seed_call(
                session,
                call,
                instance,
                **call_kwargs,
            )
        metadata: dict[str, Any] = {}
        # Handoff §5.5: task-author explicit override unions with editor
        # contributions. Explicit entries come first so their order is
        # preserved in the deduped result; the provenance source reflects
        # whether explicit, editor, or both contributed.
        explicit_override: list[str] = []
        seed_task = instance.get("seed_task")
        if isinstance(seed_task, dict):
            raw_override = seed_task.get("read_surface_urls")
            if isinstance(raw_override, list):
                explicit_override = [
                    str(u).strip() for u in raw_override if isinstance(u, str) and str(u).strip()
                ]
        editor_contribution = list(read_surface_accumulator)
        deduped = normalize_surface_urls(explicit_override + editor_contribution)
        if deduped:
            metadata["read_surface_urls"] = deduped
            if explicit_override and editor_contribution:
                source = "explicit_override+editor"
            elif explicit_override:
                source = "explicit_override"
            else:
                source = None
            if source is not None:
                # Build / overlay provenance. If editors also stamped, keep
                # their editor_method attribution; only replace source.
                provenance = dict(read_surface_provenance) if read_surface_provenance else {}
                provenance["source"] = source
                if "captured_at" not in provenance:
                    from datetime import UTC, datetime

                    provenance["captured_at"] = datetime.now(UTC).isoformat()
                metadata["read_surface_provenance"] = provenance
            elif read_surface_provenance:
                metadata["read_surface_provenance"] = read_surface_provenance
        created_resources = _dedupe_created_resources(created_resource_accumulator)
        if created_resources:
            metadata["created_resources"] = created_resources
            metadata["created_resource"] = _primary_created_resource(created_resources)
        if editor_call_result_accumulator:
            metadata["editor_call_results"] = editor_call_result_accumulator
            declared_write_tokens: dict[str, Any] = {}
            for record in editor_call_result_accumulator:
                raw_tokens = record.get("write_tokens")
                if isinstance(raw_tokens, dict):
                    declared_write_tokens.update(raw_tokens)
            if any(
                key
                not in {
                    "note_id",
                    "issue_iid",
                    "project_id",
                    "comment_id",
                    "submission_id",
                    "review_id",
                }
                for key in declared_write_tokens
            ):
                metadata["write_tokens"] = dict(sorted(declared_write_tokens.items()))
        # Hoist authoritative write-identifier tokens from the merged
        # seed_context into metadata so downstream verifiers (render-check
        # read-your-write fastpath) can match server-reported IDs instead
        # of racing the DOM hydration cascade.
        for token_key in (
            "note_id",
            "issue_iid",
            "project_id",
            "comment_id",
            "submission_id",
            "review_id",
        ):
            token_value = seed_context.get(token_key)
            if token_value not in (None, ""):
                metadata[token_key] = token_value
        if editor_instances:
            cleanup_handle = SeedCleanupHandle(
                session=session,
                editor_instances=editor_instances,
            )
            return cleanup_handle, metadata
        session.close()
        return None, metadata
    except Exception as seed_error:
        cleanup = cleanup_handle or SeedCleanupHandle(
            session=session,
            editor_instances=editor_instances,
        )
        cleanup_error: Exception | None = None
        try:
            cleanup.cleanup()
        except Exception as cleanup_exc:
            cleanup_error = cleanup_exc
            # Ordinary callers retain the historical primary exception with a
            # cleanup note. Named compositions cannot safely continue after a
            # partial mutation, however: expose a typed terminal error even
            # though no cleanup handle could be returned to the caller.
            logger.exception("seed cleanup failed after seed execution error")
            if strict_cleanup:
                from warp_taskgen.runtime_composition import RequiredSeedCleanupError

                raise RequiredSeedCleanupError(
                    f"required seed cleanup failed after seed execution error: {cleanup_error}",
                    primary_error=seed_error,
                    cleanup_error=cleanup_error,
                ) from seed_error
            seed_error.add_note(f"seed cleanup also failed: {cleanup_error}")
        if (
            strict_cleanup
            and isinstance(seed_error, EditorError)
            and seed_error.kind == "mutation_unreconciled"
        ):
            from warp_taskgen.runtime_composition import RequiredSeedCleanupError

            raise RequiredSeedCleanupError(
                "required seed cleanup could not reconcile a post-submit mutation",
                primary_error=seed_error,
                cleanup_error=cleanup_error,
            ) from seed_error
        raise


async def apply_data_seed_async(
    seed: dict[str, Any],
    instance: dict[str, Any],
    *,
    seed_registry: SeedSiteRegistry | None = None,
    seed_token_scope: str = "kind",
    strict_cleanup: bool = False,
) -> tuple[SeedCleanupHandle | None, dict[str, Any]]:
    """Apply a data seed without blocking the event loop."""
    return await asyncio.to_thread(
        apply_data_seed,
        seed,
        instance,
        seed_registry=seed_registry,
        seed_token_scope=seed_token_scope,
        strict_cleanup=strict_cleanup,
    )


def preflight_editor_seed_calls(
    seed: dict[str, Any],
    instance: dict[str, Any],
    *,
    seed_registry: SeedSiteRegistry | None = None,
) -> list[dict[str, Any]]:
    """Render and validate editor calls without firing mutations."""
    resolved_registry = _coerce_seed_registry(seed_registry)
    validate_data_seed(seed, allow_none=False, seed_registry=resolved_registry)
    seed_context = _build_seed_context(seed, instance)
    errors: list[dict[str, Any]] = []
    session = requests.Session()
    run_registry = resolved_registry
    editor_instances: dict[tuple[str, str], Any] = _EditorInstanceCache(run_registry)
    try:
        for index, call in enumerate(seed.get("editor_calls", [])):
            if not isinstance(call, dict):
                continue
            try:
                rendered = _render_editor_seed_call(call, seed_context)
                editor = _get_editor_for_seed_call(
                    rendered,
                    instance,
                    session=session,
                    editor_instances=editor_instances,
                    seed_registry=resolved_registry,
                )
                method_name = rendered["method"]
                args = rendered["args"]
                editor_method = getattr(editor, method_name, None)
                if callable(editor_method):
                    args = _filter_editor_method_args(
                        editor_method,
                        args,
                        editor_site_name=str(call.get("site", "")).strip() or "unknown",
                        method_name=str(method_name),
                    )
                    unresolved = sorted(
                        _unresolved_structural_seed_placeholder_names(
                            args,
                            free_text_arg_names=_editor_free_text_arg_names(editor_method),
                        )
                    )
                    if unresolved:
                        raise RuntimeError(
                            "editor call has unresolved template placeholders: "
                            + ", ".join(unresolved)
                        )
                editor.validate_args(method_name, args)
                preview = editor.preview_context(method_name, args)
                if isinstance(preview, dict):
                    _merge_seed_context(seed_context, preview)
            except EditorError as exc:
                errors.append(
                    {
                        "call_index": index,
                        "site": str(call.get("site", "")).strip() or "unknown",
                        "kind": exc.kind,
                        "detail": exc.detail,
                        "method": str(call.get("method", "")).strip() or "unknown",
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "call_index": index,
                        "site": str(call.get("site", "")).strip() or "unknown",
                        "kind": "editor_error",
                        "detail": str(exc),
                        "method": str(call.get("method", "")).strip() or "unknown",
                    }
                )
    finally:
        cleanup = SeedCleanupHandle(
            session=session,
            editor_instances=editor_instances,
        )
        cleanup.cleanup()
    return errors


def _get_editor_for_seed_call(
    call: dict[str, Any],
    instance: dict[str, Any],
    *,
    session: requests.Session,
    editor_instances: dict[tuple[str, str], Any],
    seed_registry: SeedSiteRegistry,
) -> Any:
    benchmark = _infer_editor_call_benchmark(call, instance)
    site = str(call.get("site") or instance.get("site_name") or "").strip().lower()
    instance_site = str(instance.get("site_name") or "").strip().lower()
    if site and instance_site and site != instance_site:
        raise EditorError(
            "site_mismatch",
            f"editor call site {site!r} does not match bound seed instance site {instance_site!r}",
        )
    key = (benchmark, site)
    editor = editor_instances.get(key)
    if editor is not None:
        _assert_seed_editor_site(editor, site)
        return editor
    registration = seed_registry.get(benchmark, site)
    if registration is None:
        raise EditorError(
            "unsupported_site",
            f"no editor registered for benchmark={benchmark!r} site={site!r}",
        )
    editor = registration.create(instance, session)
    # Retain the instance before validating the factory result so a malformed
    # registration still participates in the common rollback boundary.
    editor_instances[key] = editor
    _assert_seed_editor_site(editor, site)
    return editor


def _assert_seed_editor_site(editor: Any, site: str) -> None:
    editor_site = str(getattr(editor, "site_name", site) or "").strip().lower()
    if editor_site != site:
        raise EditorError(
            "site_mismatch",
            f"seed editor factory for site {site!r} produced editor for {editor_site!r}",
        )


def _apply_editor_seed_call(
    session: requests.Session,
    call: dict[str, Any],
    instance: dict[str, Any],
    *,
    call_index: int | None = None,
    seed_context: dict[str, Any],
    editor_instances: dict[tuple[str, str], Any],
    read_surface_accumulator: list[str] | None = None,
    read_surface_provenance: dict[str, Any] | None = None,
    created_resource_accumulator: list[dict[str, Any]] | None = None,
    editor_call_result_accumulator: list[dict[str, Any]] | None = None,
    seed_registry: SeedSiteRegistry,
    seed_token_scope: str,
) -> None:
    from datetime import UTC, datetime

    # Fail-loud: reject the call if it references a {benign_*} token that
    # the resolver's anchors don't support. Catches plans that pass the
    # legacy Option A validator (which only checks the innermost anchor)
    # but would render an empty string at substitution time.
    _assert_benign_tokens_bound(
        call,
        instance.get("seed_task"),
        seed_registry=seed_registry,
        seed_token_scope=seed_token_scope,
    )
    delay_s = _editor_call_pre_delay_s(call)
    if delay_s > 0:
        logger.info("Waiting %.2fs before applying ordered editor seed call", delay_s)
        time.sleep(delay_s)

    rendered = _render_editor_seed_call(call, seed_context)
    benchmark = _infer_editor_call_benchmark(rendered, instance)
    editor = _get_editor_for_seed_call(
        rendered,
        instance,
        session=session,
        editor_instances=editor_instances,
        seed_registry=seed_registry,
    )
    method_name = str(rendered["method"]).strip()
    args = rendered["args"]
    editor_site_name = str(getattr(editor, "site_name", rendered.get("site") or "")).strip()
    if method_name.startswith("_") or method_name not in editor.supported_methods:
        raise EditorError(
            "unsupported_method",
            f"{editor_site_name} editor does not support method {method_name!r}",
        )
    editor_method = getattr(editor, method_name, None)
    if not callable(editor_method):
        raise EditorError(
            "unsupported_method",
            f"{editor_site_name} editor does not support method {method_name!r}",
        )
    args = _filter_editor_method_args(
        editor_method, args, editor_site_name=editor_site_name, method_name=method_name
    )
    unresolved = sorted(
        _unresolved_structural_seed_placeholder_names(
            args,
            free_text_arg_names=_editor_free_text_arg_names(editor_method),
        )
    )
    if unresolved:
        raise RuntimeError(
            "editor call has unresolved template placeholders: " + ", ".join(unresolved)
        )
    editor.validate_args(method_name, args)
    result = editor_method(**args)
    if isinstance(result, dict):
        normalized_result = EditorSeedResult.from_mapping(
            result,
            editor_method=f"{editor_site_name}.{method_name}",
        )
        if editor_call_result_accumulator is not None and call_index is not None:
            editor_call_result_accumulator.append(
                _editor_call_result_record(
                    result,
                    call_index=call_index,
                    editor_site_name=editor_site_name,
                    method_name=method_name,
                    benchmark=benchmark,
                    logical_record_key=rendered.get("logical_record_key"),
                )
            )
        if created_resource_accumulator is not None:
            created_resource_accumulator.extend(
                resource.as_mapping() for resource in normalized_result.created_resources
            )
        # C1b read-surface URLs must NOT round-trip through seed_context
        # (namespace-flat; multi-call seeds would clobber each other — §12.9).
        surface_urls = normalized_result.read_surface_urls
        if read_surface_accumulator is not None:
            read_surface_accumulator.extend(surface_urls)
        if read_surface_provenance is not None and surface_urls:
            # Handoff §12.9: multi-call seeds (e.g. gitlab.create_project +
            # gitlab.create_issue) each contribute a method. Accumulate the
            # methods as a list (first-occurrence order, deduped); keep the
            # most-specific source seen so far (api_response beats
            # constructed); stamp captured_at only once on first contribution.
            provenance_source = (
                normalized_result.read_surface_provenance_source or "editor_api_response"
            )
            editor_method_str = f"{editor_site_name}.{method_name}"
            if not read_surface_provenance:
                read_surface_provenance.update(
                    {
                        "source": provenance_source,
                        "editor_method": [editor_method_str],
                        "captured_at": datetime.now(UTC).isoformat(),
                    }
                )
            else:
                methods = read_surface_provenance.get("editor_method")
                if not isinstance(methods, list):
                    methods = [str(methods)] if methods else []
                    read_surface_provenance["editor_method"] = methods
                if editor_method_str not in methods:
                    methods.append(editor_method_str)
                # api_response is the stronger claim — prefer it over constructed.
                current_source = read_surface_provenance.get("source")
                if (
                    current_source == "editor_constructed"
                    and provenance_source == "editor_api_response"
                ):
                    read_surface_provenance["source"] = provenance_source
        # Strip C1b-only keys before merging into seed_context so they do not
        # surface as placeholder values to later calls.
        if surface_urls is not None or "read_surface_provenance_source" in result:
            sanitized = {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "identity_tokens",
                    "read_surface_urls",
                    "read_surface_provenance_source",
                }
            }
            if sanitized:
                _merge_seed_context(seed_context, sanitized)
        else:
            sanitized = {k: v for k, v in result.items() if k != "identity_tokens"}
            _merge_seed_context(seed_context, sanitized)
        if normalized_result.write_tokens:
            _merge_seed_context(seed_context, dict(normalized_result.write_tokens))


__all__ = [
    "CreatedResourceFact",
    "EditorSeedResult",
    "ReadSurfaceFact",
    "SeedCleanupHandle",
    "SeedSiteRegistration",
    "SeedSiteRegistry",
    "_apply_editor_seed_call",
    "_get_editor_for_seed_call",
    "apply_data_seed",
    "apply_data_seed_async",
    "preflight_editor_seed_calls",
]
