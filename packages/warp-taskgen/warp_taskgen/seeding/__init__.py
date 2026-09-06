"""Public package surface for data seeding.

Behavior lives in the sibling modules: :mod:`~warp_taskgen.seeding.execution`
(editor dispatch and the cleanup boundary), :mod:`~warp_taskgen.seeding.context`
(seed context and template rendering), :mod:`~warp_taskgen.seeding.tokens`
(benign token binding), :mod:`~warp_taskgen.seeding.validation` (schema
validation), :mod:`~warp_taskgen.seeding.site_contracts` (the per-run
``SeedSiteRegistry``), and the reddit/map context, result-metadata,
runtime-error, editor-argument, and database helpers.  Import a private helper
from the module that owns it rather than from this surface.
"""

from __future__ import annotations

from warp_taskgen.editors import EditorError
from warp_taskgen.seeding.context import _build_seed_context
from warp_taskgen.seeding.contracts import (
    seed_has_actions,
    seed_requires_reset,
    self_contained_adversarial_seed_error,
)
from warp_taskgen.seeding.execution import (
    SeedCleanupHandle,
    apply_data_seed,
    apply_data_seed_async,
    preflight_editor_seed_calls,
)
from warp_taskgen.seeding.reddit_context import (
    _resolve_reddit_forum,
    _resolve_reddit_submission_id,
)
from warp_taskgen.seeding.runtime_errors import collect_seed_runtime_errors
from warp_taskgen.seeding.site_contracts import (
    CreatedResourceFact,
    EditorSeedResult,
    ReadSurfaceFact,
    SeedSiteRegistration,
    SeedSiteRegistry,
    default_seed_registry,
)
from warp_taskgen.seeding.tokens import (
    UnboundTokenError,
    _assert_benign_tokens_bound,
    _collect_benign_tokens,
)
from warp_taskgen.seeding.validation import validate_data_seed

__all__ = [
    "CreatedResourceFact",
    "EditorError",
    "EditorSeedResult",
    "ReadSurfaceFact",
    "SeedCleanupHandle",
    "SeedSiteRegistration",
    "SeedSiteRegistry",
    "UnboundTokenError",
    "_assert_benign_tokens_bound",
    "_build_seed_context",
    "_collect_benign_tokens",
    "_resolve_reddit_forum",
    "_resolve_reddit_submission_id",
    "apply_data_seed",
    "apply_data_seed_async",
    "collect_seed_runtime_errors",
    "default_seed_registry",
    "preflight_editor_seed_calls",
    "seed_has_actions",
    "seed_requires_reset",
    "self_contained_adversarial_seed_error",
    "validate_data_seed",
]
