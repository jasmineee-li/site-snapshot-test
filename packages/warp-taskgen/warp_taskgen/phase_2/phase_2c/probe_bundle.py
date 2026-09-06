"""Injectable collaborators for the Phase 2c verification loop.

The runner and per-task verifier accept every side-effecting dependency
through one frozen bundle instead of binding module globals at call time.
Production callers use :meth:`Phase2cProbeBundle.default`; tests build the
same bundle with fakes for the seams they exercise and never patch the loop.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from warp_taskgen.auth_tokens import acquire_tokens_for_instances
from warp_taskgen.phase_2.phase_2c import fingerprints as _fingerprints
from warp_taskgen.phase_2.phase_2c import probes as _probes
from warp_taskgen.phase_2.phase_2c import source_data_preflight as _source_data_preflight
from warp_taskgen.phase_2.phase_2c.retry_timing import phase_2c_retry_sleep
from warp_taskgen.phases.phase_2_reachability import ReachabilityOutcome, verify_reachable
from warp_taskgen.phases.phase_2_render_check import RenderOutcome, verify_seed_renders
from warp_taskgen.seeding import apply_data_seed_async


def _default_playwright_factory() -> Any:
    """Resolve Playwright lazily so render-disabled runs never import it."""

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "phase 2c render verification requires Playwright: install "
            "'playwright' and run 'playwright install chromium', or set "
            f"{_probes._SKIP_RENDER_CHECK_ENV}=1 to opt out (development only). "
            f"Underlying import error: {exc!r}"
        ) from exc
    return async_playwright()


@dataclass(frozen=True)
class Phase2cProbeBundle:
    """The collaborators ``verify_feasibility`` and ``_verify_one`` call.

    Every field is a callable the loop invokes with the same arguments the
    real sibling accepts. ``verify_seed_renders`` and ``verify_reachable`` are
    the leaf probes that ``render_check`` and ``reachability_check`` receive as
    keyword arguments, so a caller may replace either layer independently.
    ``playwright_factory`` returns the object whose ``start()`` yields a
    Playwright handle; ``None`` means no browser is available and the runner
    fails closed when render verification is enabled.
    """

    acquire_tokens: Callable[[list[dict[str, Any]]], list[str]]
    source_data_preflight: Callable[..., Awaitable[list[dict[str, Any]]]]
    apply_seed: Callable[..., Awaitable[tuple[Any, dict[str, Any]]]]
    render_check: Callable[..., Awaitable[RenderOutcome]]
    reachability_check: Callable[..., Awaitable[ReachabilityOutcome]]
    verify_seed_renders: Callable[..., Any]
    verify_reachable: Callable[..., Any]
    retry_sleep: Callable[[float], Awaitable[None]]
    host_fingerprint: Callable[[str, list[dict[str, Any]]], dict[str, str]]
    ensure_chromium_ready: Callable[[Any], Awaitable[None]]
    playwright_factory: Callable[[], Any] | None

    @classmethod
    def default(cls, **overrides: Any) -> Phase2cProbeBundle:
        """Wire the real siblings, then apply ``overrides`` by field name."""

        bundle = cls(
            acquire_tokens=acquire_tokens_for_instances,
            source_data_preflight=_source_data_preflight._run_preflight_and_filter_raw,
            apply_seed=apply_data_seed_async,
            render_check=_probes._run_render_check,
            reachability_check=_probes._run_reachability_check,
            verify_seed_renders=verify_seed_renders,
            verify_reachable=verify_reachable,
            retry_sleep=phase_2c_retry_sleep,
            host_fingerprint=_fingerprints._host_fingerprint,
            ensure_chromium_ready=_probes._ensure_playwright_chromium_ready,
            playwright_factory=_default_playwright_factory,
        )
        if overrides:
            bundle = dataclasses.replace(bundle, **overrides)
        return bundle


__all__ = ["Phase2cProbeBundle"]
