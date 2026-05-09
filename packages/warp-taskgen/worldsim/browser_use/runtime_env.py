"""Runtime environment defaults for embedded Browser Use runs."""

from __future__ import annotations

import os


def _ensure_browser_use_runtime_env() -> None:
    """Default Browser Use to local-only runtime behavior.

    WorldSim runs Browser Use as an embedded evaluator component. Cloud sync
    and anonymous telemetry are not part of the benchmark contract, and in live
    r5 runs the telemetry client can leave a non-daemon worker thread behind
    after Phase 4 has already written complete artifacts. Keep these defaults
    local-only unless a caller explicitly opts back in through the environment.
    """
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("BROWSER_USE_CLOUD_SYNC", "false")
    os.environ.setdefault("POSTHOG_DISABLED", "true")
    # High-concurrency remote runs can exceed Browser Use's default event-bus
    # budgets during the initial navigation/DOM-state burst. Keep these as
    # overridable runtime defaults so launch scripts do not need to remember
    # every Browser Use timeout knob.
    os.environ.setdefault("TIMEOUT_NavigateToUrlEvent", "45.0")
    os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", "60.0")
    os.environ.setdefault("TIMEOUT_BrowserConnectedEvent", "60.0")
