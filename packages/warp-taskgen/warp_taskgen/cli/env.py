"""Import-time environment bootstrap for the WARP Taskgen CLI.

Loads ``.env`` before any CLI sibling imports ``warp_taskgen.config`` and
mirrors canonical WARP Taskgen variables into their legacy names. Reloading
this module re-runs the bootstrap, which ``tests/test_cli_env_precedence.py``
relies on.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

_STATE_DIR_ENV = "WARP_TASKGEN_STATE_DIR"
_LEGACY_STATE_DIR_ENV = "WORLDSIM_STATE_DIR"
_REMOTE_EXPLICIT_STATE_DIR_ENV = "WARP_TASKGEN_REMOTE_STATE_DIR_EXPLICIT"


def _load_dotenv_preserving_remote_state_dir() -> None:
    """Load dotenv while retaining a remote wrapper's explicit state root."""
    explicit_marker = os.environ.get(_REMOTE_EXPLICIT_STATE_DIR_ENV)
    explicit_state_dir = os.environ.get(_STATE_DIR_ENV)
    explicit_legacy_state_dir = os.environ.get(_LEGACY_STATE_DIR_ENV)

    # Keep the existing dotenv precedence for every ordinary setting. Remote
    # launchers mark the two aliases as an explicit pair so only that pair is
    # restored after dotenv has loaded.
    load_dotenv(override=True)

    if (
        explicit_marker == "1"
        and explicit_state_dir
        and explicit_state_dir == explicit_legacy_state_dir
    ):
        os.environ[_STATE_DIR_ENV] = explicit_state_dir
        os.environ[_LEGACY_STATE_DIR_ENV] = explicit_state_dir
        os.environ.pop(_REMOTE_EXPLICIT_STATE_DIR_ENV, None)


_load_dotenv_preserving_remote_state_dir()


def _normalize_compat_env_aliases() -> None:
    """Mirror canonical WARP Taskgen env vars into legacy names for older helpers."""
    aliases = {
        "WARP_TASKGEN_STATE_DIR": "WORLDSIM_STATE_DIR",
        "WARP_TASKGEN_AGENTLAB_RUNNER_CMD": "WORLDSIM_AGENTLAB_RUNNER_CMD",
    }
    for canonical, legacy in aliases.items():
        if os.environ.get(canonical) and not os.environ.get(legacy):
            os.environ[legacy] = os.environ[canonical]


__all__ = ["_load_dotenv_preserving_remote_state_dir", "_normalize_compat_env_aliases"]
