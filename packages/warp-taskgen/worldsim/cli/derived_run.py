"""Explicit CLI operation for materializing and resuming a Derived Run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from worldsim import run_materialization
from worldsim.atomic_io import write_json_atomic
from worldsim.cli.run_identity import resolve_cli_run_transition, resume_state_inputs
from worldsim.run_definition import define_run
from worldsim.run_materialization import DerivedRunContext
from worldsim.state import (
    LEGACY_STATE_DIR_ENV,
    RESUME_POINTER_ENV,
    STATE_DIR_ENV,
    bind_state_paths,
    get_state_dir,
    load_state,
)


def _load_source_state() -> dict[str, Any] | None:
    state = load_state()
    if state is None:
        return None
    # Keep the same normalized input view as ordinary resume. The authoritative
    # materializer still re-reads pipeline_state.json before writing anything.
    return resume_state_inputs(state)


def _child_pointer(context: DerivedRunContext) -> Path:
    """Create or validate the child-local discovery pointer."""

    state_path = context.child_root / "pipeline_state.json"
    pointer_path = context.child_root / "last_run_state.json"
    if pointer_path.is_symlink():
        raise ValueError("Derived Run child discovery pointer must not be a symlink")
    if pointer_path.exists() and not pointer_path.is_file():
        raise ValueError("Derived Run child discovery pointer must be a file")
    if pointer_path.exists():
        try:
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Derived Run child discovery pointer is unreadable") from exc
        if not isinstance(payload, dict):
            raise ValueError("Derived Run child discovery pointer must contain an object")
        if _pointer_target(payload) != state_path.resolve(strict=False):
            raise ValueError("Derived Run child discovery pointer targets a different state root")
        try:
            if define_run(payload) != context.definition:
                raise ValueError("Derived Run child discovery pointer has conflicting identity")
        except ValueError as exc:
            raise ValueError(
                "Derived Run child discovery pointer has conflicting identity"
            ) from exc
        return pointer_path

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Derived Run child checkpoint is unreadable") from exc
    if not isinstance(payload, dict) or define_run(payload) != context.definition:
        raise ValueError("Derived Run child checkpoint has conflicting identity")
    write_json_atomic(
        pointer_path,
        {**payload, "state_file": str(state_path.resolve(strict=False))},
        failpoint_base="run_materialization.child_pointer",
    )
    return pointer_path


def _pointer_target(payload: dict[str, Any]) -> Path | None:
    state_file = payload.get("state_file")
    logs_dir = payload.get("logs_dir")
    if state_file:
        return Path(str(state_file)).expanduser().resolve(strict=False)
    if logs_dir:
        return Path(str(logs_dir)).expanduser().resolve(strict=False) / "pipeline_state.json"
    return None


@contextmanager
def _bind_child_execution(context: DerivedRunContext, pointer: Path) -> Iterator[None]:
    """Route every child read/write, including legacy env consumers, locally."""

    previous = {
        STATE_DIR_ENV: os.environ.get(STATE_DIR_ENV),
        LEGACY_STATE_DIR_ENV: os.environ.get(LEGACY_STATE_DIR_ENV),
        RESUME_POINTER_ENV: os.environ.get(RESUME_POINTER_ENV),
    }
    os.environ[STATE_DIR_ENV] = str(context.child_root)
    os.environ[LEGACY_STATE_DIR_ENV] = str(context.child_root)
    os.environ[RESUME_POINTER_ENV] = str(pointer)
    try:
        with bind_state_paths(context.child_root, resume_pointer=pointer):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def dispatch_derived_resume(args: argparse.Namespace) -> int:
    """Materialize and dispatch one isolated child for explicit operator intent."""

    try:
        state = _load_source_state()
    except ValueError as exc:
        print(f"Derived Run request rejected by Run Definition: {exc}", file=sys.stderr)
        return 2
    if state is None:
        print("No pipeline state found; run a phase first.", file=sys.stderr)
        return 1

    try:
        transition = resolve_cli_run_transition(args, existing_state=state)
    except ValueError as exc:
        print(f"Derived Run request rejected by Run Definition: {exc}", file=sys.stderr)
        return 2
    if transition.kind == "legacy":
        print(
            "Derived Run requires an identified source; legacy runs receive no invented identity.",
            file=sys.stderr,
        )
        return 2
    if transition.kind != "derived_required":
        print(
            "Derived Run request rejected: explicit derivation requires result-affecting drift.",
            file=sys.stderr,
        )
        return 2

    source_root = Path(str(state.get("logs_dir") or get_state_dir()))
    try:
        context = run_materialization.materialize_derived_run(source_root, transition)
        pointer = _child_pointer(context)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Derived Run materialization failed: {exc}", file=sys.stderr)
        return 2

    action = "Created" if context.created else "Reused"
    print(
        f"{action} isolated Derived Run {context.definition.run_id}; "
        f"executing from {context.child_root}."
    )
    child_args = argparse.Namespace(**vars(args))
    child_args.command = "resume"
    # The persisted definition normalizes order-insensitive ``sites`` to a
    # list, while the phase adapter's historical CLI surface accepts the
    # comma-separated spelling. Keep that compatibility conversion local to
    # this feature-owned child dispatch.
    if getattr(child_args, "sites", None) is None:
        sites = context.definition.input_projection().get("sites")
        if isinstance(sites, (list, tuple)):
            child_args.sites = ",".join(str(site) for site in sites)
    with _bind_child_execution(context, pointer):
        # Import lazily to keep the feature adapter independent of the CLI
        # compatibility facade during parser construction.
        from worldsim.cli import _impl

        try:
            return _impl._dispatch_resume(child_args)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Derived Run child dispatch failed: {exc}", file=sys.stderr)
            return 2


__all__ = ["dispatch_derived_resume"]
