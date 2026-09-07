"""Prospective Phase 2b request evidence in the existing Run artifact tree.

Capture JSON request arguments immediately before Instructor calls the SDK.
These are model-facing SDK inputs, not HTTP wire bytes or authorization headers.
Retention is observational: failed writes are explicit diagnostics, never a new
checkpoint/admission rule or a reason to retry a model call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import uuid4

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.host_api_request_archive import archive_model_request

logger = logging.getLogger(__name__)
_ARCHIVE_PATHS: ContextVar[tuple[Path, Path] | None] = ContextVar(
    "phase_2b_request_archive", default=None
)


@contextmanager
def bind_text_fill_request_archive(run_root: Path, request_dir: Path) -> Iterator[None]:
    """Bind only the admitted task's async context; standalone APIs opt in."""

    token = _ARCHIVE_PATHS.set((run_root, request_dir))
    try:
        yield
    finally:
        _ARCHIVE_PATHS.reset(token)


def text_fill_request_recorder(
    *, task_id: str, site: str, configured_model: str, client_provider: str | None
) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    paths = _ARCHIVE_PATHS.get()
    if paths is None:
        return None
    run_root, request_dir = paths
    call_id = uuid4().hex
    attempt = 0

    def record(kwargs: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempt
        attempt += 1
        reference: dict[str, Any] = {"call_id": call_id, "request_index": attempt}
        try:
            reference.update(
                archive_model_request(
                    kwargs,
                    metadata={
                        "schema_version": 1,
                        "phase": "phase_2b",
                        "call_id": call_id,
                        "request_index": attempt,
                        "task_id": task_id,
                        "site": site,
                        "configured_model": configured_model,
                        "resolved_client_provider": client_provider,
                    },
                    path=request_dir / call_id / f"{attempt}.json",
                    run_root=run_root,
                    write_json=write_json_atomic,
                )
            )
            if reference["status"] == "partial_retention":
                logger.warning(
                    "Phase 2b request retention incomplete, call=%s attempt=%d", call_id, attempt
                )
        except Exception as exc:
            # Hooks swallow observer exceptions. Record failure explicitly and
            # do not log kwargs, credentials, or exception text containing them.
            reference.update(status="retention_failed", error_type=type(exc).__name__)
            logger.warning(
                "Phase 2b request retention failed (%s), call=%s attempt=%d",
                type(exc).__name__,
                call_id,
                attempt,
            )
        return reference

    return record
