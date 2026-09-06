"""Prospective Phase 2b request evidence in the existing Run artifact tree.

Capture JSON request arguments immediately before Instructor calls the SDK.
These are model-facing SDK inputs, not HTTP wire bytes or authorization headers.
Retention is observational: failed writes are explicit diagnostics, never a new
checkpoint/admission rule or a reason to retry a model call.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from warp_taskgen.atomic_io import write_json_atomic

logger = logging.getLogger(__name__)
_ARCHIVE_PATHS: ContextVar[tuple[Path, Path] | None] = ContextVar(
    "phase_2b_request_archive", default=None
)
# Only model-facing SDK arguments. Never serialize SDK/client state, headers,
# credentials, callbacks, or transport options with a permissive repr fallback.
_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "thinking",
        "metadata",
        "stream",
        "service_tier",
        "container",
        "context_management",
        "output_config",
        "cache_control",
        "inference_geo",
        "betas",
        "extra_body",
    }
)
_TRANSPORT_FIELDS = frozenset({"extra_headers", "extra_query", "timeout"})


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
            omitted_fields = sorted(kwargs.keys() - _REQUEST_FIELDS)
            unknown_fields = sorted(kwargs.keys() - _REQUEST_FIELDS - _TRANSPORT_FIELDS)
            # Snapshot before Instructor mutates messages to construct a reask.
            encoded = json.dumps(
                {key: value for key, value in kwargs.items() if key in _REQUEST_FIELDS},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=_request_json_default,
                allow_nan=False,
            )
            request_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            path = request_dir / call_id / f"{attempt}.json"
            relative_path = path.relative_to(run_root).as_posix()
            envelope = {
                "schema_version": 1,
                "phase": "phase_2b",
                "call_id": call_id,
                "request_index": attempt,
                "task_id": task_id,
                "site": site,
                "configured_model": configured_model,
                "resolved_client_provider": client_provider,
                "request_sha256": request_hash,
                "omitted_argument_names": omitted_fields,
                "request": json.loads(encoded),
            }
            write_json_atomic(path, envelope)
            reference.update(
                status="partial_retention" if unknown_fields else "retained",
                path=relative_path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                request_sha256=request_hash,
            )
            if unknown_fields:
                reference["unknown_argument_names"] = unknown_fields
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


def _request_json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError("request contains a non-JSON value")
