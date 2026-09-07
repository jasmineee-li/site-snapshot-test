"""JSON snapshots for the Phase 2a and Phase 2b request archives.

Feature owners supply metadata, paths, failure handling, and attempt identity.
This module neither binds a context nor changes model dispatch or admission.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from warp_taskgen.atomic_io import write_json_atomic

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


def archive_model_request(
    kwargs: dict[str, Any],
    *,
    metadata: dict[str, Any],
    path: Path,
    run_root: Path,
    write_json: Callable[[Path, Any], None] = write_json_atomic,
) -> dict[str, Any]:
    """Snapshot model-facing kwargs; let the feature report any retention failure."""
    omitted_fields = sorted(kwargs.keys() - _REQUEST_FIELDS)
    unknown_fields = sorted(kwargs.keys() - _REQUEST_FIELDS - _TRANSPORT_FIELDS)
    encoded = json.dumps(
        {key: value for key, value in kwargs.items() if key in _REQUEST_FIELDS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_request_json_default,
        allow_nan=False,
    )
    request_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    relative_path = path.relative_to(run_root).as_posix()
    envelope = {
        **metadata,
        "request_sha256": request_hash,
        "omitted_argument_names": omitted_fields,
        "request": json.loads(encoded),
    }
    write_json(path, envelope)
    reference: dict[str, Any] = {
        "status": "partial_retention" if unknown_fields else "retained",
        "path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "request_sha256": request_hash,
    }
    if unknown_fields:
        reference["unknown_argument_names"] = unknown_fields
    return reference


def _request_json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError("request contains a non-JSON value")
