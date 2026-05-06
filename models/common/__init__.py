"""Shared utilities for local CUA model adapters."""

from models.common.registry import (
    LOCAL_MODELS,
    LocalModelSpec,
    is_local_model_id,
    resolve_local_model,
    strip_local_prefix,
)

__all__ = [
    "LOCAL_MODELS",
    "LocalModelSpec",
    "is_local_model_id",
    "resolve_local_model",
    "strip_local_prefix",
]
