"""Runtime-validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam import app_pipeline


def validate_runtime(
    app_dir: str | Path,
    port: int | None = None,
) -> dict[str, Any]:
    return app_pipeline.validate_app(app_dir, port=port)
