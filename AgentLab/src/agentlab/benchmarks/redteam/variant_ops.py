"""Variant generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam import app_pipeline


VariantGenerationResult = app_pipeline.VariantGenerationResult


def generate_variants_result(
    app_dir: str | Path,
    behavior_spec: dict[str, Any],
    design_guides_dir: str | Path | None = None,
) -> VariantGenerationResult:
    return app_pipeline._generate_variants_result(
        app_dir=app_dir,
        behavior_spec=behavior_spec,
        design_guides_dir=design_guides_dir,
    )
