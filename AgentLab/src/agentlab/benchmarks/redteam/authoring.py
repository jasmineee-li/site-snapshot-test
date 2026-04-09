"""Controller-facing authoring helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam import app_pipeline


def load_prompt_template(name: str, **kwargs: str) -> str:
    return app_pipeline.load_prompt_template(name, **kwargs)


def run_prompt(
    *,
    prompt: str,
    working_dir: str | Path,
    timeout: int,
) -> tuple[int, str, str]:
    return app_pipeline.run_claude_code(
        prompt=prompt,
        working_dir=working_dir,
        timeout=timeout,
    )


def ensure_trusted_template(
    app_dir: str | Path,
    template_dir: str | Path | None = None,
) -> str | None:
    return app_pipeline.ensure_trusted_server_template(app_dir, template_dir)


def generation_prompt(
    behavior_spec: dict[str, Any],
    design_guides_dir: str | Path | None = None,
) -> str:
    return app_pipeline._build_generation_prompt(behavior_spec, design_guides_dir)
