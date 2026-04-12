"""Benchmark configuration schema.

Matches the JSON example in the v5 spec "Prerequisites" section. The user
hands the orchestrator a BenchmarkConfig; the orchestrator never starts,
stops, or provisions benchmark instances — it only connects to the URLs,
DBs, and reset endpoints supplied here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from worldsim.placeholders import merge_placeholder_maps


class BenchmarkInstance(BaseModel):
    """One pre-running benchmark instance.

    For parallel evaluation, supply multiple instances of the same site on
    different ports. Workers are distributed across them in
    ``worldsim.eval_worker_pool.run_eval``.
    """

    site_name: str = Field(..., description="E.g. 'shopping', 'forum', 'gitlab'.")
    site_url: str = Field(..., description="Base URL of the running service.")
    db_connection: str | None = Field(
        None,
        description="Connection string for data seeding, e.g. 'mysql://user:pass@host:3306/db'. "
        "Optional: not every seeding mechanism needs a DB (see worldsim.seeding).",
    )
    reset_endpoint: str | None = Field(
        None,
        description="HTTP endpoint called between tasks to restore initial state. "
        "Optional but strongly recommended for Phase 3/4 reproducibility.",
    )
    url_placeholders: dict[str, str] = Field(
        default_factory=dict,
        description="Maps URL template tokens (e.g. __SHOPPING__) to actual URLs. "
        "Used for resolving cross-site URL references in reward evaluation.",
    )


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration supplied by the user."""

    benchmark_name: str
    instances: list[BenchmarkInstance]
    url_placeholders: dict[str, str] = Field(
        default_factory=dict,
        description="Top-level mapping of URL template tokens to live site URLs. "
        "This is the canonical config shape from the v5 spec.",
    )
    benchmark_codebase: Path = Field(
        ...,
        description="Absolute path to the benchmark's source tree. Used in "
        "Phase 0 (read-only reconnaissance). Must exist on disk.",
    )

    @model_validator(mode="after")
    def _merge_legacy_instance_placeholders(self) -> BenchmarkConfig:
        """Preserve backwards compatibility with older per-instance configs."""
        self.url_placeholders = merge_placeholder_maps(
            self.url_placeholders,
            *[instance.url_placeholders for instance in self.instances],
        )
        return self
