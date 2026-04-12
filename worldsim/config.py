"""Benchmark configuration schema.

Matches the JSON example in the v5 spec "Prerequisites" section. The user
hands the orchestrator a BenchmarkConfig; the orchestrator never starts,
stops, or provisions benchmark instances — it only connects to the URLs,
DBs, and reset endpoints supplied here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class BenchmarkInstance(BaseModel):
    """One pre-running benchmark instance.

    For parallel evaluation, supply multiple instances of the same site on
    different ports. Workers are distributed across them in
    ``worldsim.eval_worker_pool.run_eval``.
    """

    site_name: str = Field(..., description="E.g. 'shopping', 'forum', 'gitlab'.")
    site_url: str = Field(..., description="Base URL of the running service.")
    db_connection: Optional[str] = Field(
        None,
        description="Connection string for data seeding, e.g. 'mysql://user:pass@host:3306/db'. "
        "Optional: not every seeding mechanism needs a DB (see worldsim.seeding).",
    )
    reset_endpoint: Optional[str] = Field(
        None,
        description="HTTP endpoint called between tasks to restore initial state. "
        "Optional but strongly recommended for Phase 3/4 reproducibility.",
    )


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration supplied by the user."""

    benchmark_name: str
    instances: list[BenchmarkInstance]
    benchmark_codebase: Path = Field(
        ...,
        description="Absolute path to the benchmark's source tree. Used in "
        "Phase 0 (read-only reconnaissance). Must exist on disk.",
    )
