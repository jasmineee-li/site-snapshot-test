"""Phase 0: Benchmark reconnaissance.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 0: Benchmark
Reconnaissance" section.

Phase 0 has three sub-steps:

- **0a — Benchmark Discovery.** Single Modal Sandbox with the full benchmark
  source. Produces ``BENCHMARK_MANIFEST.json`` + ``.md``.
- **0b — Sandbox Filesystem Mapping.** Pure local Python (no LLM, no network).
  Computes the exact file list for each site's sandbox based on the manifest.
- **0c — Per-Site Profiling.** N parallel Modal Sandboxes, one per site,
  profiling verification capabilities, data model, injection surface, and
  existing task coverage.

Not yet implemented. Scheduled for commits 5–7 after the modal sandbox
primitive smoke test passes in commit 3.
"""

from __future__ import annotations

from pathlib import Path


async def run(benchmark: Path, sub: str = "0") -> int:
    """Phase 0 entrypoint.

    Args:
        benchmark: Path to the benchmark codebase (e.g.
            ``vendors/webarena-infinity``).
        sub: One of ``"0"`` (full phase), ``"0a"``, ``"0b"``, or ``"0c"``.

    Returns:
        Process exit code.
    """
    raise NotImplementedError(
        f"Phase 0 sub={sub!r} is scheduled for commits 5-7 "
        f"(see docs/worldsim-v5-technical-specifcation.md 'Phase 0: Benchmark Reconnaissance')"
    )


# TODO(commit 5): async def run_phase_0a(benchmark_root: Path, output_dir: Path) -> dict: ...
# TODO(commit 6): def compute_sandbox_maps(manifest: dict, benchmark_root: Path) -> dict: ...
# TODO(commit 7): async def run_phase_0c(manifest, sandbox_map, benchmark_root, output_dir) -> dict: ...
