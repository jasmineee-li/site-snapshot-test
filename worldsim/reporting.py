"""Cross-benchmark comparison reporting.

Reads Phase 3-4 result files across multiple benchmarks and produces
a comparison table showing benign completion rate, baseline ASR,
WorldSim ASR, and the delta for each benchmark.

Usage::

    from worldsim.reporting import generate_comparison_table
    table = generate_comparison_table(results_dir)
    print(table.to_markdown())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldsim.task_paths import safe_task_path_component

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for one benchmark + attack mode combination."""

    benchmark: str
    attack_mode: str  # "baseline", "worldsim", or "benign"
    total_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    attack_succeeded: int = 0  # Phase 4: agent complied with attack
    attack_resisted: int = 0   # Phase 4: agent resisted attack
    ecologically_valid: int = 0  # Phase 4: passed Gate 1

    @property
    def completion_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    @property
    def asr(self) -> float:
        """Attack success rate (complied / total ecologically valid)."""
        if self.ecologically_valid == 0:
            return 0.0
        return self.attack_succeeded / self.ecologically_valid

    @property
    def resistance_rate(self) -> float:
        """Agent resistance rate (resisted / total ecologically valid)."""
        if self.ecologically_valid == 0:
            return 0.0
        return self.attack_resisted / self.ecologically_valid


@dataclass
class ComparisonTable:
    """Cross-benchmark comparison data."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def add_row(
        self,
        benchmark: str,
        benign_cr: float,
        baseline_asr: float | None,
        worldsim_asr: float,
        **extra: Any,
    ) -> None:
        delta = (worldsim_asr - baseline_asr) if baseline_asr is not None else None
        self.rows.append({
            "benchmark": benchmark,
            "benign_cr": benign_cr,
            "baseline_asr": baseline_asr,
            "worldsim_asr": worldsim_asr,
            "delta": delta,
            **extra,
        })

    def to_markdown(self) -> str:
        """Render as a Markdown table."""
        if not self.rows:
            return "No comparison data available."

        header = "| Benchmark | Benign CR | Baseline ASR | WorldSim ASR | Delta |"
        sep = "|-----------|----------|-------------|-------------|-------|"
        lines = [header, sep]

        for row in self.rows:
            benign = f"{row['benign_cr']:.1%}"
            baseline = f"{row['baseline_asr']:.1%}" if row["baseline_asr"] is not None else "N/A"
            worldsim = f"{row['worldsim_asr']:.1%}"
            delta = f"+{row['delta']:.1%}" if row["delta"] is not None and row["delta"] > 0 else (
                f"{row['delta']:.1%}" if row["delta"] is not None else "N/A"
            )
            lines.append(f"| {row['benchmark']} | {benign} | {baseline} | {worldsim} | {delta} |")

        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return rows as a list of dicts for JSON serialization."""
        return list(self.rows)


def load_phase_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load canonical Phase 3/4 results without counting nested reruns.

    Preferred input is the phase summary file or directory containing
    ``results.json``. As a fallback, read only top-level per-task
    ``result.json`` sentinels whose directory name matches the task id's
    canonical path component.
    """
    results_dir = Path(results_dir)

    summary_path = results_dir if results_dir.is_file() else results_dir / "results.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed summary file %s: %s", summary_path, exc)
        else:
            if isinstance(data, list):
                return data
            logger.warning("Summary file %s did not contain a JSON array", summary_path)

    results: list[dict[str, Any]] = []
    if not results_dir.exists() or not results_dir.is_dir():
        return results

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        result_file = subdir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed result file %s: %s", result_file, exc)
            continue
        task_id = str(data.get("task_id") or "")
        if not task_id:
            continue
        if subdir.name != safe_task_path_component(task_id):
            continue
        results.append(data)
    return results


def compute_metrics(
    results: list[dict[str, Any]],
    benchmark: str,
    attack_mode: str,
) -> BenchmarkMetrics:
    """Compute aggregate metrics from a list of task results."""
    metrics = BenchmarkMetrics(benchmark=benchmark, attack_mode=attack_mode)
    for r in results:
        metrics.total_tasks += 1
        if r.get("passed"):
            metrics.passed_tasks += 1
        else:
            metrics.failed_tasks += 1

        # Phase 4 fields (may not be present for Phase 3 benign results)
        outcome = r.get("outcome")
        if outcome == "complied":
            metrics.attack_succeeded += 1
        elif outcome in ("refused_or_ignored", "resisted"):
            metrics.attack_resisted += 1

        if r.get("ecologically_valid", False):
            metrics.ecologically_valid += 1

    return metrics


def generate_comparison_table(
    benchmarks: dict[str, dict[str, Path]],
) -> ComparisonTable:
    """Generate a cross-benchmark comparison table.

    Args:
        benchmarks: mapping of benchmark name to a dict with keys:
            - "benign": Path to Phase 3 results directory
            - "baseline": Path to Phase 4 baseline results directory (optional)
            - "worldsim": Path to Phase 4 WorldSim results directory

    Returns:
        A ComparisonTable with one row per benchmark.
    """
    table = ComparisonTable()

    for benchmark, paths in sorted(benchmarks.items()):
        benign_results = load_phase_results(paths["benign"]) if "benign" in paths else []
        baseline_results = load_phase_results(paths["baseline"]) if "baseline" in paths else []
        worldsim_results = load_phase_results(paths["worldsim"]) if "worldsim" in paths else []

        benign_metrics = compute_metrics(benign_results, benchmark, "benign")
        baseline_metrics = compute_metrics(baseline_results, benchmark, "baseline") if baseline_results else None
        worldsim_metrics = compute_metrics(worldsim_results, benchmark, "worldsim")

        table.add_row(
            benchmark=benchmark,
            benign_cr=benign_metrics.completion_rate,
            baseline_asr=baseline_metrics.asr if baseline_metrics else None,
            worldsim_asr=worldsim_metrics.asr,
            total_tasks=benign_metrics.total_tasks,
        )

    return table
