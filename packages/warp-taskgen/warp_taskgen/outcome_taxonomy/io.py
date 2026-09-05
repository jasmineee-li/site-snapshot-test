"""Disk-backed outcome classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from warp_taskgen.outcome_taxonomy.classification import classify
from warp_taskgen.outcome_taxonomy.signals import ClassifiedOutcome, extract_signals
from warp_taskgen.outcome_taxonomy.trajectory_io import _load_history

__all__ = [
    "classify_from_dir",
]


# ---------------------------------------------------------------------------
# High-level helper used by both the pipeline and the reclassifier CLI
# ---------------------------------------------------------------------------


def classify_from_dir(
    trajectory_dir: Path,
    task: dict[str, Any],
    *,
    benign_passed: bool | None = None,
    adversarial_passed: bool | None = None,
    ecologically_valid: bool | None = None,
    seed_ok: bool | None = None,
    max_steps: int = 50,
) -> ClassifiedOutcome:
    """One-shot helper: extract signals from disk + classify."""
    trajectory_dir = Path(trajectory_dir)
    signals = extract_signals(
        trajectory_dir,
        task,
        benign_passed=benign_passed,
        adversarial_passed=adversarial_passed,
        ecologically_valid=ecologically_valid,
        seed_ok=seed_ok,
        max_steps=max_steps,
    )
    history = _load_history(trajectory_dir / "history.json")
    return classify(signals, task, history=history)
