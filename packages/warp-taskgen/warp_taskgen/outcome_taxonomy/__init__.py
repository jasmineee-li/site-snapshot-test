"""Phase 4 outcome taxonomy classifier.

Post-processes a Phase 4 trajectory into a richer, diagnosable outcome ontology:

- 5 top-level outcomes (``complied_benign_succeeded``, ``complied_benign_failed``,
  ``resistant``, ``task_broke``, ``evaluator_unavailable``)
- ``resistant`` → ``resistant_aware`` / ``resistant_unaware``
- ``task_broke`` → 8 sub-classes (``self_abandoned``, ``wrong_answer``,
  ``partial_attack_engagement``, ``step_limit_exhausted``, ``agent_exception``,
  ``injection_unreached``, ``worker_timeout``, ``no_engagement``, ``other``)
- 7+ orthogonal flags
- ``diagnosable_confidence`` ∈ {high, medium, low}

Offline only: reads existing trajectory artifacts
(``history.json``, ``network_trace.json``, ``final_response.json``,
``processed_result.json``). Never calls a live platform.

Canonical derivation: ``docs/handoffs/codex-handoff-outcome-taxonomy.md``.
"""

from __future__ import annotations

from warp_taskgen.outcome_taxonomy.classification import classify
from warp_taskgen.outcome_taxonomy.io import classify_from_dir
from warp_taskgen.outcome_taxonomy.serialization import as_result_fields
from warp_taskgen.outcome_taxonomy.signals import (
    CLASSIFIER_VERSION,
    ClassifiedOutcome,
    TrajectorySignals,
    extract_signals,
)
from warp_taskgen.outcome_taxonomy.summary import (
    format_stratified_summary_log,
    stratified_summary,
)

__all__ = [
    "CLASSIFIER_VERSION",
    "ClassifiedOutcome",
    "TrajectorySignals",
    "as_result_fields",
    "classify",
    "classify_from_dir",
    "extract_signals",
    "format_stratified_summary_log",
    "stratified_summary",
]
