"""Serialization for embedding taxonomy results in result records."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from warp_taskgen.outcome_taxonomy.signals import ClassifiedOutcome

__all__ = [
    "as_result_fields",
]


def as_result_fields(classified: ClassifiedOutcome) -> dict[str, Any]:
    """Serialize a ``ClassifiedOutcome`` for embedding in ``processed_result.json``."""
    return {
        "outcome_fine": classified.outcome_fine,
        "flags": list(classified.flags),
        "diagnosable_confidence": classified.diagnosable_confidence,
        "signals": asdict(classified.signals) if classified.signals else None,
        "classifier_version": classified.classifier_version,
        "classifier_rationale": classified.rationale,
    }
