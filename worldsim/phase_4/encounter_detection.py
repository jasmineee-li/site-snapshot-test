"""PVPO Stage 1 — encounter detection over all trajectory steps.

Reads the per-step artifacts written by :mod:`worldsim.phase_4.pvpo_capture`
under ``trajectory_dir``:

    trajectory_dir / pvpo        / step_{N}.json   # visibility_vec + clip + hasDamage
    trajectory_dir / screenshots / step_{N}.png    # atomic beginFrame PNG

For each step, runs :func:`pvpo_capture.paint_verified_vector` to compute the
per-character boolean vector, then aggregates:

- ``per_step_coverage[k] = rendered_k / non_space_total`` where ``rendered_k``
  is the sum of ``True`` entries in step ``k`` and ``non_space_total`` is the
  number of non-whitespace characters in the seeded payload.
- ``max_coverage = max(per_step_coverage)`` (or ``0.0`` when the trajectory
  has no captures at all, e.g. the agent crashed before the first step).
- ``reference_step`` = argmax of ``per_step_coverage`` when ``max_coverage > 0``.

The detector does not discretize. There is no threshold inside Stage 1; the
only discrete split is the binary ``max_coverage == 0`` vs ``> 0`` routing,
and that split is made by the caller (``run_adversarial_task``) to decide
whether to route to placement-fix or run Stage 2 P(eval). Handoff §3.6, §5.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldsim.phase_4.pvpo_capture import Rect, paint_verified_vector

logger = logging.getLogger(__name__)


_STEP_FILENAME_RE = re.compile(r"^step_(\d+)\.json$")


@dataclass
class EncounterResult:
    """Output of :func:`determine_encounter`.

    ``max_coverage`` is a continuous float in ``[0.0, 1.0]`` and is the
    primary encounter signal. Stage 2 reasons over it directly; no internal
    threshold is applied here.
    """

    max_coverage: float
    reference_step: int | None
    reference_screenshot_path: Path | None
    per_char_visibility: list[list[bool]] = field(default_factory=list)
    per_step_coverage: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form for ``result.json`` persistence."""
        return {
            "max_coverage": self.max_coverage,
            "reference_step": self.reference_step,
            "reference_screenshot_path": (
                str(self.reference_screenshot_path)
                if self.reference_screenshot_path is not None
                else None
            ),
            "per_char_visibility": [list(step) for step in self.per_char_visibility],
            "per_step_coverage": list(self.per_step_coverage),
        }


def determine_encounter(task: dict[str, Any], trajectory_dir: Path) -> EncounterResult:
    """Aggregate per-step PVPO artifacts into an :class:`EncounterResult`.

    Args:
        task: the task dict. The seeded payload is read from
            ``task["payload_texts"][0]["rendered_payload"]``.
        trajectory_dir: per-task directory with ``pvpo/step_*.json`` and
            ``screenshots/step_*.png`` subdirectories.

    Raises:
        FileNotFoundError: when a ``pvpo/step_N.json`` file references a step
            whose paired ``screenshots/step_N.png`` is missing (or vice
            versa). The pair must be consistent — silent defaults would mask
            capture bugs.

    Returns:
        An :class:`EncounterResult`. Empty trajectories (no pvpo dir, no step
        files) yield ``max_coverage=0.0, reference_step=None``. Empty payload
        strings yield the same.
    """
    trajectory_dir = Path(trajectory_dir)
    payload = _extract_payload(task)
    non_space_total = sum(1 for c in payload if not c.isspace())

    pvpo_dir = trajectory_dir / "pvpo"
    screenshots_dir = trajectory_dir / "screenshots"

    step_files = _enumerate_step_files(pvpo_dir)

    per_step_coverage: list[float] = []
    per_char_visibility: list[list[bool]] = []

    for step_idx, pvpo_path in step_files:
        png_path = screenshots_dir / f"step_{step_idx}.png"
        if not png_path.is_file():
            raise FileNotFoundError(
                f"pvpo artifact {pvpo_path} exists but paired screenshot "
                f"{png_path} is missing; capture is inconsistent"
            )

        pvpo_json = json.loads(pvpo_path.read_text(encoding="utf-8"))
        visibility_vec = pvpo_json.get("visibility_vec") or []
        clip = _parse_clip(pvpo_json.get("clip"))
        png_bytes = png_path.read_bytes()

        paint_vec = paint_verified_vector(png_bytes, visibility_vec, clip=clip)
        per_char_visibility.append(paint_vec)

        rendered = sum(1 for v in paint_vec if v)
        coverage = rendered / non_space_total if non_space_total else 0.0
        per_step_coverage.append(coverage)

    if not per_step_coverage:
        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
        )

    max_coverage = max(per_step_coverage)
    reference_step: int | None
    reference_path: Path | None
    if max_coverage > 0.0:
        argmax = per_step_coverage.index(max_coverage)
        reference_step = step_files[argmax][0]
        reference_path = screenshots_dir / f"step_{reference_step}.png"
    else:
        reference_step = None
        reference_path = None

    return EncounterResult(
        max_coverage=max_coverage,
        reference_step=reference_step,
        reference_screenshot_path=reference_path,
        per_char_visibility=per_char_visibility,
        per_step_coverage=per_step_coverage,
    )


def _extract_payload(task: dict[str, Any]) -> str:
    payloads = task.get("payload_texts") or []
    if not payloads:
        return ""
    first = payloads[0] or {}
    rendered = first.get("rendered_payload") or ""
    return str(rendered)


def _enumerate_step_files(pvpo_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(step_idx, path)`` pairs sorted by numeric index."""
    if not pvpo_dir.is_dir():
        return []
    pairs: list[tuple[int, Path]] = []
    for entry in pvpo_dir.iterdir():
        m = _STEP_FILENAME_RE.match(entry.name)
        if not m:
            continue
        pairs.append((int(m.group(1)), entry))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _parse_clip(clip: dict[str, Any] | None) -> Rect | None:
    if not clip:
        return None
    try:
        return Rect(x=int(clip["x"]), y=int(clip["y"]), w=int(clip["w"]), h=int(clip["h"]))
    except (KeyError, TypeError, ValueError):
        return None
