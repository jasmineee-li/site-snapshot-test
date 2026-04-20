"""PVPO Stage 1 — encounter detection over per-step capture artifacts.

Reads the per-step artifacts written by
:func:`worldsim.phase_4.pvpo_capture.save_step_artifacts`:

    trajectory_dir / pvpo        / step_{N}.json   # visibility_vec + bg + clip + hasDamage
    trajectory_dir / screenshots / step_{N}.png    # atomic beginFrame PNG

For each step, runs :func:`worldsim.phase_4.ink_occupancy.ink_occupancy_vector`
over the ``(png, visibility_vec, background_color, clip)`` tuple, producing
per-character booleans where ``True`` means the character was both layout-
visible and painted at non-background ink density above the ink-occupancy
thresholds. Aggregates to:

- ``per_step_coverage[k] = rendered_k / non_space_total``
- ``max_coverage = max(per_step_coverage)``
- ``reference_step`` = argmax when ``max_coverage > 0``, else ``None``.

No threshold, no tier inside the detector. The score is preserved end-to-
end as a float. The only discrete split is binary routing on
``max_coverage == 0`` vs ``> 0``, made by the caller. Handoff §3.6, §5.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldsim.phase_4.ink_occupancy import ink_occupancy_vector
from worldsim.phase_4.pvpo_capture import Rect

logger = logging.getLogger(__name__)


_STEP_FILENAME_RE = re.compile(r"^step_(\d+)\.json$")
_DEFAULT_BG: tuple[int, int, int] = (255, 255, 255)


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
            ``task["payload_texts"][selected_payload_index]["rendered_payload"]``
            when ``selected_payload_index`` is present, otherwise the first
            payload.
        trajectory_dir: per-task directory with ``pvpo/step_*.json`` and
            ``screenshots/step_*.png`` subdirectories.

    Raises:
        FileNotFoundError: when a ``pvpo/step_N.json`` exists but its
            paired ``screenshots/step_N.png`` is missing. This is a
            ``save_step_artifacts`` contract violation, not an
            on-disk corruption symptom, so we still raise to surface it.
            Per-file parse / decode corruption (partial JSON write,
            truncated PNG) is logged at warning and the offending step
            is skipped — see Finding 1 in
            ``docs/todo-pvpo-post-ship-review.md``.

    Returns:
        :class:`EncounterResult`. Empty trajectories yield
        ``max_coverage=0.0, reference_step=None``; empty payload strings
        the same.
    """
    trajectory_dir = Path(trajectory_dir)
    payload = _extract_payload(task)
    non_space_total = sum(1 for c in payload if not c.isspace())

    pvpo_dir = trajectory_dir / "pvpo"
    screenshots_dir = trajectory_dir / "screenshots"
    step_files = _enumerate_step_files(pvpo_dir)

    per_step_coverage: list[float] = []
    per_char_visibility: list[list[bool]] = []
    kept_step_files: list[tuple[int, Path]] = []

    for step_idx, pvpo_path in step_files:
        png_path = screenshots_dir / f"step_{step_idx}.png"
        if not png_path.is_file():
            raise FileNotFoundError(
                f"pvpo artifact {pvpo_path} exists but paired screenshot "
                f"{png_path} is missing; capture is inconsistent"
            )

        try:
            pvpo_json = json.loads(pvpo_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "pvpo: skipping step %d — could not parse %s: %s",
                step_idx,
                pvpo_path,
                exc,
            )
            continue

        visibility_vec = pvpo_json.get("visibility_vec") or []
        clip = _parse_clip(pvpo_json.get("clip"))
        bg = _parse_bg(pvpo_json.get("background_color"))

        try:
            png_bytes = png_path.read_bytes()
            paint_vec = ink_occupancy_vector(
                png_bytes, visibility_vec, background_color=bg, clip=clip
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "pvpo: skipping step %d — could not decode %s: %s",
                step_idx,
                png_path,
                exc,
            )
            continue

        per_char_visibility.append(paint_vec)
        rendered = sum(1 for v in paint_vec if v)
        # Clamp to [0, 1] to preserve the invariant even when the Python
        # ``str.isspace()`` denominator and the JS ``isSpace`` numerator
        # disagree on exotic Unicode (ZWSP U+200B, narrow NBSP U+202F,
        # combining marks). Real-world WebArena payloads are ASCII so the
        # clamp is usually a no-op, but documenting and enforcing the invariant
        # avoids a silent ``max_coverage=1.0001`` in the paper's numbers.
        coverage = rendered / non_space_total if non_space_total else 0.0
        if coverage > 1.0:
            coverage = 1.0
        per_step_coverage.append(coverage)
        kept_step_files.append((step_idx, pvpo_path))

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
        reference_step = kept_step_files[argmax][0]
        candidate = screenshots_dir / f"step_{reference_step}.png"
        # Defensive re-check: the per-step loop above already verified this
        # PNG exists, but a concurrent sweep of the screenshots dir (cleanup,
        # resume/retry) could delete it between the loop and the downstream
        # gate. If the file has vanished, conservatively route to
        # placement-fix by forcing max_coverage to 0 rather than passing the
        # gate and failing later in transcript_purpose_api with a less
        # specific ``missing_screenshot``.
        #
        # Reject symlinks and paths that resolve outside the screenshots dir:
        # ``Path.is_file()`` follows symlinks, so a symlink planted in the
        # screenshots directory (accidental from a cleanup script, or a
        # resume path reusing another run's artefacts via a link) would
        # otherwise silently exfiltrate arbitrary filesystem content into
        # the Transcript Purpose API request body.
        screenshots_root = screenshots_dir.resolve()
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, FileNotFoundError):
            resolved = None
        is_real_file = (
            resolved is not None
            and resolved.is_file()
            and not candidate.is_symlink()
            and resolved.is_relative_to(screenshots_root)
        )
        if is_real_file:
            reference_path = candidate
        else:
            logger.warning(
                "pvpo: reference screenshot %s is not a regular file inside %s "
                "(symlink / dangling / vanished / escaped dir); forcing "
                "max_coverage=0 to route to placement-fix",
                candidate,
                screenshots_root,
            )
            max_coverage = 0.0
            reference_step = None
            reference_path = None
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
    selected_index = task.get("selected_payload_index", 0)
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(payloads)):
        selected_index = 0
    selected = payloads[selected_index] or {}
    rendered = selected.get("rendered_payload") or ""
    return str(rendered)


def _enumerate_step_files(pvpo_dir: Path) -> list[tuple[int, Path]]:
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


def _parse_bg(bg: Any) -> tuple[int, int, int]:
    if not bg:
        return _DEFAULT_BG
    try:
        r, g, b = bg[0], bg[1], bg[2]
        return (int(r), int(g), int(b))
    except (IndexError, KeyError, TypeError, ValueError):
        return _DEFAULT_BG
