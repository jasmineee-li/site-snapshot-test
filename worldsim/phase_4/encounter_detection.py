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
from urllib.parse import urlparse

from worldsim.phase_4.ink_occupancy import ink_occupancy_vector
from worldsim.phase_4.pvpo_capture import Rect, load_capture_summary

logger = logging.getLogger(__name__)


_STEP_FILENAME_RE = re.compile(r"^step_(\d+)\.json$")
_DEFAULT_BG: tuple[int, int, int] = (255, 255, 255)


@dataclass
class EncounterResult:
    """Output of :func:`determine_encounter`.

    ``max_coverage`` is a continuous float in ``[0.0, 1.0]`` and is the
    primary encounter signal. Stage 2 reasons over it directly; no internal
    threshold is applied here.

    ``reference_screenshot_bytes`` is populated when the strict-validation
    block below accepts the reference PNG; the bytes are read at validation
    time so no second file-open (and no second symlink/resolve check) is
    required in the downstream transcript_purpose API call. This closes the
    TOCTOU window between ``determine_encounter`` and the base64-encode in
    :func:`worldsim.phase_4.transcript_purpose_api.run_transcript_purpose_api`.
    Excluded from :meth:`as_dict` to keep the JSON payload small (the path is
    already serialized for forensics).
    """

    max_coverage: float
    reference_step: int | None
    reference_screenshot_path: Path | None
    per_char_visibility: list[list[bool]] = field(default_factory=list)
    per_step_coverage: list[float] = field(default_factory=list)
    reference_screenshot_bytes: bytes | None = field(default=None, repr=False)
    pvpo_status: str = "ok"
    pvpo_failure: str | None = None
    pvpo_steps_seen: int = 0
    pvpo_steps_captured: int = 0
    pvpo_issue_steps: int = 0
    pvpo_artifact_steps: int = 0
    pvpo_skipped_steps: int = 0
    pvpo_match_found_steps: int = 0
    pvpo_matched_witness_ids: list[str] = field(default_factory=list)

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
            "pvpo_status": self.pvpo_status,
            "pvpo_failure": self.pvpo_failure,
            "pvpo_steps_seen": self.pvpo_steps_seen,
            "pvpo_steps_captured": self.pvpo_steps_captured,
            "pvpo_issue_steps": self.pvpo_issue_steps,
            "pvpo_artifact_steps": self.pvpo_artifact_steps,
            "pvpo_skipped_steps": self.pvpo_skipped_steps,
            "pvpo_match_found_steps": self.pvpo_match_found_steps,
            "pvpo_matched_witness_ids": list(self.pvpo_matched_witness_ids),
            "coverage_basis": "attack_witness",
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
    if _has_invalid_selected_payload_index(task):
        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
            pvpo_status="no_payload",
            pvpo_failure="invalid_selected_payload_index",
        )
    payload = _extract_payload(task)
    non_space_total = sum(1 for c in payload if not c.isspace())

    pvpo_dir = trajectory_dir / "pvpo"
    screenshots_dir = trajectory_dir / "screenshots"
    step_files = _enumerate_step_files(pvpo_dir)
    capture_summary = load_capture_summary(trajectory_dir) or {}
    steps_seen = _summary_int(capture_summary, "steps_seen")
    steps_captured = _summary_int(capture_summary, "steps_captured")
    issue_steps = _summary_int(capture_summary, "issue_steps")
    summary_status = _summary_str(capture_summary, "status") or "unknown"
    summary_failure = _summary_str(capture_summary, "first_issue_class")

    per_step_coverage: list[float] = []
    per_char_visibility: list[list[bool]] = []
    kept_step_files: list[tuple[int, Path]] = []
    skipped_steps = 0
    match_found_steps = 0
    matched_witness_ids: list[str] = []
    off_surface_match_steps = 0

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
            skipped_steps += 1
            continue

        visibility_vec = pvpo_json.get("visibility_vec") or []
        match_found = bool(pvpo_json.get("match_found"))
        if match_found and not _pvpo_page_url_allowed(task, pvpo_json.get("page_url")):
            match_found = False
            visibility_vec = []
            off_surface_match_steps += 1
        if match_found:
            match_found_steps += 1
        witness_id = pvpo_json.get("matched_witness_id")
        if isinstance(witness_id, str) and witness_id and witness_id not in matched_witness_ids:
            matched_witness_ids.append(witness_id)
        clip = _parse_clip(pvpo_json.get("clip"))
        bg = _parse_bg(pvpo_json.get("background_color"))
        if "match_found" in pvpo_json and not match_found:
            per_step_coverage.append(0.0)
            kept_step_files.append((step_idx, pvpo_path))
            continue

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
            skipped_steps += 1
            continue

        per_char_visibility.append(paint_vec)
        rendered = sum(1 for v in paint_vec if v)
        step_non_space_total = sum(
            1
            for entry in visibility_vec
            if isinstance(entry, dict) and not bool(entry.get("isSpace"))
        )
        denominator = step_non_space_total or non_space_total
        # Clamp to [0, 1] to preserve the invariant even when the Python
        # ``str.isspace()`` denominator and the JS ``isSpace`` numerator
        # disagree on exotic Unicode (ZWSP U+200B, narrow NBSP U+202F,
        # combining marks). Real-world WebArena payloads are ASCII so the
        # clamp is usually a no-op, but documenting and enforcing the invariant
        # avoids a silent ``max_coverage=1.0001`` in the paper's numbers.
        coverage = rendered / denominator if denominator else 0.0
        if coverage > 1.0:
            coverage = 1.0
        per_step_coverage.append(coverage)
        kept_step_files.append((step_idx, pvpo_path))

    if not per_step_coverage:
        status, failure = _derive_pvpo_status(
            summary_status=summary_status,
            summary_failure=summary_failure,
            step_file_count=len(step_files),
            skipped_steps=skipped_steps,
            retained_steps=0,
        )
        return EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
            pvpo_status=status,
            pvpo_failure=failure,
            pvpo_steps_seen=steps_seen,
            pvpo_steps_captured=steps_captured,
            pvpo_issue_steps=issue_steps,
            pvpo_artifact_steps=len(step_files),
            pvpo_skipped_steps=skipped_steps,
            pvpo_match_found_steps=match_found_steps,
            pvpo_matched_witness_ids=matched_witness_ids,
        )

    max_coverage = max(per_step_coverage)
    reference_step: int | None
    reference_path: Path | None
    reference_bytes: bytes | None = None
    status, failure = _derive_pvpo_status(
        summary_status=summary_status,
        summary_failure=summary_failure,
        step_file_count=len(step_files),
        skipped_steps=skipped_steps,
        retained_steps=len(per_step_coverage),
    )
    if max_coverage > 0.0:
        argmax = per_step_coverage.index(max_coverage)
        reference_step = kept_step_files[argmax][0]
        candidate = screenshots_dir / f"step_{reference_step}.png"
        # Defensive re-check: the per-step loop above already verified this
        # PNG exists, but a concurrent sweep of the screenshots dir (cleanup,
        # resume/retry) could delete it between the loop and the downstream
        # gate. If the file has vanished, conservatively route to
        # placement-fix by forcing max_coverage to 0 rather than passing the
        # gate with inconsistent PVPO encounter evidence.
        #
        # Reject symlinks and paths that resolve outside the screenshots dir:
        # ``Path.is_file()`` follows symlinks, so a symlink planted in the
        # screenshots directory (accidental from a cleanup script, or a
        # resume path reusing another run's artefacts via a link) would
        # otherwise silently substitute arbitrary filesystem content for
        # PVPO encounter evidence.
        #
        # We also eagerly read the bytes here so the Stage 2 gate receives an
        # immutable snapshot rather than a path it would have to re-validate.
        # Re-validation in a
        # separate process step leaves a TOCTOU window where a symlink
        # could be planted between the check and the ``read_bytes``; reading
        # here closes that window entirely.
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
            try:
                reference_bytes = candidate.read_bytes()
            except OSError as exc:
                logger.warning(
                    "pvpo: reference screenshot %s became unreadable during "
                    "validation (%s); forcing max_coverage=0",
                    candidate,
                    exc,
                )
                reference_bytes = None
        if is_real_file and reference_bytes:
            reference_path = candidate
        else:
            if is_real_file and not reference_bytes:
                # Strict checks passed but the file was empty; treat the same
                # as a vanished / escaped-dir file for downstream routing.
                logger.warning(
                    "pvpo: reference screenshot %s validated but is empty; "
                    "forcing max_coverage=0 to route to placement-fix",
                    candidate,
                )
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
            reference_bytes = None
            status = "degraded"
            failure = "reference_screenshot_invalid"
    else:
        reference_step = None
        reference_path = None
        if match_found_steps > 0 and max_coverage == 0.0:
            failure = failure or "payload_witness_not_painted"
        elif match_found_steps == 0 and len(step_files) > 0:
            failure = failure or (
                "payload_witness_off_surface"
                if off_surface_match_steps
                else "payload_witness_not_matched"
            )

    return EncounterResult(
        max_coverage=max_coverage,
        reference_step=reference_step,
        reference_screenshot_path=reference_path,
        per_char_visibility=per_char_visibility,
        per_step_coverage=per_step_coverage,
        reference_screenshot_bytes=reference_bytes,
        pvpo_status=status,
        pvpo_failure=failure,
        pvpo_steps_seen=steps_seen,
        pvpo_steps_captured=steps_captured,
        pvpo_issue_steps=issue_steps,
        pvpo_artifact_steps=len(step_files),
        pvpo_skipped_steps=skipped_steps,
        pvpo_match_found_steps=match_found_steps,
        pvpo_matched_witness_ids=matched_witness_ids,
    )


def _extract_payload(task: dict[str, Any]) -> str:
    payloads = task.get("payload_texts") or []
    if not payloads:
        return ""
    selected_index = task.get("selected_payload_index", 0)
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(payloads)):
        logger.warning(
            "pvpo: invalid selected_payload_index=%r for task %s; refusing to fall back to payload 0",
            selected_index,
            task.get("id", "unknown"),
        )
        return ""
    selected = payloads[selected_index] or {}
    rendered = selected.get("rendered_payload") or ""
    return str(rendered)


def _has_invalid_selected_payload_index(task: dict[str, Any]) -> bool:
    payloads = task.get("payload_texts") or []
    if not payloads:
        return False
    selected_index = task.get("selected_payload_index", 0)
    return not isinstance(selected_index, int) or not (0 <= selected_index < len(payloads))


def _pvpo_page_url_allowed(task: dict[str, Any], page_url: Any) -> bool:
    if not isinstance(page_url, str) or not page_url:
        return True
    parsed = urlparse(page_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return True
    allowed = _allowed_pvpo_origins(task)
    if not allowed:
        return True
    return (parsed.scheme, parsed.netloc.lower()) in allowed


def _allowed_pvpo_origins(task: dict[str, Any]) -> set[tuple[str, str]]:
    urls: list[Any] = []
    for key in ("site_url", "start_url", "start_url_resolved", "benign_read_url"):
        urls.append(task.get(key))
    read_surface_urls = task.get("read_surface_urls")
    if isinstance(read_surface_urls, list):
        urls.extend(read_surface_urls)

    contract = task.get("exposure_contract")
    if isinstance(contract, dict):
        urls.extend(
            [
                contract.get("benign_read_url"),
                (contract.get("verification") or {}).get("url")
                if isinstance(contract.get("verification"), dict)
                else None,
                ((contract.get("verification") or {}).get("entry") or {}).get("url")
                if isinstance((contract.get("verification") or {}).get("entry"), dict)
                else None,
            ]
        )

    feasibility = task.get("feasibility")
    if isinstance(feasibility, dict):
        exposure = feasibility.get("exposure")
        if isinstance(exposure, dict):
            verification = exposure.get("verification")
            if isinstance(verification, dict):
                urls.append(verification.get("url"))

    resource = task.get("benign_target_resource")
    if isinstance(resource, dict):
        for key in ("site_url", "start_url", "start_url_resolved", "benign_read_url"):
            urls.append(resource.get(key))

    origins: set[tuple[str, str]] = set()
    for url in urls:
        if not isinstance(url, str) or not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            origins.add((parsed.scheme, parsed.netloc.lower()))
    return origins


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


def _summary_int(summary: dict[str, Any], key: str) -> int:
    try:
        return int(summary.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _summary_str(summary: dict[str, Any], key: str) -> str | None:
    value = summary.get(key)
    return value if isinstance(value, str) and value else None


def _derive_pvpo_status(
    *,
    summary_status: str,
    summary_failure: str | None,
    step_file_count: int,
    skipped_steps: int,
    retained_steps: int,
) -> tuple[str, str | None]:
    if summary_status == "disabled_no_payload":
        return ("disabled_no_payload", None)
    if skipped_steps > 0 and retained_steps == 0:
        return ("degraded", summary_failure or "no_valid_pvpo_steps")
    if summary_status == "degraded":
        return ("degraded", summary_failure or "pvpo_capture_degraded")
    if step_file_count == 0:
        if summary_status == "initialized":
            return ("no_artifacts", summary_failure or "pvpo_no_artifacts")
        return ("no_artifacts", summary_failure)
    if skipped_steps > 0:
        return ("degraded", summary_failure or "partial_pvpo_artifact_loss")
    return ("ok", None)
