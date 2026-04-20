from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from worldsim.phase_4.encounter_detection import (
    EncounterResult,
    determine_encounter,
)


def _rect(x: int, y: int, w: int, h: int) -> dict[str, int]:
    return {"x": x, "y": y, "w": w, "h": h}


def _make_png_with_ink(
    ink_boxes: list[tuple[int, int, int, int]],
    *,
    size: tuple[int, int] = (400, 200),
    bg: tuple[int, int, int] = (255, 255, 255),
    ink_color: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    """Build a deterministic PNG: ``bg`` everywhere except ``ink_boxes`` which
    are filled with ``ink_color``. Used to verify ink-occupancy classification."""
    img = Image.new("RGB", size, bg)
    for x, y, w, h in ink_boxes:
        for ix in range(x, x + w):
            for iy in range(y, y + h):
                if 0 <= ix < size[0] and 0 <= iy < size[1]:
                    img.putpixel((ix, iy), ink_color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_step(
    trajectory_dir: Path,
    step_idx: int,
    visibility_vec: list[dict[str, Any]],
    png_bytes: bytes,
    *,
    clip: dict[str, int] | None = None,
    bg: tuple[int, int, int] = (255, 255, 255),
    has_damage: bool = True,
) -> None:
    shots = trajectory_dir / "screenshots"
    pvpo = trajectory_dir / "pvpo"
    shots.mkdir(parents=True, exist_ok=True)
    pvpo.mkdir(parents=True, exist_ok=True)
    (shots / f"step_{step_idx}.png").write_bytes(png_bytes)
    (pvpo / f"step_{step_idx}.json").write_text(
        json.dumps(
            {
                "step_idx": step_idx,
                "visibility_vec": visibility_vec,
                "background_color": list(bg),
                "has_damage": has_damage,
                "clip": clip or {"x": 0, "y": 0, "w": 400, "h": 200},
            },
            indent=2,
        )
    )


def _task(payload: str) -> dict[str, Any]:
    return {
        "id": "adv_t1",
        "site": "shopping_admin",
        "payload_texts": [{"rendered_payload": payload}],
    }


def _visible_entry(i: int, char: str, live: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        "i": i,
        "char": char,
        "isSpace": False,
        "layoutVisible": True,
        "liveRect": _rect(*live),
    }


def _hidden_entry(i: int, char: str) -> dict[str, Any]:
    return {
        "i": i,
        "char": char,
        "isSpace": False,
        "layoutVisible": False,
        "liveRect": None,
    }


def _space_entry(i: int, char: str = " ") -> dict[str, Any]:
    return {
        "i": i,
        "char": char,
        "isSpace": True,
        "layoutVisible": False,
        "liveRect": None,
    }


def test_empty_trajectory_returns_zero_coverage(tmp_path: Path):
    result = determine_encounter(_task("abc"), tmp_path)
    assert result.max_coverage == 0.0
    assert result.reference_step is None
    assert result.reference_screenshot_path is None
    assert result.per_step_coverage == []


def test_empty_payload_does_not_crash(tmp_path: Path):
    png = _make_png_with_ink([])
    _write_step(tmp_path, 0, [], png)
    result = determine_encounter(_task(""), tmp_path)
    assert result.max_coverage == 0.0
    assert result.reference_step is None


def test_single_step_full_coverage_with_ink(tmp_path: Path):
    # Two chars: both layout-visible AND have ink pixels inside their rects.
    png = _make_png_with_ink([(10, 10, 5, 5), (20, 10, 5, 5)])
    vec = [
        _visible_entry(0, "a", (10, 10, 5, 5)),
        _visible_entry(1, "b", (20, 10, 5, 5)),
    ]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("ab"), tmp_path)
    assert result.max_coverage == 1.0
    assert result.reference_step == 0
    assert result.reference_screenshot_path == tmp_path / "screenshots" / "step_0.png"
    assert result.per_step_coverage == [1.0]
    assert result.per_char_visibility == [[True, True]]


def test_layout_visible_but_no_ink_is_zero_coverage(tmp_path: Path):
    """A character whose rect is all background pixels (e.g. color==bg trick)
    must be reported as not-rendered even if ``layoutVisible=True``. This is
    the one case where ink-occupancy is strictly stronger than layout-only
    visibility."""
    png = _make_png_with_ink([])  # all white
    vec = [
        _visible_entry(0, "a", (10, 10, 5, 5)),
        _visible_entry(1, "b", (20, 10, 5, 5)),
    ]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("ab"), tmp_path)
    assert result.max_coverage == 0.0
    assert result.reference_step is None


def test_multi_step_picks_max_coverage_step(tmp_path: Path):
    def _vec(visible: set[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, c in enumerate("abcd"):
            if i in visible:
                out.append(_visible_entry(i, c, (10 + 10 * i, 10, 5, 5)))
            else:
                out.append(_hidden_entry(i, c))
        return out

    def _png_with_ink_for(indices: list[int]) -> bytes:
        return _make_png_with_ink([(10 + 10 * k, 10, 5, 5) for k in indices])

    _write_step(tmp_path, 0, _vec({0}), _png_with_ink_for([0]))
    _write_step(tmp_path, 1, _vec({0, 1, 2}), _png_with_ink_for([0, 1, 2]))
    _write_step(tmp_path, 2, _vec({2, 3}), _png_with_ink_for([2, 3]))

    result = determine_encounter(_task("abcd"), tmp_path)
    assert result.per_step_coverage == [0.25, 0.75, 0.5]
    assert result.max_coverage == 0.75
    assert result.reference_step == 1
    assert result.reference_screenshot_path == tmp_path / "screenshots" / "step_1.png"


@pytest.mark.parametrize(
    "visible_indices,payload,expected",
    [
        ({0}, "abcdefgh", 0.125),
        ({0, 1, 2, 3}, "abcdefgh", 0.5),
        ({0, 1, 2, 3, 4, 5, 6}, "abcdefgh", 0.875),
        ({0, 1, 2, 3, 4, 5, 6, 7}, "abcdefgh", 1.0),
    ],
)
def test_continuous_coverage_preserved_without_bucketing(
    tmp_path: Path, visible_indices: set[int], payload: str, expected: float
):
    ink_boxes = [(10 + 10 * k, 10, 5, 5) for k in visible_indices]
    png = _make_png_with_ink(ink_boxes)
    vec: list[dict[str, Any]] = []
    for i, c in enumerate(payload):
        if i in visible_indices:
            vec.append(_visible_entry(i, c, (10 + 10 * i, 10, 5, 5)))
        else:
            vec.append(_hidden_entry(i, c))
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task(payload), tmp_path)
    assert result.max_coverage == pytest.approx(expected)
    assert isinstance(result.max_coverage, float)


def test_whitespace_excluded_from_denominator(tmp_path: Path):
    png = _make_png_with_ink([(10, 10, 5, 5)])
    vec = [
        _visible_entry(0, "a", (10, 10, 5, 5)),
        _space_entry(1),
        _hidden_entry(2, "b"),
    ]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("a b"), tmp_path)
    assert result.max_coverage == pytest.approx(0.5)


def test_boundary_zero_coverage_no_reference_step(tmp_path: Path):
    png = _make_png_with_ink([])
    vec = [_hidden_entry(0, "a"), _hidden_entry(1, "b")]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("ab"), tmp_path)
    assert result.max_coverage == 0.0
    assert result.reference_step is None
    assert result.reference_screenshot_path is None


def test_boundary_just_above_zero_emits_reference_step(tmp_path: Path):
    png = _make_png_with_ink([(10, 10, 5, 5)])
    vec = [
        _visible_entry(0, "a", (10, 10, 5, 5)),
        *[_hidden_entry(i, c) for i, c in enumerate("bcdefgh", start=1)],
    ]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("abcdefgh"), tmp_path)
    assert result.max_coverage == pytest.approx(0.125)
    assert result.reference_step == 0
    assert result.reference_screenshot_path is not None


def test_missing_png_for_pvpo_raises(tmp_path: Path):
    (tmp_path / "pvpo").mkdir(parents=True)
    (tmp_path / "pvpo" / "step_0.json").write_text(
        json.dumps(
            {
                "visibility_vec": [],
                "background_color": [255, 255, 255],
                "has_damage": True,
                "clip": {"x": 0, "y": 0, "w": 10, "h": 10},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="paired screenshot"):
        determine_encounter(_task("ab"), tmp_path)


def test_reference_png_vanishing_after_loop_forces_zero_coverage(tmp_path: Path):
    """Concurrent sweep of screenshots dir between loop and gate: route conservatively.

    The per-step loop verifies each PNG exists. A downstream cleanup/resume
    could then delete the reference PNG before the gate's final sanity
    check reads it. Rather than passing the gate and letting
    transcript_purpose_api fail with ``missing_screenshot``, the detector
    should force ``max_coverage=0`` so the trajectory routes to
    placement-fix — the correct failure class for "no reference evidence
    available."
    """
    png = _make_png_with_ink([(10, 10, 5, 5)])
    vec = [_visible_entry(0, "a", (10, 10, 5, 5))]
    _write_step(tmp_path, 0, vec, png)
    png_path = tmp_path / "screenshots" / "step_0.png"

    # Simulate the race: let the per-step loop read the file successfully,
    # then delete it before the final sanity check. We wrap read_bytes to
    # delete the file after the loop reads its contents — so everything the
    # loop needs succeeds, but the Path.is_file() re-check at the end sees
    # a missing file.
    from worldsim.phase_4 import encounter_detection as _ed

    original_read_bytes = Path.read_bytes
    deleted = {"done": False}

    def read_then_delete(self: Path) -> bytes:  # type: ignore[override]
        data = original_read_bytes(self)
        if self == png_path and not deleted["done"]:
            self.unlink()
            deleted["done"] = True
        return data

    import unittest.mock

    with unittest.mock.patch.object(Path, "read_bytes", read_then_delete):
        result = _ed.determine_encounter(_task("a"), tmp_path)

    assert deleted["done"] is True
    assert result.max_coverage == 0.0
    assert result.reference_step is None
    assert result.reference_screenshot_path is None


def test_encounter_result_as_dict_json_serializable(tmp_path: Path):
    png = _make_png_with_ink([(10, 10, 5, 5)])
    vec = [_visible_entry(0, "a", (10, 10, 5, 5))]
    _write_step(tmp_path, 0, vec, png)
    result = determine_encounter(_task("a"), tmp_path)

    payload = result.as_dict()
    dumped = json.dumps(payload)
    round_trip = json.loads(dumped)
    assert round_trip["max_coverage"] == 1.0
    assert round_trip["reference_step"] == 0
    assert isinstance(round_trip["reference_screenshot_path"], str)
    assert round_trip["per_char_visibility"] == [[True]]
    assert round_trip["per_step_coverage"] == [1.0]


def test_as_dict_handles_none_reference_path():
    result = EncounterResult(
        max_coverage=0.0,
        reference_step=None,
        reference_screenshot_path=None,
    )
    payload = result.as_dict()
    assert payload["reference_screenshot_path"] is None
    assert payload["reference_step"] is None
    json.dumps(payload)


def test_step_files_enumerated_numerically_not_lexically(tmp_path: Path):
    for idx in (0, 1, 2, 10):
        _write_step(tmp_path, idx, [], _make_png_with_ink([]))
    result = determine_encounter(_task(""), tmp_path)
    assert len(result.per_step_coverage) == 4


def test_corrupt_pvpo_json_skipped_with_other_steps_intact(tmp_path: Path):
    """Finding 1: a single malformed step JSON must not abort the trajectory.

    The bad step is dropped (logged as warning); good steps still
    contribute to ``per_step_coverage`` and ``max_coverage``.
    """
    png = _make_png_with_ink([(10, 10, 5, 5)])
    vec = [_visible_entry(0, "a", (10, 10, 5, 5))]
    _write_step(tmp_path, 0, vec, png)
    # step_1 has the PNG but a malformed JSON sibling.
    (tmp_path / "screenshots" / "step_1.png").write_bytes(png)
    (tmp_path / "pvpo" / "step_1.json").write_text("{not valid json")
    _write_step(tmp_path, 2, vec, png)

    result = determine_encounter(_task("a"), tmp_path)
    # Two good steps (0 and 2), step 1 was skipped without raising.
    assert result.per_step_coverage == [1.0, 1.0]
    assert result.max_coverage == 1.0
    # Reference step must point at one of the kept steps, not at the
    # skipped step 1.
    assert result.reference_step in {0, 2}
    assert result.reference_screenshot_path is not None
    assert result.reference_screenshot_path.name in {"step_0.png", "step_2.png"}


def test_truncated_png_skipped_with_other_steps_intact(tmp_path: Path):
    """Finding 1: truncated PNG raises ``OSError`` from PIL load; that
    step is skipped, neighbours unaffected."""
    png = _make_png_with_ink([(10, 10, 5, 5)])
    vec = [_visible_entry(0, "a", (10, 10, 5, 5))]
    _write_step(tmp_path, 0, vec, png)
    # step_1: valid JSON, but the PNG is one byte (PIL can't open it).
    (tmp_path / "screenshots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "screenshots" / "step_1.png").write_bytes(b"\x89")
    (tmp_path / "pvpo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pvpo" / "step_1.json").write_text(
        json.dumps(
            {
                "step_idx": 1,
                "visibility_vec": [
                    {
                        "i": 0,
                        "char": "a",
                        "isSpace": False,
                        "layoutVisible": True,
                        "liveRect": _rect(10, 10, 5, 5),
                    }
                ],
                "background_color": [255, 255, 255],
                "has_damage": True,
                "clip": {"x": 0, "y": 0, "w": 400, "h": 200},
            }
        )
    )
    _write_step(tmp_path, 2, vec, png)

    result = determine_encounter(_task("a"), tmp_path)
    assert result.per_step_coverage == [1.0, 1.0]
    assert result.max_coverage == 1.0
    assert result.reference_step in {0, 2}


def test_non_white_background_color_from_step_json(tmp_path: Path):
    """Ink-occupancy must compare against the step's recorded background,
    not a hardcoded white. A grey-bg page with darker-grey text still has
    ink above the ``delta`` threshold."""
    png = _make_png_with_ink(
        [(10, 10, 5, 5)],
        bg=(200, 200, 200),
        ink_color=(50, 50, 50),  # Δ = 450 per pixel, well above delta=24
    )
    vec = [_visible_entry(0, "a", (10, 10, 5, 5))]
    _write_step(tmp_path, 0, vec, png, bg=(200, 200, 200))

    result = determine_encounter(_task("a"), tmp_path)
    assert result.max_coverage == 1.0
