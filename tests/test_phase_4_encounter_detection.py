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


def _make_png(
    boxes: dict[tuple[int, int, int, int], tuple[int, int, int]],
    *,
    size: tuple[int, int] = (400, 200),
) -> bytes:
    img = Image.new("RGB", size, (255, 255, 255))
    for (x, y, w, h), color in boxes.items():
        for ix in range(x, x + w):
            for iy in range(y, y + h):
                if 0 <= ix < size[0] and 0 <= iy < size[1]:
                    img.putpixel((ix, iy), color)
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


def _visible_entry(
    i: int, char: str, live: tuple[int, int, int, int], ref: tuple[int, int, int, int]
) -> dict[str, Any]:
    return {
        "i": i,
        "char": char,
        "isSpace": False,
        "layoutVisible": True,
        "liveRect": _rect(*live),
        "refRect": _rect(*ref),
    }


def _hidden_entry(i: int, char: str) -> dict[str, Any]:
    return {
        "i": i,
        "char": char,
        "isSpace": False,
        "layoutVisible": False,
        "liveRect": None,
        "refRect": None,
    }


def _space_entry(i: int, char: str = " ") -> dict[str, Any]:
    return {
        "i": i,
        "char": char,
        "isSpace": True,
        "layoutVisible": False,
        "liveRect": None,
        "refRect": None,
    }


def test_empty_trajectory_returns_zero_coverage(tmp_path: Path):
    result = determine_encounter(_task("abc"), tmp_path)
    assert result.max_coverage == 0.0
    assert result.reference_step is None
    assert result.reference_screenshot_path is None
    assert result.per_step_coverage == []


def test_empty_payload_does_not_crash(tmp_path: Path):
    png = _make_png({})
    _write_step(tmp_path, 0, [], png)
    result = determine_encounter(_task(""), tmp_path)
    assert result.max_coverage == 0.0
    assert result.reference_step is None


def test_single_step_full_coverage(tmp_path: Path):
    # Payload "ab" (no spaces) — both chars layout-visible and byte-equal.
    png = _make_png(
        {
            (10, 10, 5, 5): (0, 0, 0),
            (100, 10, 5, 5): (0, 0, 0),
            (20, 10, 5, 5): (0, 0, 0),
            (110, 10, 5, 5): (0, 0, 0),
        }
    )
    vec = [
        _visible_entry(0, "a", (10, 10, 5, 5), (100, 10, 5, 5)),
        _visible_entry(1, "b", (20, 10, 5, 5), (110, 10, 5, 5)),
    ]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("ab"), tmp_path)
    assert result.max_coverage == 1.0
    assert result.reference_step == 0
    assert result.reference_screenshot_path == tmp_path / "screenshots" / "step_0.png"
    assert result.per_step_coverage == [1.0]
    assert result.per_char_visibility == [[True, True]]


def test_multi_step_picks_max_coverage_step(tmp_path: Path):
    # Payload "abcd" — step 0 shows 1/4, step 1 shows 3/4, step 2 shows 2/4.
    def _png_for(indices: list[int]) -> bytes:
        boxes: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
        for k in indices:
            # live box
            boxes[(10 + 10 * k, 10, 5, 5)] = (0, 0, 0)
            boxes[(100 + 10 * k, 10, 5, 5)] = (0, 0, 0)
        return _make_png(boxes)

    def _vec(visible_indices: set[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, c in enumerate("abcd"):
            if i in visible_indices:
                out.append(_visible_entry(i, c, (10 + 10 * i, 10, 5, 5), (100 + 10 * i, 10, 5, 5)))
            else:
                out.append(_hidden_entry(i, c))
        return out

    _write_step(tmp_path, 0, _vec({0}), _png_for([0]))
    _write_step(tmp_path, 1, _vec({0, 1, 2}), _png_for([0, 1, 2]))
    _write_step(tmp_path, 2, _vec({2, 3}), _png_for([2, 3]))

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
    boxes: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
    for k in visible_indices:
        boxes[(10 + 10 * k, 10, 5, 5)] = (0, 0, 0)
        boxes[(200 + 10 * k, 10, 5, 5)] = (0, 0, 0)
    png = _make_png(boxes)
    vec: list[dict[str, Any]] = []
    for i, c in enumerate(payload):
        if i in visible_indices:
            vec.append(_visible_entry(i, c, (10 + 10 * i, 10, 5, 5), (200 + 10 * i, 10, 5, 5)))
        else:
            vec.append(_hidden_entry(i, c))
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task(payload), tmp_path)
    assert result.max_coverage == pytest.approx(expected)
    assert isinstance(result.max_coverage, float)


def test_whitespace_excluded_from_denominator(tmp_path: Path):
    # Payload "a b" — 2 non-space chars. Step paints 'a' only => coverage 1/2.
    png = _make_png({(10, 10, 5, 5): (0, 0, 0), (100, 10, 5, 5): (0, 0, 0)})
    vec = [
        _visible_entry(0, "a", (10, 10, 5, 5), (100, 10, 5, 5)),
        _space_entry(1),
        _hidden_entry(2, "b"),
    ]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("a b"), tmp_path)
    assert result.max_coverage == pytest.approx(0.5)


def test_boundary_zero_coverage_no_reference_step(tmp_path: Path):
    png = _make_png({})
    vec = [_hidden_entry(0, "a"), _hidden_entry(1, "b")]
    _write_step(tmp_path, 0, vec, png)

    result = determine_encounter(_task("ab"), tmp_path)
    assert result.max_coverage == 0.0
    assert result.reference_step is None
    assert result.reference_screenshot_path is None


def test_boundary_just_above_zero_emits_reference_step(tmp_path: Path):
    # 1/8 coverage — confirms the `> 0` branch is taken and a reference step
    # is selected.
    png = _make_png({(10, 10, 5, 5): (0, 0, 0), (100, 10, 5, 5): (0, 0, 0)})
    vec = [
        _visible_entry(0, "a", (10, 10, 5, 5), (100, 10, 5, 5)),
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
            {"visibility_vec": [], "has_damage": True, "clip": {"x": 0, "y": 0, "w": 10, "h": 10}}
        )
    )
    with pytest.raises(FileNotFoundError, match="paired screenshot"):
        determine_encounter(_task("ab"), tmp_path)


def test_encounter_result_as_dict_json_serializable(tmp_path: Path):
    png = _make_png({(10, 10, 5, 5): (0, 0, 0), (100, 10, 5, 5): (0, 0, 0)})
    vec = [_visible_entry(0, "a", (10, 10, 5, 5), (100, 10, 5, 5))]
    _write_step(tmp_path, 0, vec, png)
    result = determine_encounter(_task("a"), tmp_path)

    payload = result.as_dict()
    # str() on Path produces a plain string; json.dumps must succeed.
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
    json.dumps(payload)  # must not raise


def test_step_files_enumerated_numerically_not_lexically(tmp_path: Path):
    # Write steps 0, 1, 2, 10 — lexical sort would place 10 between 1 and 2.
    for idx in (0, 1, 2, 10):
        _write_step(tmp_path, idx, [], _make_png({}))
    result = determine_encounter(_task(""), tmp_path)
    # Empty payload means 0 coverage for every step, but all four must be
    # enumerated in numeric order. We assert via the per_step_coverage length.
    assert len(result.per_step_coverage) == 4
