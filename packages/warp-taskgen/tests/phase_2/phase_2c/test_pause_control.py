from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from worldsim.phase_2.phase_2c.pause_control import (
    assert_preflight_boundary,
    promotion_boundary,
    run_verification_units,
)
from worldsim.run_control import PauseBoundaryReached, request_pause
from worldsim.state import save_state


def _write_feasibility_state(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "running",
                "phase_2_stage": "feasibility",
                "timestamp": "2026-08-11T00:00:00+00:00",
                "logs_dir": str(root),
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_verification_scheduler_drains_claimed_units_and_stops_later_claims(
    tmp_path: Path,
) -> None:
    _write_feasibility_state(tmp_path)
    started: list[int] = []
    finished: list[int] = []
    both_claimed = asyncio.Event()

    async def operation(item: int) -> int:
        started.append(item)
        if len(started) == 2:
            both_claimed.set()
        if item == 0:
            await both_claimed.wait()
            request_pause(tmp_path)
        else:
            await asyncio.sleep(0)
        finished.append(item)
        return item

    with pytest.raises(PauseBoundaryReached):
        await run_verification_units([0, 1, 2], operation, concurrency=2, state_dir=tmp_path)

    assert sorted(started) == [0, 1]
    assert sorted(finished) == [0, 1]


def test_preflight_pause_stops_before_first_verification_claim(tmp_path: Path) -> None:
    _write_feasibility_state(tmp_path)
    request_pause(tmp_path)

    with pytest.raises(PauseBoundaryReached):
        assert_preflight_boundary(tmp_path)


@pytest.mark.asyncio
async def test_legacy_scheduler_does_not_poll_default_lifecycle_root(tmp_path: Path) -> None:
    results = await run_verification_units(
        [1, 2], lambda item: asyncio.sleep(0, result=item), concurrency=1
    )

    assert results == [1, 2]


@pytest.mark.asyncio
async def test_custom_operation_boundary_stops_sibling_claims() -> None:
    started: list[int] = []

    async def operation(item: int) -> int:
        started.append(item)
        if item == 0:
            raise PauseBoundaryReached()
        return item

    with pytest.raises(PauseBoundaryReached):
        await run_verification_units([0, 1], operation, concurrency=2)

    assert started == [0]


@pytest.mark.asyncio
async def test_verification_scheduler_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        await run_verification_units([], lambda item: asyncio.sleep(0, result=item), concurrency=0)


def test_promotion_request_wins_before_boundary_without_entering_writer(tmp_path: Path) -> None:
    _write_feasibility_state(tmp_path)
    request_pause(tmp_path)
    entered = False

    with pytest.raises(PauseBoundaryReached):
        with promotion_boundary(tmp_path):
            entered = True

    assert entered is False


def test_threaded_request_wins_before_promotion_without_entering_writer(tmp_path: Path) -> None:
    _write_feasibility_state(tmp_path)
    request_finished = threading.Event()
    request_error: list[BaseException] = []

    def request_before_promotion() -> None:
        try:
            request_pause(tmp_path)
        except BaseException as exc:
            request_error.append(exc)
        finally:
            request_finished.set()

    requester = threading.Thread(target=request_before_promotion)
    requester.start()
    assert request_finished.wait(timeout=2)
    requester.join(timeout=2)
    assert not request_error

    entered = False
    with pytest.raises(PauseBoundaryReached):
        with promotion_boundary(tmp_path):
            entered = True

    assert entered is False
    assert (tmp_path / "pause_request.json").exists()


def test_promotion_wins_before_threaded_request_and_completes_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_feasibility_state(tmp_path)
    request_finished = threading.Event()
    request_error: list[BaseException] = []

    def request_after_promotion_begins() -> None:
        try:
            request_pause(tmp_path)
        except BaseException as exc:  # the terminal race is expected to reject it
            request_error.append(exc)
        finally:
            request_finished.set()

    with promotion_boundary(tmp_path):
        writer = threading.Thread(target=request_after_promotion_begins)
        writer.start()
        save_state(
            "phase_2",
            status="complete",
            phase_2_stage="feasibility",
            checkpoint="aggregate-promoted",
            _pause_lock_held=True,
        )
    writer.join(timeout=2)

    assert request_finished.is_set()
    assert request_error and isinstance(request_error[0], ValueError)
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["checkpoint"] == "aggregate-promoted"
    assert not (tmp_path / "pause_request.json").exists()
