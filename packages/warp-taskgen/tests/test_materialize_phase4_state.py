from pathlib import Path

import pytest

from scripts.materialize_phase4_state import materialize_phase4_state


def _write(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_materialize_phase4_state_copies_phase_inputs_without_phase4(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _write(source / "phase_0c" / "BENCHMARK_PROFILE_reddit.json", "{}")
    _write(source / "phase_1" / "benign_tasks.json", "[]")
    _write(source / "phase_2" / "adversarial_tasks.json", "[]")
    _write(source / "phase_3" / "contracts.json", "[]")
    _write(source / "phase_4" / "results.json", "[]")
    _write(source / "pipeline_state.json", '{"step":"phase_4","status":"complete"}')
    _write(source / "cost_report.json", "[]")

    _write(source / "phase_2" / "feasibility.lock", "")
    materialize_phase4_state(source, dest)

    assert (dest / "phase_0c" / "BENCHMARK_PROFILE_reddit.json").exists()
    assert (dest / "phase_1" / "benign_tasks.json").exists()
    assert (dest / "phase_2" / "adversarial_tasks.json").exists()
    assert (dest / "phase_3" / "contracts.json").exists()
    assert not (dest / "phase_4").exists()
    assert not (dest / "pipeline_state.json").exists()
    assert not (dest / "cost_report.json").exists()
    assert not (dest / "phase_2" / "feasibility.lock").exists()


def test_materialize_phase4_state_fails_if_destination_exists(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _write(source / "phase_2" / "adversarial_tasks.json", "[]")
    _write(source / "phase_3" / "contracts.json", "[]")
    dest.mkdir()

    with pytest.raises(SystemExit, match="destination already exists"):
        materialize_phase4_state(source, dest)
