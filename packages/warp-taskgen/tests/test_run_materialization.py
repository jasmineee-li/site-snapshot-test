from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from worldsim import run_materialization
from worldsim.cli import _impl as cli_impl
from worldsim.run_materialization import materialize_derived_run
from worldsim.run_transition import resolve_run_request
from worldsim.state import save_state


def _identified_source(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    transition = resolve_run_request(
        {
            "benchmark_path": str(tmp_path / "benchmark"),
            "agent_model": "source-model",
            "sites": ["gitlab"],
        },
        existing_state=None,
        new_run_id="run-source",
    )
    assert transition.definition is not None
    state: dict[str, object] = {
        "step": "phase_4",
        "iteration": 0,
        "timestamp": "2026-08-11T12:00:00+00:00",
        "logs_dir": str(source_root),
        "status": "running",
        "agent_model": "source-model",
        "run_definition": transition.definition.to_dict(),
    }
    (source_root / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    return source_root, state


def _drift(state: dict[str, object]):
    transition = resolve_run_request({"agent_model": "child-model"}, existing_state=state)
    assert transition.kind == "derived_required"
    return transition


def test_materialization_is_idempotent_isolated_and_preserves_parent(tmp_path: Path):
    source_root, state = _identified_source(tmp_path)
    source_before = (source_root / "pipeline_state.json").read_bytes()
    pointer = tmp_path / "logs" / "last_run_state.json"
    pointer.parent.mkdir()
    pointer.write_text('{"sentinel":"parent"}', encoding="utf-8")
    pointer_before = pointer.read_bytes()
    transition = _drift(state)

    first = materialize_derived_run(
        source_root,
        transition,
        collection_root=tmp_path / "children",
    )
    second = materialize_derived_run(
        source_root,
        transition,
        collection_root=tmp_path / "children",
    )

    assert first.created is True
    assert second.created is False
    assert second.child_root == first.child_root
    assert second.definition == first.definition
    assert first.definition.run_id != "run-source"
    assert first.definition.source_run_id == "run-source"
    assert first.definition.definition_digest == transition.definition.definition_digest
    assert (source_root / "pipeline_state.json").read_bytes() == source_before
    assert pointer.read_bytes() == pointer_before
    assert not first.child_root.is_relative_to(source_root)

    child_state = json.loads((first.child_root / "pipeline_state.json").read_text())
    assert child_state["step"] == "phase_0a"
    assert child_state["status"] == "failed"
    assert child_state["reason"] == "derived_run_materialized"
    assert child_state["run_definition"] == first.definition.to_dict()
    assert not any(first.child_root.glob("phase_*"))


def test_materialization_recovers_reserved_identity_after_missing_child_state(tmp_path: Path):
    source_root, state = _identified_source(tmp_path)
    transition = _drift(state)
    first = materialize_derived_run(
        source_root,
        transition,
        collection_root=tmp_path / "children",
    )
    (first.child_root / "pipeline_state.json").unlink()

    recovered = materialize_derived_run(
        source_root,
        transition,
        collection_root=tmp_path / "children",
    )

    assert recovered.definition.run_id == first.definition.run_id
    assert recovered.created is True
    assert (recovered.child_root / "pipeline_state.json").is_file()


def test_materialization_fails_closed_on_source_or_reservation_drift(tmp_path: Path):
    source_root, state = _identified_source(tmp_path)
    transition = _drift(state)
    child = materialize_derived_run(
        source_root,
        transition,
        collection_root=tmp_path / "children",
    )
    reservation = json.loads(child.reservation_path.read_text())
    reservation["source_run_id"] = "run-other"
    child.reservation_path.write_text(json.dumps(reservation), encoding="utf-8")

    with pytest.raises(ValueError, match="source_run_id"):
        materialize_derived_run(
            source_root,
            transition,
            collection_root=tmp_path / "children",
        )

    child.reservation_path.write_text(
        json.dumps({**reservation, "source_run_id": "run-source"}),
        encoding="utf-8",
    )
    source_state = json.loads((source_root / "pipeline_state.json").read_text())
    source_state["timestamp"] = "2026-08-11T12:00:01+00:00"
    (source_root / "pipeline_state.json").write_text(json.dumps(source_state), encoding="utf-8")
    with pytest.raises(ValueError, match="source_state_sha256"):
        materialize_derived_run(
            source_root,
            transition,
            collection_root=tmp_path / "children",
        )


def test_materialization_rejects_path_identity_drift_and_nonfresh_child(tmp_path: Path):
    source_root, state = _identified_source(tmp_path)
    transition = _drift(state)
    child = materialize_derived_run(
        source_root,
        transition,
        collection_root=tmp_path / "children",
    )
    reservation = json.loads(child.reservation_path.read_text())
    reservation["child_root"] = str(child.child_root.parent / "run-other")
    child.reservation_path.write_text(json.dumps(reservation), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its Run ID"):
        materialize_derived_run(
            source_root,
            transition,
            collection_root=tmp_path / "children",
        )

    reservation["child_root"] = str(child.child_root)
    child.reservation_path.write_text(json.dumps(reservation), encoding="utf-8")
    (child.child_root / "pipeline_state.json").unlink()
    (child.child_root / "derived_run.json").unlink()
    (child.child_root / "unrelated.txt").write_text("not this run", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated files"):
        materialize_derived_run(
            source_root,
            transition,
            collection_root=tmp_path / "children",
        )


def test_materialization_recovers_after_uncommitted_reservation_write(monkeypatch, tmp_path: Path):
    source_root, state = _identified_source(tmp_path)
    transition = _drift(state)
    real_write = run_materialization.write_json_atomic

    def crash_before_reservation(path, payload, *, failpoint_base=None):
        if failpoint_base == "run_materialization.reservation":
            raise RuntimeError("synthetic crash before reservation commit")
        return real_write(path, payload, failpoint_base=failpoint_base)

    monkeypatch.setattr(run_materialization, "write_json_atomic", crash_before_reservation)
    with pytest.raises(RuntimeError, match="before reservation commit"):
        materialize_derived_run(
            source_root,
            transition,
            collection_root=tmp_path / "children",
        )

    monkeypatch.setattr(run_materialization, "write_json_atomic", real_write)
    recovered = materialize_derived_run(
        source_root,
        transition,
        collection_root=tmp_path / "children",
    )
    assert recovered.created is True
    assert recovered.reservation_path.is_file()


def test_materialization_rejects_legacy_missing_benchmark_and_nested_collection(tmp_path: Path):
    source_root, state = _identified_source(tmp_path)
    with pytest.raises(ValueError, match="derived_required"):
        materialize_derived_run(source_root, resolve_run_request({}, existing_state=state))

    no_benchmark = resolve_run_request(
        {"agent_model": "one"},
        existing_state=None,
        new_run_id="run-no-benchmark",
    )
    assert no_benchmark.definition is not None
    no_benchmark_state = {
        **state,
        "run_definition": no_benchmark.definition.to_dict(),
        "agent_model": "one",
    }
    (source_root / "pipeline_state.json").write_text(
        json.dumps(no_benchmark_state), encoding="utf-8"
    )
    drift = resolve_run_request({"agent_model": "two"}, existing_state=no_benchmark_state)
    with pytest.raises(ValueError, match="benchmark_path"):
        materialize_derived_run(source_root, drift)

    source_root, state = _identified_source(tmp_path / "other")
    with pytest.raises(ValueError, match="outside the source root"):
        materialize_derived_run(
            source_root,
            _drift(state),
            collection_root=source_root / "children",
        )


def test_resume_materializes_child_then_child_resume_restores_definition_inputs(
    monkeypatch, tmp_path, capsys
):
    source_root, _state = _identified_source(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(source_root))
    source_before = (source_root / "pipeline_state.json").read_bytes()
    pointer = tmp_path / "logs" / "last_run_state.json"
    pointer.parent.mkdir()
    pointer.write_text('{"sentinel":"parent"}', encoding="utf-8")
    pointer_before = pointer.read_bytes()
    called = False

    def unexpected_dispatch(_args):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cli_impl, "_dispatch_phase_with_run_context", unexpected_dispatch)
    assert cli_impl._dispatch_resume(Namespace(agent_model="child-model")) == 0
    assert called is False
    assert (source_root / "pipeline_state.json").read_bytes() == source_before
    output = capsys.readouterr().out
    assert "Created isolated Derived Run" in output
    child_root = next((tmp_path / ".warp-derived-runs").glob("*/run-*"))
    child_pointer = child_root / "last_run_state.json"
    assert f"WARP_TASKGEN_RESUME_POINTER={child_pointer}" in output

    captured = {}

    def child_dispatch(args):
        captured["phase"] = args.phase
        captured["benchmark"] = args.benchmark
        captured["agent_model"] = args.agent_model
        save_state("phase_0a", status="running")
        return 0

    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(child_root))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(child_pointer))
    monkeypatch.setattr(cli_impl, "_install_verification_proxy_from_args", lambda _args: None)
    monkeypatch.setattr(cli_impl, "_dispatch_phase_with_run_context", child_dispatch)
    assert cli_impl._dispatch_resume(Namespace()) == 0
    assert captured == {
        "phase": "0a",
        "benchmark": tmp_path / "benchmark",
        "agent_model": "child-model",
    }
    assert pointer.read_bytes() == pointer_before
    assert child_pointer.is_file()
    child_state = json.loads((child_root / "pipeline_state.json").read_text())
    assert child_state["run_definition"]["source_run_id"] == "run-source"


def test_terminal_complete_resume_materializes_explicit_definition_drift(
    monkeypatch, tmp_path: Path
):
    source_root, state = _identified_source(tmp_path)
    state["status"] = "complete"
    (source_root / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(source_root))

    assert cli_impl._dispatch_resume(Namespace(agent_model="child-model")) == 0

    child_root = next((tmp_path / ".warp-derived-runs").glob("*/run-*"))
    child_state = json.loads((child_root / "pipeline_state.json").read_text())
    assert child_state["run_definition"]["source_run_id"] == "run-source"


def test_resume_reports_materialization_filesystem_failure(monkeypatch, tmp_path, capsys):
    source_root, _state = _identified_source(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(source_root))

    def fail_materialization(*_args, **_kwargs):
        raise OSError("read-only derived collection")

    monkeypatch.setattr(run_materialization, "materialize_derived_run", fail_materialization)
    assert cli_impl._dispatch_resume(Namespace(agent_model="child-model")) == 2
    assert "read-only derived collection" in capsys.readouterr().err
