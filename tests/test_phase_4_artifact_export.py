from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_phase_4_artifacts.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_phase_4_artifacts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_sweep_state_run_dirs_dedupe_and_skip_failed(tmp_path: Path) -> None:
    exporter = _load_export_module()
    state = tmp_path / "sweep_state.json"
    state.write_text(
        json.dumps(
            {
                "completed_runs": [{"run_dir": "logs/gpt"}],
                "records": [
                    {"run_dir": "logs/sonnet", "status": "completed"},
                    {"run_dir": "logs/glm_failed", "status": "failed"},
                    {"run_dir": "logs/sonnet", "status": "completed"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert exporter.collect_run_dirs(["logs/manual"], [state]) == [
        "logs/manual",
        "logs/gpt",
        "logs/sonnet",
    ]
    assert exporter.collect_run_dirs([], [state], include_failed=True) == [
        "logs/gpt",
        "logs/sonnet",
        "logs/glm_failed",
    ]


def test_manifest_includes_compact_files_and_excludes_large_raw_trace_files(tmp_path: Path) -> None:
    exporter = _load_export_module()
    run = tmp_path / "logs" / "phase4_deadlines_model"
    _write(run / "phase_4" / "results.json", "[]")
    _write(run / "phase_4" / "summary.txt", "summary")
    _write(run / "phase_4" / "20260501_000000" / "task_a" / "processed_result.json")
    _write(run / "phase_4" / "20260501_000000" / "task_a" / "pvpo" / "step_1.json")
    _write(
        run / "phase_4" / "20260501_000000" / "task_a" / "network_trace.json",
        json.dumps(
            [
                {
                    "url": "http://example.test/page",
                    "response_status": 200,
                    "resource_type": "document",
                    "is_document_load": True,
                }
            ]
        ),
    )
    _write(run / "phase_4" / "20260501_000000" / "task_a" / "history.json")
    _write(run / "phase_4" / "20260501_000000" / "task_a" / "screenshots" / "step_1.png")
    _write(
        run
        / "phase_4"
        / "20260501_000000"
        / "task_a"
        / "variant_generation"
        / "0_specificity"
        / "initial"
        / "host_validation.json"
    )
    _write(
        run
        / "phase_4"
        / "20260501_000000"
        / "task_a"
        / "variant_generation"
        / "0_specificity"
        / "initial"
        / "redacted_prompt_input.json"
    )
    _write(run / "phase_2" / "adversarial_tasks.json", "[]")
    _write(run / "phase_3" / "contracts.json", "[]")

    manifest = exporter.build_manifest(
        tmp_path,
        ["logs/phase4_deadlines_model"],
        max_file_bytes=5_000_000,
    )

    paths = {row["path"] for row in manifest["files"]}
    assert "logs/phase4_deadlines_model/phase_4/results.json" in paths
    assert (
        "logs/phase4_deadlines_model/phase_4/20260501_000000/task_a/"
        "processed_result.json"
    ) in paths
    assert (
        "logs/phase4_deadlines_model/phase_4/20260501_000000/task_a/pvpo/step_1.json"
    ) in paths
    assert (
        "logs/phase4_deadlines_model/phase_4/20260501_000000/task_a/"
        "variant_generation/0_specificity/initial/host_validation.json"
    ) in paths
    assert all("history.json" not in path for path in paths)
    assert all("network_trace.json" not in path for path in paths)
    assert all("screenshots" not in path for path in paths)
    assert all("redacted_prompt_input" not in path for path in paths)

    with_network = exporter.build_manifest(
        tmp_path,
        ["logs/phase4_deadlines_model"],
        max_file_bytes=5_000_000,
        include_network_traces=True,
    )
    assert any(row["path"].endswith("network_trace.json") for row in with_network["files"])

    filtered_network = exporter.build_manifest(
        tmp_path,
        ["logs/phase4_deadlines_model"],
        max_file_bytes=5_000_000,
        include_network_traces=True,
        network_trace_task_ids={"other_task"},
    )
    assert all("network_trace.json" not in row["path"] for row in filtered_network["files"])

    matching_network = exporter.build_manifest(
        tmp_path,
        ["logs/phase4_deadlines_model"],
        max_file_bytes=5_000_000,
        include_network_traces=True,
        network_trace_task_ids={"task_a"},
    )
    assert any(row["path"].endswith("network_trace.json") for row in matching_network["files"])
    assert matching_network["network_trace_summaries"] == []

    network_summary = exporter.build_manifest(
        tmp_path,
        ["logs/phase4_deadlines_model"],
        max_file_bytes=5_000_000,
        summarize_network_traces=True,
        network_trace_task_ids={"task_a"},
    )
    assert len(network_summary["network_trace_summaries"]) == 1
    assert network_summary["network_trace_summaries"][0]["task_id"] == "task_a"
    assert network_summary["network_trace_summaries"][0]["attacker_invalid_requested"] is False


def test_local_export_extracts_manifest_and_compact_files(tmp_path: Path) -> None:
    exporter = _load_export_module()
    root = tmp_path / "root"
    output = tmp_path / "out"
    _write(root / "logs" / "run" / "phase_4" / "results.json", "[]")
    _write(root / "logs" / "run" / "phase_4" / "20260501" / "task" / "result.json")

    rc = exporter.run_local_export(
        root=root,
        run_dirs=["logs/run"],
        output_dir=output,
        max_file_bytes=5_000_000,
        dry_run=False,
        emit_tar=False,
        overwrite=False,
        include_phase3_contracts=False,
        include_network_traces=False,
        network_trace_task_ids=set(),
        summarize_network_traces=False,
    )

    assert rc == 0
    manifest = json.loads((output / "artifact_export_manifest.json").read_text())
    assert manifest["schema_version"] == "phase4_compact_artifact_export_v1"
    assert (output / "logs" / "run" / "phase_4" / "results.json").exists()
    assert (output / "logs" / "run" / "phase_4" / "20260501" / "task" / "result.json").exists()


def test_remote_dry_run_uses_ssh_without_local_extraction(monkeypatch, tmp_path: Path, capsys) -> None:
    exporter = _load_export_module()
    host_config = tmp_path / "r5.yaml"
    host_config.write_text(
        "\n".join(
            [
                "advertise_host: 203.0.113.10",
                "ssh_user: ubuntu",
                "compose_dir_remote: /home/ubuntu",
                "",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        assert kwargs["input"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=b'{"file_count": 1}\n',
            stderr=b"",
        )

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)

    rc = exporter.run_remote_export(
        host_config_path=host_config,
        remote_dir="/home/ubuntu/browser-sim",
        ssh_key=str(tmp_path / "key.pem"),
        run_dirs=["logs/run"],
        output_dir=None,
        max_file_bytes=5_000_000,
        dry_run=True,
        overwrite=False,
        include_phase3_contracts=False,
        include_network_traces=False,
        network_trace_task_ids=set(),
        summarize_network_traces=False,
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"file_count": 1}
    assert calls[0][:3] == ["ssh", "-i", str(tmp_path / "key.pem")]
    assert "ubuntu@203.0.113.10" in calls[0]
    assert "--root" in calls[0]
    assert "/home/ubuntu/browser-sim" in calls[0]
    assert "--run-dir" in calls[0]
    assert "logs/run" in calls[0]
    assert "--dry-run" in calls[0]


def test_network_trace_task_filter_matches_variant_and_retry_dirs() -> None:
    exporter = _load_export_module()

    assert exporter._network_trace_matches_task(
        Path("phase_4/20260501/adv_1/network_trace.json"), {"adv_1"}
    )
    assert exporter._network_trace_matches_task(
        Path("phase_4/20260501/adv_1_variant_2/network_trace.json"), {"adv_1"}
    )
    assert exporter._network_trace_matches_task(
        Path("phase_4/20260501/adv_1__placement_1/network_trace.json"), {"adv_1"}
    )
    assert not exporter._network_trace_matches_task(
        Path("phase_4/20260501/adv_10/network_trace.json"), {"adv_1"}
    )


def test_unsafe_run_dir_is_rejected(tmp_path: Path) -> None:
    exporter = _load_export_module()
    state = tmp_path / "sweep_state.json"
    state.write_text(json.dumps({"records": [{"run_dir": "../logs/run", "status": "completed"}]}))

    try:
        exporter.collect_run_dirs([], [state])
    except ValueError as exc:
        assert "unsafe path component" in str(exc)
    else:
        raise AssertionError("unsafe run dir was accepted")
