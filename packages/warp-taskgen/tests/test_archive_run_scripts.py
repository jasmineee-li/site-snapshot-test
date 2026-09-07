from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _fake_aws(bin_dir: Path, log_path: Path) -> None:
    _write_executable(
        bin_dir / "aws",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

log_path = Path(os.environ["FAKE_AWS_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    json.dump(sys.argv[1:], handle)
    handle.write("\\n")

if sys.argv[1:3] in (["s3", "ls"], ["s3", "sync"]):
    if sys.argv[1:3] == ["s3", "sync"]:
        source = Path(sys.argv[3])
        files = sorted(
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file()
        )
        Path(os.environ["FAKE_AWS_SYNC_FILES"]).write_text(
            json.dumps(files), encoding="utf-8"
        )
    raise SystemExit(0)
raise SystemExit(f"unexpected fake aws call: {sys.argv[1:]}")
""",
    )


def _run_archive(
    tmp_path: Path,
    *,
    logs_dir: Path,
    run_id: str,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], list[str], Path]:
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_log = tmp_path / "aws.jsonl"
    sync_files = tmp_path / "sync-files.json"
    _fake_aws(fake_bin, aws_log)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["FAKE_AWS_LOG"] = str(aws_log)
    env["FAKE_AWS_SYNC_FILES"] = str(sync_files)
    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "archive_run_to_s3.sh"),
            run_id,
            "--logs-dir",
            str(logs_dir),
            "--bucket",
            "test-archives",
            "--prefix",
            "worldsim-runs",
            "--region",
            "us-east-2",
            "--dryrun",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in aws_log.read_text().splitlines()]
    synced = json.loads(sync_files.read_text(encoding="utf-8")) if sync_files.exists() else []
    return completed, calls, synced, logs_dir / run_id


def test_archive_legacy_root_keeps_canonical_s3_layout(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    run_id = "legacy-run-20260508"
    run_dir = logs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "pipeline_state.json").write_text(
        json.dumps({"step": "phase_3", "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "phase_2" / "shards").mkdir(parents=True)
    (run_dir / "phase_2" / "shards" / "shard-000.json").write_text(
        json.dumps({"status": "complete"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "phase_2" / "text_fill" / "checkpoints").mkdir(parents=True)
    (run_dir / "phase_2" / "text_fill" / "checkpoints" / "task-000.json").write_text(
        json.dumps({"status": "complete"}) + "\n", encoding="utf-8"
    )
    (run_dir / "phase_2" / "feasibility_checkpoints").mkdir(parents=True)
    (run_dir / "phase_2" / "feasibility_checkpoints" / "task-000.json").write_text(
        json.dumps({"status": "verified"}) + "\n", encoding="utf-8"
    )

    completed, calls, synced_files, archived_dir = _run_archive(
        tmp_path, logs_dir=logs_dir, run_id=run_id
    )

    assert completed.returncode == 0, completed.stderr
    assert archived_dir == run_dir
    manifest = json.loads((run_dir / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["destination"] == f"s3://test-archives/worldsim-runs/{run_id}/"
    sync_calls = [call for call in calls if call[:2] == ["s3", "sync"]]
    assert len(sync_calls) == 1
    assert sync_calls[0][2] == f"{run_dir}/"
    assert sync_calls[0][3] == f"s3://test-archives/worldsim-runs/{run_id}/"
    assert "--dryrun" in sync_calls[0]
    assert manifest["file_count_local"] == 4
    assert {
        "pipeline_state.json",
        "phase_2/shards/shard-000.json",
        "phase_2/text_fill/checkpoints/task-000.json",
        "phase_2/feasibility_checkpoints/task-000.json",
    } <= set(synced_files)
    archived_state = json.loads((run_dir / "pipeline_state.json").read_text(encoding="utf-8"))
    assert "run_id" not in archived_state
    assert "definition_digest" not in archived_state


def test_archive_identified_root_includes_phase2_checkpoint_families(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    run_id = "run-identified-20260811"
    run_dir = logs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "paused",
                "run_id": run_id,
                "definition_digest": "d" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_paths = {
        "phase_2/shards/shard-000.json": {"status": "complete", "run_id": run_id},
        "phase_2/text_fill/checkpoints/task-000.json": {
            "status": "complete",
            "run_id": run_id,
        },
        "phase_2/feasibility_checkpoints/task-000.json": {
            "status": "verified",
            "run_id": run_id,
        },
        "phase_4/task-000/strategy_variation_checkpoint.json": {
            "status": "complete",
            "run_id": run_id,
        },
    }
    for relative_path, payload in checkpoint_paths.items():
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    completed, _calls, synced_files, _archived_dir = _run_archive(
        tmp_path, logs_dir=logs_dir, run_id=run_id
    )

    assert completed.returncode == 0, completed.stderr
    assert set(checkpoint_paths) <= set(synced_files)


def test_archive_derived_child_uses_child_id_without_recursive_collection(tmp_path: Path) -> None:
    collection = tmp_path / ".warp-derived-runs" / "request-key"
    child_id = "run-child-abc123"
    child_dir = collection / child_id
    child_dir.mkdir(parents=True)
    (child_dir / "derived_run.json").write_text(
        json.dumps({"run_id": child_id, "source_run_id": "run-parent"}) + "\n",
        encoding="utf-8",
    )
    (collection / "sibling-not-a-child.txt").write_text("not archived\n", encoding="utf-8")

    completed, calls, _synced_files, archived_dir = _run_archive(
        tmp_path,
        logs_dir=collection,
        run_id=child_id,
    )

    assert completed.returncode == 0, completed.stderr
    assert archived_dir == child_dir
    manifest = json.loads((child_dir / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["destination"] == f"s3://test-archives/worldsim-runs/{child_id}/"
    sync_call = next(call for call in calls if call[:2] == ["s3", "sync"])
    assert sync_call[2] == f"{child_dir}/"
    assert "sibling-not-a-child.txt" not in manifest["source_path"]


def test_archive_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_log = tmp_path / "aws.jsonl"
    _fake_aws(fake_bin, aws_log)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["FAKE_AWS_LOG"] = str(aws_log)
    env["FAKE_AWS_SYNC_FILES"] = str(tmp_path / "sync-files.json")

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "archive_run_to_s3.sh"),
            "../escaped",
            "--logs-dir",
            str(logs_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "opaque directory name" in completed.stderr
    assert not (escaped / "ARCHIVE_MANIFEST.json").exists()


def test_archive_help_describes_selected_artifact_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "archive_run_to_s3.sh"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "one selected Run Artifact root" in completed.stdout
    assert "Derived Run collection" in completed.stdout


def test_archive_includes_default_phase4_exact_rewrite_requests(tmp_path: Path) -> None:
    from warp_taskgen.phase_4.eval_awareness_request_archive import RewriteRequestArchive

    logs_dir = tmp_path / "logs"
    run_id = "run-phase4-retention"
    task_root = logs_dir / run_id / "phase_4" / "20260907_120000"
    archive = RewriteRequestArchive(task_root, "task-1", "task-1", 1)
    archive.record_request(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "exact"}]},
        semantic_attempt=1,
    )
    archive.record_output({})
    relative_directory = archive.directory.relative_to(logs_dir / run_id)
    completed, calls, synced_files, _ = _run_archive(tmp_path, logs_dir=logs_dir, run_id=run_id)
    assert completed.returncode == 0, completed.stderr
    assert {
        (relative_directory / "1.json").as_posix(),
        (relative_directory / "diagnostics.json").as_posix(),
    } <= set(synced_files)
    sync = next(call for call in calls if call[:2] == ["s3", "sync"])
    assert "--exclude" not in sync
