from __future__ import annotations

import json
from pathlib import Path

from scripts import write_phase4_artifact_manifest


def test_phase4_artifact_manifest_hashes_inputs(tmp_path: Path) -> None:
    state_dir = tmp_path / "run"
    (state_dir / "phase_0c").mkdir(parents=True)
    (state_dir / "phase_0c" / "BENCHMARK_PROFILE_gitlab.json").write_text("profile")
    (state_dir / "phase_2").mkdir()
    (state_dir / "phase_2" / "adversarial_tasks.json").write_text("tasks")
    (state_dir / "phase_3").mkdir()
    (state_dir / "phase_3" / "contracts.json").write_text("contracts")
    instances = tmp_path / "instances.scale.json"
    instances.write_text('{"instances":[]}')
    output = state_dir / "artifact_manifest.json"

    rc = write_phase4_artifact_manifest.main(
        [
            "--state-dir",
            str(state_dir),
            "--artifacts-source",
            "s3://bucket/run",
            "--instances",
            str(instances),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    manifest = json.loads(output.read_text())
    assert manifest["kind"] == "phase4_artifact_manifest"
    assert manifest["artifacts_source"] == "s3://bucket/run"
    assert manifest["instances"]["sha256"] == write_phase4_artifact_manifest.sha256_file(instances)
    artifact_by_path = {artifact["path"]: artifact for artifact in manifest["artifacts"]}
    assert artifact_by_path["phase_0c"]["file_count"] == 1
    assert artifact_by_path["phase_2"]["file_count"] == 1
    assert artifact_by_path["phase_3"]["file_count"] == 1
    assert artifact_by_path["phase_2"]["files"][0]["path"] == "phase_2/adversarial_tasks.json"
