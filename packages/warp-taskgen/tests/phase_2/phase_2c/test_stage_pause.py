from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from warp_taskgen.phase_2.phase_2c import stage as phase_2_stage
from warp_taskgen.phase_2.phase_2c.types import FeasibilityReport
from warp_taskgen.run_control import PauseBoundaryReached, request_pause


def _task() -> dict[str, object]:
    return {
        "id": "pause-stage-task",
        "site": "shopping",
        "benchmark": "webarena_verified",
    }


def _args(instances_path: Path) -> Namespace:
    return Namespace(
        skip_feasibility=False,
        feasibility_only=True,
        feasibility_instances=str(instances_path),
        feasibility_concurrency=1,
        feasibility_retry_count=0,
        feasibility_ttl_hours=None,
        force_reverify=False,
    )


@pytest.mark.asyncio
async def test_pause_before_aggregate_promotion_keeps_canonical_files_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    original = json.dumps([_task()], indent=2)
    output_path.write_text(original, encoding="utf-8")
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "instances": [{"site_name": "shopping", "site_url": "http://shopping.test"}],
            }
        ),
        encoding="utf-8",
    )

    async def fake_verify(*_args: object, **_kwargs: object):
        # The request wins before the feature-owned promotion boundary. The
        # stage must not write even a partial sidecar or canonical aggregate.
        request_pause(tmp_path)
        return FeasibilityReport(
            verified=[{**_task(), "feasibility": {"status": "verified"}}],
            infeasible=[],
            skipped_already_verified=[],
            cleanup_warnings=[],
            host_fingerprint={},
            elapsed_seconds=0.0,
            per_site_counts={},
            phase_2_status="complete",
        )

    monkeypatch.setattr(phase_2_stage, "verify_feasibility", fake_verify)

    with pytest.raises(PauseBoundaryReached):
        await phase_2_stage._run_feasibility_stage(
            args=_args(instances_path),
            output_path=output_path,
            output_dir=output_dir,
            state_metadata={"feasibility_only": True},
            prior_phase_2_status="complete",
        )

    assert output_path.read_text(encoding="utf-8") == original
    assert not (output_dir / "adversarial_tasks.infeasible.json").exists()
    assert not (output_dir / "adversarial_tasks.dropped_source_data.json").exists()
    assert not (output_dir / "feasibility_report.json").exists()
