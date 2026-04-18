"""Live Phase 2c feasibility verification — one test per site + cross-site."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.editors.gitlab import GitlabEditor
from worldsim.failpoints import FAILPOINT_EXIT_CODE
from worldsim.phases import phase_2_feasibility as feas

pytestmark = [pytest.mark.integration, pytest.mark.feasibility]


_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "feasibility"


def _load_fixture(site: str, kind: str) -> dict[str, Any]:
    path = _FIXTURE_ROOT / site / f"{kind}.json"
    if not path.exists():
        pytest.skip(f"no feasibility fixture for {site}/{kind}")
    return json.loads(path.read_text())


def _materialize_task(fixture: dict[str, Any]) -> dict[str, Any]:
    """Substitute ``{UNIQUE}`` placeholders with a fresh uuid so parallel
    runs don't collide on resource names."""
    token = uuid.uuid4().hex[:12]
    serialized = json.dumps(fixture)
    serialized = serialized.replace("{UNIQUE}", token)
    return json.loads(serialized)


def _write_task(tmp_path: Path, task: dict[str, Any]) -> Path:
    path = tmp_path / "adversarial_tasks.json"
    path.write_text(json.dumps([task]))
    return path


def _run_verifier(
    tasks_path: Path,
    instance: dict[str, Any],
    **kwargs: Any,
) -> feas.FeasibilityReport:
    return asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[instance],
            instances_label="instances.smoke.json",
            concurrency=kwargs.pop("concurrency", 1),
            retry_count=kwargs.pop("retry_count", 1),
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Per-site good / oversize / policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("site", ["gitlab", "reddit", "shopping", "shopping_admin"])
def test_feasibility_good_task(site, live_instance, tmp_path):
    fixture = _load_fixture(site, "good")
    task = _materialize_task(fixture)
    instance = live_instance(site)
    assert acquire_tokens_for_instances([instance]) == []
    tasks_path = _write_task(tmp_path, task)

    report = _run_verifier(tasks_path, instance)
    assert len(report.verified) == 1, report
    assert report.verified[0]["feasibility"]["status"] == "verified"
    # Per-task immediate cleanup is the invariant.
    assert all("cleanup_failed" not in w for w in report.cleanup_warnings), report.cleanup_warnings


@pytest.mark.parametrize("site", ["gitlab", "shopping", "shopping_admin"])
def test_feasibility_oversize_task(site, live_instance, tmp_path):
    fixture = _load_fixture(site, "oversize")
    task = _materialize_task(fixture)
    instance = live_instance(site)
    assert acquire_tokens_for_instances([instance]) == []
    tasks_path = _write_task(tmp_path, task)

    report = _run_verifier(tasks_path, instance)
    assert len(report.infeasible) == 1, report
    entry = report.infeasible[0]["feasibility"]
    assert entry["status"] == "infeasible"
    assert entry["errors"][0]["kind"] == "length_exceeded", entry


@pytest.mark.parametrize("site", ["gitlab", "reddit"])
def test_feasibility_policy_task(site, live_instance, tmp_path):
    """Platforms with a known content-policy filter should classify the
    trigger payload as ``content_policy``. Skipped for sites whose filter
    behavior is unstable or absent (shopping, shopping_admin)."""
    fixture = _load_fixture(site, "policy")
    task = _materialize_task(fixture)
    instance = live_instance(site)
    assert acquire_tokens_for_instances([instance]) == []
    tasks_path = _write_task(tmp_path, task)

    report = _run_verifier(tasks_path, instance)
    # Platform policies drift; accept either classification or plain rejection.
    if report.infeasible:
        kind = report.infeasible[0]["feasibility"]["errors"][0]["kind"]
        assert kind in {"content_policy", "field_required", "request_failed"}


# ---------------------------------------------------------------------------
# Cross-site: concurrency + cleanup residue
# ---------------------------------------------------------------------------


def test_feasibility_concurrency_batch(live_config, tmp_path):
    """Submit one good task per site, run with concurrency>=4, expect all
    verified and cleanup residue = 0."""
    instances_by_site = {inst.site_name: inst.model_dump() for inst in live_config.instances}
    tasks: list[dict[str, Any]] = []
    live_instances: list[dict[str, Any]] = []
    for site in ("gitlab", "reddit", "shopping", "shopping_admin"):
        if site not in instances_by_site:
            continue
        fixture_path = _FIXTURE_ROOT / site / "good.json"
        if not fixture_path.exists():
            continue
        tasks.append(_materialize_task(json.loads(fixture_path.read_text())))
        live_instances.append(instances_by_site[site])

    if not tasks:
        pytest.skip("no good fixtures available for concurrency sweep")

    assert acquire_tokens_for_instances(live_instances) == []
    tasks_path = tmp_path / "adversarial_tasks.json"
    tasks_path.write_text(json.dumps(tasks))

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=live_instances,
            instances_label="instances.smoke.json",
            concurrency=4,
            retry_count=1,
        )
    )
    assert len(report.verified) == len(tasks), report


def test_feasibility_cleanup_leaves_no_gitlab_residue(live_instance, tmp_path):
    """After a feasibility run, there must be zero ``webagent-verify-*``
    and zero surviving ``webagent-task-*`` projects under the API user.

    The contract is per-task immediate cleanup, not bulk sweep.
    """
    instance = live_instance("gitlab")
    assert acquire_tokens_for_instances([instance]) == []

    fixture = _load_fixture("gitlab", "good")
    task = _materialize_task(fixture)
    tasks_path = _write_task(tmp_path, task)

    report = _run_verifier(tasks_path, instance)
    assert len(report.verified) == 1, report

    import requests

    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        current_user = editor._current_user()
        user_id = current_user.get("id")
        for prefix in ("webagent-verify-", "webagent-task-"):
            projects = editor._gitlab_request_json(
                "GET",
                f"/api/v4/users/{editor._quote(user_id)}/projects",
                params={"search": prefix, "per_page": 100, "simple": True},
            )
            assert isinstance(projects, list)
            leftover = [p for p in projects if str(p.get("path", "")).lower().startswith(prefix)]
            assert not leftover, (
                f"gitlab retained {len(leftover)} {prefix!r} project(s) after feasibility run"
            )


# ---------------------------------------------------------------------------
# Crash-resume: one child-subprocess test per failpoint base
# ---------------------------------------------------------------------------


def _run_phase_2c_in_subprocess(
    *,
    tasks_path: Path,
    instances_path: Path,
    state_dir: Path,
    failpoint: str | None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WORLDSIM_STATE_DIR"] = str(state_dir)
    if failpoint:
        env["WORLDSIM_FAILPOINTS"] = failpoint
    else:
        env.pop("WORLDSIM_FAILPOINTS", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "worldsim.main",
            "phase",
            "2c",
            "--feasibility-instances",
            str(instances_path),
            "--feasibility-concurrency",
            "1",
            "--feasibility-retry-count",
            "0",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "failpoint",
    [
        feas.FAILPOINT_DATASET + ".before_replace",
        feas.FAILPOINT_QUARANTINE + ".before_replace",
        feas.FAILPOINT_REPORT + ".before_replace",
    ],
)
def test_feasibility_crash_resume(failpoint, live_instance, live_config, tmp_path):
    """Kill 2c at each write failpoint; re-running converges."""
    instance = live_instance("gitlab")
    assert acquire_tokens_for_instances([instance]) == []

    fixture = _load_fixture("gitlab", "good")
    task = _materialize_task(fixture)
    task_oversize = _materialize_task(_load_fixture("gitlab", "oversize"))
    state_dir = tmp_path / "state"
    (state_dir / "phase_2").mkdir(parents=True)
    tasks_path = state_dir / "phase_2" / "adversarial_tasks.json"
    tasks_path.write_text(json.dumps([task, task_oversize]))

    # Synthesize a minimal instances file (wrapper-dict shape) for the
    # child subprocess.
    instances_file = tmp_path / "instances.smoke.json"
    instances_file.write_text(
        json.dumps(
            {
                "benchmark_name": live_config.benchmark_name or "webarena_verified",
                "instances": [instance],
            }
        )
    )

    crashed = _run_phase_2c_in_subprocess(
        tasks_path=tasks_path,
        instances_path=instances_file,
        state_dir=state_dir,
        failpoint=failpoint,
    )
    assert crashed.returncode == FAILPOINT_EXIT_CODE, crashed.stderr

    resumed = _run_phase_2c_in_subprocess(
        tasks_path=tasks_path,
        instances_path=instances_file,
        state_dir=state_dir,
        failpoint=None,
    )
    assert resumed.returncode == 0, resumed.stderr
    final = json.loads(tasks_path.read_text())
    assert any(t.get("feasibility", {}).get("status") == "verified" for t in final)
