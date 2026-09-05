from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from warp_taskgen.phases.phase_1_generate_new_tasks import (
    EligibleSiteProfile,
    SiteGenerateNewTasksResult,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "run_phase1_provider_microcanaries.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_phase1_provider_microcanaries", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module() -> ModuleType:
    return _load_module()


def _source(module: ModuleType, tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    profile_path = tmp_path / "source" / "phase_0c" / "BENCHMARK_PROFILE_gitlab.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("{}")
    return module.SourceInputs(
        source_run=tmp_path / "source",
        benchmark_root=benchmark,
        manifest={"evaluation": {"eval_types": []}},
        site=EligibleSiteProfile("gitlab", profile_path, {}),
        task_card_plan={"task_cards": []},
    )


def test_load_source_inputs_uses_pipeline_state_task_card_digest(
    module: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    source_run = tmp_path / "source"
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    manifest = source_run / "phase_0a" / "BENCHMARK_MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"evaluation": {"eval_types": []}}))
    profile = source_run / "phase_0c" / "BENCHMARK_PROFILE_gitlab.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}")
    profile.with_name("AGENT_CONTEXT_gitlab.json").write_text("{}")
    task_card_plan_path = source_run / "inputs" / "task-card-plan.json"
    task_card_plan_path.parent.mkdir()
    task_card_plan = {"task_cards": [{"id": "card", "site": "gitlab"}]}
    task_card_plan_path.write_text(json.dumps(task_card_plan))
    state = {
        "task_card_plan_digest": module.task_card_plan_digest(task_card_plan),
        "run_definition": {
            "contributions": {
                "pipeline": {
                    "benchmark_path": str(benchmark),
                    "manifest_path": str(manifest),
                },
                "phase_1": {
                    "sandbox_model": module.MODEL,
                    "task_card_plan_path": str(task_card_plan_path),
                },
            }
        },
    }
    (source_run / "pipeline_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(module, "load_and_validate_profile", lambda *_a, **_k: {})

    loaded = module.load_source_inputs(source_run)

    assert loaded.task_card_plan == task_card_plan
    assert loaded.benchmark_root == benchmark


def _configure_orchestration(module: ModuleType, monkeypatch, tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    output = logs / "provider-canary"
    monkeypatch.setattr(module, "PACKAGE_LOGS_ROOT", logs)
    monkeypatch.setattr(module, "load_source_inputs", lambda _path: _source(module, tmp_path))
    monkeypatch.setattr(
        module,
        "select_one_card",
        lambda _plan, card_id: {"backend": "direct" if card_id == "direct" else "sandbox"},
    )
    monkeypatch.setattr(
        module,
        "_use_contract_bound_action_api",
        lambda plan: plan["backend"] == "direct",
    )
    monkeypatch.setattr(module, "validate_frozen_route", lambda _env: "sentinel-token")
    monkeypatch.setattr(module, "check_openrouter_capacity", lambda _token: None)
    return output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_is_api", "sandbox_is_api", "match"),
    [
        (False, False, "direct card"),
        (True, True, "sandbox card"),
    ],
)
async def test_route_mismatch_stops_before_capacity_generation_or_modal(
    module: ModuleType,
    monkeypatch,
    tmp_path: Path,
    direct_is_api: bool,
    sandbox_is_api: bool,
    match: str,
) -> None:
    output = _configure_orchestration(module, monkeypatch, tmp_path)
    calls: list[str] = []

    def classify(plan):
        return direct_is_api if plan["backend"] == "direct" else sandbox_is_api

    monkeypatch.setattr(module, "_use_contract_bound_action_api", classify)
    monkeypatch.setattr(
        module, "check_openrouter_capacity", lambda _token: calls.append("capacity")
    )
    monkeypatch.setattr(
        module, "generate_new_tasks_for_site", lambda **_kwargs: calls.append("generation")
    )
    monkeypatch.setattr(module, "preflight_sandbox_environment", lambda: calls.append("modal"))

    with pytest.raises(module.MicrocanaryError, match=match):
        await module.run_microcanaries(
            source_run=tmp_path / "source",
            output_root=output,
            direct_card_id="direct",
            sandbox_card_id="sandbox",
        )

    assert calls == []


@pytest.mark.asyncio
async def test_capacity_failure_stops_before_generation_or_modal(
    module: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    output = _configure_orchestration(module, monkeypatch, tmp_path)
    calls: list[str] = []

    def fail_capacity(_token):
        calls.append("capacity")
        raise module.MicrocanaryError("no capacity", code="capacity_unavailable")

    monkeypatch.setattr(module, "check_openrouter_capacity", fail_capacity)
    monkeypatch.setattr(
        module, "generate_new_tasks_for_site", lambda **_kwargs: calls.append("generation")
    )
    monkeypatch.setattr(module, "preflight_sandbox_environment", lambda: calls.append("modal"))

    with pytest.raises(module.MicrocanaryError, match="no capacity"):
        await module.run_microcanaries(
            source_run=tmp_path / "source",
            output_root=output,
            direct_card_id="direct",
            sandbox_card_id="sandbox",
        )

    assert calls == ["capacity"]
    assert not output.exists()


@pytest.mark.asyncio
async def test_fail_closed_route_checks_do_not_enter_paid_or_modal_boundaries(
    module: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    output = _configure_orchestration(module, monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "validate_frozen_route",
        lambda _env: (_ for _ in ()).throw(module.MicrocanaryError("bad route")),
    )
    monkeypatch.setattr(
        module, "generate_new_tasks_for_site", lambda **_kwargs: calls.append("paid")
    )
    monkeypatch.setattr(module, "preflight_sandbox_environment", lambda: calls.append("modal"))

    with pytest.raises(module.MicrocanaryError, match="bad route"):
        await module.run_microcanaries(
            source_run=tmp_path / "source",
            output_root=output,
            direct_card_id="direct",
            sandbox_card_id="sandbox",
        )

    assert calls == []
    assert not output.exists()


@pytest.mark.asyncio
async def test_direct_failure_never_enters_sandbox_boundary(
    module: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    output = _configure_orchestration(module, monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        module, "compute_generate_new_tasks_shared_inputs_fingerprint", lambda **_k: "shared"
    )
    monkeypatch.setattr(module, "compute_site_cache_fingerprint", lambda **_k: "site")

    async def failed_direct(**_kwargs):
        calls.append("direct")
        return SiteGenerateNewTasksResult("gitlab", [], ["provider error"])

    async def forbidden_modal():
        calls.append("modal")

    monkeypatch.setattr(module, "generate_new_tasks_for_site", failed_direct)
    monkeypatch.setattr(module, "preflight_sandbox_environment", forbidden_modal)

    with pytest.raises(module.MicrocanaryError, match="returned 1 error"):
        await module.run_microcanaries(
            source_run=tmp_path / "source",
            output_root=output,
            direct_card_id="direct",
            sandbox_card_id="sandbox",
        )

    assert calls == ["direct"]
    assert not (output / "phase_1" / "sandbox").exists()


@pytest.mark.asyncio
async def test_success_uses_production_facade_fingerprints_and_serial_boundaries(
    module: ModuleType, monkeypatch, tmp_path: Path
) -> None:
    output = _configure_orchestration(module, monkeypatch, tmp_path)
    calls: list[tuple[str, object]] = []

    def shared_fingerprint(**kwargs):
        calls.append(("shared", kwargs["task_card_plan"]["backend"]))
        return f"shared-{kwargs['task_card_plan']['backend']}"

    def site_fingerprint(**kwargs):
        calls.append(("site", kwargs["shared_inputs_fingerprint"]))
        assert kwargs["novel_tasks_per_site"] == 1
        return f"site-{kwargs['shared_inputs_fingerprint']}"

    async def generate(**kwargs):
        boundary = kwargs["task_card_plan"]["backend"]
        calls.append(("generate", boundary))
        assert kwargs["_allow_task_card_slicing"] is False
        assert kwargs["novel_tasks_per_site"] == 1
        assert kwargs["output_dir"] == output / "phase_1" / boundary
        if boundary == "direct":
            assert kwargs["benchmark_volume"] is None
        else:
            assert kwargs["benchmark_volume"] == "production-volume"
        return SiteGenerateNewTasksResult("gitlab", [{"id": boundary}], [])

    async def preflight():
        calls.append(("modal-preflight", True))

    async def upload(path):
        calls.append(("upload", path))
        return "production-volume"

    def capacity(_token):
        calls.append(("capacity", True))

    monkeypatch.setattr(
        module, "compute_generate_new_tasks_shared_inputs_fingerprint", shared_fingerprint
    )
    monkeypatch.setattr(module, "compute_site_cache_fingerprint", site_fingerprint)
    monkeypatch.setattr(module, "generate_new_tasks_for_site", generate)
    monkeypatch.setattr(module, "preflight_sandbox_environment", preflight)
    monkeypatch.setattr(module, "upload_to_volume", upload)
    monkeypatch.setattr(module, "check_openrouter_capacity", capacity)

    await module.run_microcanaries(
        source_run=tmp_path / "source",
        output_root=output,
        direct_card_id="direct",
        sandbox_card_id="sandbox",
    )

    assert [name for name, _value in calls] == [
        "capacity",
        "shared",
        "site",
        "generate",
        "modal-preflight",
        "upload",
        "shared",
        "site",
        "generate",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_result,match",
    [
        (SiteGenerateNewTasksResult("gitlab", [], []), "returned 0 rows"),
        (SiteGenerateNewTasksResult("gitlab", [{}, {}], []), "returned 2 rows"),
        (SiteGenerateNewTasksResult("gitlab", [], ["bad row"]), "returned 1 error"),
    ],
)
async def test_sandbox_wrong_rows_or_errors_fail(
    module: ModuleType,
    monkeypatch,
    tmp_path: Path,
    bad_result: SiteGenerateNewTasksResult,
    match: str,
) -> None:
    output = _configure_orchestration(module, monkeypatch, tmp_path)
    monkeypatch.setattr(
        module, "compute_generate_new_tasks_shared_inputs_fingerprint", lambda **_k: "shared"
    )
    monkeypatch.setattr(module, "compute_site_cache_fingerprint", lambda **_k: "site")
    results = iter([SiteGenerateNewTasksResult("gitlab", [{}], []), bad_result])

    async def generate(**_kwargs):
        return next(results)

    async def preflight():
        return None

    async def upload(_path):
        return "volume"

    monkeypatch.setattr(module, "generate_new_tasks_for_site", generate)
    monkeypatch.setattr(module, "preflight_sandbox_environment", preflight)
    monkeypatch.setattr(module, "upload_to_volume", upload)

    with pytest.raises(module.MicrocanaryError, match=match):
        await module.run_microcanaries(
            source_run=tmp_path / "source",
            output_root=output,
            direct_card_id="direct",
            sandbox_card_id="sandbox",
        )


@pytest.mark.parametrize(
    "env,match",
    [
        ({"ANTHROPIC_AUTH_TOKEN": "token"}, "ANTHROPIC_BASE_URL"),
        ({"ANTHROPIC_BASE_URL": "https://openrouter.ai/api"}, "ANTHROPIC_AUTH_TOKEN"),
        (
            {
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/",
                "ANTHROPIC_AUTH_TOKEN": "token",
                "WORLDSIM_PHASE1_CONTRACT_BOUND_API": "1",
                "CLAUDE_CODE_OAUTH_TOKEN": "oauth",
            },
            "higher-precedence",
        ),
        (
            {
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/",
                "ANTHROPIC_AUTH_TOKEN": "token",
            },
            "WORLDSIM_PHASE1_CONTRACT_BOUND_API",
        ),
    ],
)
def test_route_validation_fails_closed(module: ModuleType, env: dict[str, str], match: str) -> None:
    with pytest.raises(module.MicrocanaryError, match=match):
        module.validate_frozen_route(env)


def test_main_emits_fixed_code_without_exception_or_secret_text(
    module: ModuleType, monkeypatch, capsys
) -> None:
    secret = "sentinel-secret-value"
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", secret)

    async def fail(**_kwargs):
        raise module.MicrocanaryError(f"provider rejected {secret}")

    monkeypatch.setattr(module, "run_microcanaries", fail)
    code = module.main(
        [
            "--source-run",
            "source",
            "--output-root",
            "logs/out",
            "--direct-card-id",
            "direct",
            "--sandbox-card-id",
            "sandbox",
        ]
    )

    rendered = capsys.readouterr().err
    assert code == 1
    assert secret not in rendered
    assert "provider rejected" not in rendered
    assert '"error_code": "microcanary_invariant_failed"' in rendered
