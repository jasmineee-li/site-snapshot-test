"""Browser ledger and serialized-pair checks for the matched rewrite adapter."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from phase_4.test_matched_rewrite_study import _baseline, _candidate, _pass_qa
from warp_taskgen.phase_4 import matched_rewrite_study as study
from warp_taskgen.phase_4.matched_rewrite_contracts import (
    BrowserOutcome,
    DiagnosisOutcome,
    MatchedCallPolicy,
    MatchedStudyBudget,
    OrdinaryGuidance,
    Phase4Runtime,
    ProposalOutcome,
    TPGuidance,
    Usage,
)
from warp_taskgen.phase_4.matched_rewrite_ordinary_api import browser_usage
from warp_taskgen.phase_4.matched_rewrite_provider import ExistingPhase4AttemptAdapter


def _config() -> study.MatchedRewriteStudyConfig:
    return study.MatchedRewriteStudyConfig(
        call_policy=MatchedCallPolicy(
            model="sandbox-model",
            provider="openrouter",
            runner="messages",
        ),
        budget=MatchedStudyBudget(
            per_arm_max_tokens=100,
            total_max_tokens=500,
            per_arm_max_cost_usd=10.0,
            total_max_cost_usd=20.0,
        ),
    )


def _runtime(tmp_path: Path) -> Phase4Runtime:
    return Phase4Runtime(
        primary_instance=object(),
        all_instances=(),
        agent_factory=lambda: object(),
        task_dir_root=tmp_path,
        host_client=object(),
        host_provider="openrouter",
        browser_runner="agentlab",
    )


def _write_ledger(path: Path, *, provider: str = "openrouter") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "request_model": "sandbox-model",
            "provider": provider,
            "runner": "agentlab",
            "usage": {"input_tokens": 3, "output_tokens": 4, "cost_usd": 0.11},
        },
        {
            "request_model": "sandbox-model",
            "provider": provider,
            "runner": "agentlab",
            "usage": {"input_tokens": 5, "output_tokens": 6, "cost_usd": 0.13},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_agentlab_sidecar_retains_provider_reported_cost(tmp_path):
    source = Path(__file__).parents[2] / "packages/worldsim-agentlab-runner/src/worldsim_agentlab_runner/model_args.py"
    spec = importlib.util.spec_from_file_location("matched_sidecar_model_args", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    metadata = tmp_path / "worldsim_model_calls.jsonl"
    model = module.WorldSimChatModel(
        model_name="sandbox-model",
        transport="openrouter",
        provider="openrouter",
        required_env_var="",
        temperature=None,
        max_tokens=128,
        extra_body={},
        max_retry=1,
        min_retry_wait_time=0,
        metadata_path=str(metadata),
    )
    model._record_usage(
        SimpleNamespace(
            model="sandbox-model",
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=4,
                total_tokens=7,
                cost=0.42,
            ),
        )
    )
    model._record_call_metadata(SimpleNamespace(model="sandbox-model"))
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["runner"] == "agentlab"
    assert payload["usage"]["cost_usd"] == 0.42


def test_browser_usage_requires_exact_complete_sidecar_ledger(tmp_path):
    ledger = tmp_path / "worldsim_model_calls.jsonl"
    _write_ledger(ledger)
    usage = browser_usage(
        {},
        artifact_dir=tmp_path,
        expected_model="sandbox-model",
        expected_provider="openrouter",
        expected_runner="agentlab",
    )
    assert usage == Usage(8, 10, 0.24, attempts=2)

    ledger.write_text(ledger.read_text().replace('"cost_usd": 0.13', '"cost_usd": null'))
    missing_cost = browser_usage(
        {},
        artifact_dir=tmp_path,
        expected_model="sandbox-model",
        expected_provider="openrouter",
        expected_runner="agentlab",
    )
    assert missing_cost.unavailable_reason == "browser_usage_malformed"

    ledger.write_text(ledger.read_text().replace('"cost_usd": null', '"cost_usd": 1e999'))
    nonfinite_cost = browser_usage(
        {},
        artifact_dir=tmp_path,
        expected_model="sandbox-model",
        expected_provider="openrouter",
        expected_runner="agentlab",
    )
    assert nonfinite_cost.unavailable_reason == "browser_usage_malformed"


def test_existing_adapter_pair_uses_serialized_exact_browser_ledgers(monkeypatch, tmp_path):
    baseline = _baseline()
    config = _config()
    adapter = ExistingPhase4AttemptAdapter(_runtime(tmp_path))
    calls: list[str] = []

    async def diagnose(self, request):
        calls.append(f"{request.arm}:{request.stage}")
        return DiagnosisOutcome(
            "ok",
            TPGuidance(trigger_source="non-real TP", rewrite_guidance="Keep anchors."),
            Usage(1, 1, 0.01),
        )

    async def ordinary_critique(self, request):
        calls.append(f"{request.arm}:{request.stage}")
        return DiagnosisOutcome(
            "ok",
            OrdinaryGuidance(critique="Keep the wording clear."),
            Usage(1, 1, 0.01),
        )

    async def rewrite(self, request):
        calls.append(f"{request.arm}:{request.stage}")
        return ProposalOutcome("ok", _candidate(baseline.task_copy()), Usage(1, 1, 0.01))

    async def evaluate(*args, **kwargs):
        namespace = Path(args[7])
        calls.append(f"{namespace.parent.name}:browser")
        _write_ledger(namespace / "worldsim_model_calls.jsonl")
        return {
            "task_id": "task-1",
            "outcome": "refused_or_ignored",
            "trajectory_dir": str(namespace),
        }

    from warp_taskgen.phase_4 import variant_eval

    monkeypatch.setattr(study, "validate_baseline_binding", lambda *_args: None)
    monkeypatch.setattr(study, "_qa", _pass_qa)
    monkeypatch.setattr(study, "_select", lambda _baseline, rows: {"arms": rows})
    monkeypatch.setattr(ExistingPhase4AttemptAdapter, "_diagnose", diagnose)
    monkeypatch.setattr(ExistingPhase4AttemptAdapter, "_ordinary_critique", ordinary_critique)
    monkeypatch.setattr(ExistingPhase4AttemptAdapter, "_rewrite", rewrite)
    monkeypatch.setattr(variant_eval, "_evaluate_variant", evaluate)

    result = asyncio.run(
        study.run_matched_rewrite_study(
            baseline,
            attempt_provider=adapter,
            config=config,
        )
    )
    arms = result["primary"]["pairs"][0]["arms"]
    assert arms["tp_guided"]["status"] == arms["ordinary"]["status"] == "evaluated"
    assert result["primary"]["accounting"]["browser_attempts"] == 2
    assert result["primary"]["accounting"]["usage_status"] == "available"
    assert [item for item in calls if item.endswith(":browser")] == [
        "tp_guided:browser",
        "ordinary:browser",
    ]


def test_existing_adapter_blocks_after_unknown_or_mismatched_browser_ledger(monkeypatch, tmp_path):
    baseline = _baseline()
    config = _config()
    adapter = ExistingPhase4AttemptAdapter(_runtime(tmp_path))
    adapter.bind(baseline.binding)

    async def evaluate(*args, **kwargs):
        namespace = Path(args[7])
        return {"task_id": "task-1", "trajectory_dir": str(namespace)}

    from warp_taskgen.phase_4 import variant_eval

    monkeypatch.setattr(variant_eval, "_evaluate_variant", evaluate)
    first = asyncio.run(
        adapter.run(study._request(baseline, config, arm="tp_guided", stage="browser", variant_task=_candidate(baseline.task_copy())))
    )
    second = asyncio.run(
        adapter.run(study._request(baseline, config, arm="ordinary", stage="browser", variant_task=_candidate(baseline.task_copy())))
    )
    assert isinstance(first, BrowserOutcome)
    assert first.usage.unavailable_reason == "browser_usage_recorded_by_phase4_artifact"
    assert isinstance(second, BrowserOutcome)
    assert second.failure == "matched_budget_usage_unknown"

    adapter = ExistingPhase4AttemptAdapter(_runtime(tmp_path / "mismatch"))
    adapter.bind(baseline.binding)

    async def mismatched_evaluate(*args, **kwargs):
        namespace = Path(args[7])
        _write_ledger(namespace / "worldsim_model_calls.jsonl", provider="anthropic")
        return {"task_id": "task-1", "trajectory_dir": str(namespace)}

    monkeypatch.setattr(variant_eval, "_evaluate_variant", mismatched_evaluate)
    mismatched = asyncio.run(
        adapter.run(study._request(baseline, config, arm="tp_guided", stage="browser", variant_task=_candidate(baseline.task_copy())))
    )
    blocked = asyncio.run(
        adapter.run(study._request(baseline, config, arm="ordinary", stage="browser", variant_task=_candidate(baseline.task_copy())))
    )
    assert mismatched.usage.unavailable_reason == "browser_usage_identity_mismatch"
    assert blocked.failure == "matched_budget_usage_unknown"
