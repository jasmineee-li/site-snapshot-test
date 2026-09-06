"""Per-site novel-task generation: eligibility, retries, prompts, and merges."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from warp_taskgen.adversarial_actions.capability_task_cards import compile_capability_task_card_plan
from warp_taskgen.phase_1 import novel_task_validation as phase_1_generate_new_tasks_validation
from warp_taskgen.phases import phase_1_existing_tasks, phase_1_generate_new_tasks

from ._fixtures import (  # noqa: F401
    _add_gitlab_issue_sample,
    _add_reddit_submission_sample,
    _agent_context,
    _novel_task,
    _novel_task_list,
    _profile,
    _raw_task,
    _site_cache_metadata,
    _stub_generate_new_tasks_sandbox_preflight,
)


def test_load_generate_new_tasks_eligible_sites_uses_carrier_routes_not_coverage_gaps(
    tmp_path,
):
    profiles_dir = tmp_path / "phase_0c"
    profiles_dir.mkdir()
    (profiles_dir / "BENCHMARK_PROFILE_shopping.json").write_text(
        json.dumps(_profile(uncovered=["surface-1"]))
    )
    covered_reddit = _profile(uncovered=[])
    covered_reddit["site_name"] = "reddit"
    _add_reddit_submission_sample(covered_reddit)
    covered_reddit["existing_task_coverage"]["injection_surfaces_with_task_coverage"] = [
        "submissionbodydetail",
        "commentbodythread",
    ]
    (profiles_dir / "BENCHMARK_PROFILE_reddit.json").write_text(json.dumps(covered_reddit))

    eligible = phase_1_generate_new_tasks.load_generate_new_tasks_eligible_sites(
        profiles_dir=profiles_dir,
        manifest_eval_types=["NetworkEventEvaluator", "AgentResponseEvaluator"],
    )

    assert [site.site_name for site in eligible] == ["reddit"]


def test_load_generate_new_tasks_eligible_sites_honors_site_filter(tmp_path):
    profiles_dir = tmp_path / "phase_0c"
    profiles_dir.mkdir()
    for site_name in ("gitlab", "reddit", "shopping"):
        profile = _profile(uncovered=["surface-1"])
        profile["site_name"] = site_name
        if site_name == "gitlab":
            _add_gitlab_issue_sample(profile)
        if site_name == "reddit":
            _add_reddit_submission_sample(profile)
        (profiles_dir / f"BENCHMARK_PROFILE_{site_name}.json").write_text(json.dumps(profile))

    eligible = phase_1_generate_new_tasks.load_generate_new_tasks_eligible_sites(
        profiles_dir=profiles_dir,
        manifest_eval_types=["NetworkEventEvaluator", "AgentResponseEvaluator"],
        site_filter={"gitlab", "reddit"},
    )

    assert [site.site_name for site in eligible] == ["gitlab", "reddit"]


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_reuses_valid_cached_output(monkeypatch, tmp_path):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    cached_tasks = _novel_task_list()
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    manifest = {"evaluation": {"eval_types": ["NetworkEventEvaluator", "AgentResponseEvaluator"]}}
    site = phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="shopping",
        profile_path=tmp_path / "profile.json",
        profile=_profile(uncovered=["surface-1"]),
    )
    (output_dir / "novel_tasks_shopping.json").write_text(json.dumps(cached_tasks))
    (output_dir / "novel_tasks_shopping.json.metadata.json").write_text(
        json.dumps(
            _site_cache_metadata(
                benchmark_root=benchmark_root,
                manifest=manifest,
                site=site,
            )
        )
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("sandbox should not run when cached tasks validate")

    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", fail_if_called)

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=site,
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint=_site_cache_metadata(
            benchmark_root=benchmark_root,
            manifest=manifest,
            site=site,
        )["fingerprint"],
    )

    assert result.errors == []
    assert result.benign_tasks == cached_tasks


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_embeds_agent_context_when_available(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json"
    profile_path.parent.mkdir()
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))
    agent_context = _agent_context(requires_structured_output=True)
    (profile_path.parent / "AGENT_CONTEXT_shopping.json").write_text(json.dumps(agent_context))

    generated_tasks = _novel_task_list()
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "run_claude_in_sandbox",
        AsyncMock(
            return_value={
                phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps(generated_tasks),
                "_summary": None,
            }
        ),
    )

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )

    assert result.errors == []
    assert all(task["agent_context"] == agent_context for task in result.benign_tasks)


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_skips_active_site_with_no_route_families(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "profile.json"
    profile = _profile(uncovered=["catalog_sidebar"])
    profile["injection_surface"] = [
        {
            "id": "catalog_sidebar",
            "location_page": "/category/{id}",
        }
    ]
    profile_path.write_text(json.dumps(profile))

    mock_sandbox = AsyncMock(return_value={})
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", mock_sandbox)
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "build_task_route_contracts",
        lambda **kwargs: {"route_families": []},
    )

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="reddit",
            profile_path=profile_path,
            profile=profile,
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )

    assert result.errors == []
    assert result.benign_tasks == []
    assert mock_sandbox.call_count == 0
    contracts = json.loads((output_dir / "TASK_ROUTE_CONTRACTS_reddit.json").read_text())
    assert contracts["route_families"] == []


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_retries_once_and_succeeds(monkeypatch, tmp_path):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))

    invalid_task = _novel_task(start_urls=["__GITLAB__/orders"])
    valid_tasks = _novel_task_list()

    mock_sandbox = AsyncMock(
        side_effect=[
            {
                phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps([invalid_task]),
                "_summary": None,
            },
            {
                phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps(valid_tasks),
                "_summary": None,
            },
        ]
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", mock_sandbox)

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )

    assert result.errors == []
    assert result.benign_tasks == valid_tasks
    assert mock_sandbox.call_count == 2
    assert (output_dir / "novel_tasks_shopping.json").exists()


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_fails_after_max_retries(monkeypatch, tmp_path):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))

    invalid_task = _novel_task(start_urls=["__GITLAB__/orders"])
    mock_sandbox = AsyncMock(
        return_value={
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps([invalid_task]),
            "_summary": None,
        }
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", mock_sandbox)

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )

    assert result.benign_tasks == []
    assert "start_urls must use __SHOPPING__" in result.errors[0]
    assert (
        mock_sandbox.call_count
        == 1 + phase_1_generate_new_tasks.GENERATE_NEW_TASKS_FIX_MAX_ITERATIONS
    )
    assert not (output_dir / "novel_tasks_shopping.json").exists()


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_rejects_invalid_cached_output_and_regenerates(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))
    (output_dir / "novel_tasks_shopping.json").write_text(
        json.dumps([_novel_task(site="gitlab", task_id="novel_gitlab_1")])
    )

    regenerated_tasks = _novel_task_list()
    mock_sandbox = AsyncMock(
        return_value={
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps(regenerated_tasks),
            "_summary": None,
        }
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", mock_sandbox)

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )

    assert result.errors == []
    assert result.benign_tasks == regenerated_tasks
    assert mock_sandbox.call_count == 1


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_rejects_underfilled_cached_output(monkeypatch, tmp_path):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))
    (output_dir / "novel_tasks_shopping.json").write_text(json.dumps([_novel_task()]))

    regenerated_tasks = _novel_task_list()
    mock_sandbox = AsyncMock(
        return_value={
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps(regenerated_tasks),
            "_summary": None,
        }
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", mock_sandbox)

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )

    assert result.errors == []
    assert result.benign_tasks == regenerated_tasks
    assert mock_sandbox.call_count == 1


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_handles_missing_and_invalid_sandbox_outputs(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))

    missing_output = AsyncMock(return_value={"_summary": None})
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", missing_output)
    missing_result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )
    assert missing_result.errors == ["sandbox did not produce benign_tasks.json"]

    invalid_json = AsyncMock(
        return_value={
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: "{broken",
            "_summary": None,
        }
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", invalid_json)
    invalid_result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="test-cache-fingerprint",
    )
    assert "invalid sandbox JSON" in invalid_result.errors[0]


def test_merge_benign_tasks_is_deterministic():
    existing_task_wraps = [{"id": "1", "site": "shopping"}, {"id": "2", "site": "gitlab"}]
    novel = [
        {"id": "novel_shopping_2", "site": "shopping"},
        {"id": "novel_gitlab_1", "site": "gitlab"},
        {"id": "novel_shopping_1", "site": "shopping"},
    ]

    merged = phase_1_generate_new_tasks_validation.merge_benign_tasks(existing_task_wraps, novel)

    assert [task["id"] for task in merged] == [
        "1",
        "2",
        "novel_gitlab_1",
        "novel_shopping_1",
        "novel_shopping_2",
    ]


def test_render_generate_benign_tasks_prompt_preserves_literal_example_braces():
    prompt = phase_1_generate_new_tasks.render_generate_benign_tasks_prompt(
        site_name="shopping",
        num_tasks=17,
    )

    assert "{site_name}" not in prompt
    assert "{num_tasks}" not in prompt
    assert '"id": "novel_shopping_1"' in prompt
    assert "route_id" in prompt
    assert "{{" in prompt and "}}" in prompt
    assert "AGENT_CONTEXT.json" in prompt
    assert "TASK_ROUTE_CONTRACTS.json" in prompt
    assert "TASK_CARD_PLAN.json" in prompt
    assert "listing or parent page" in prompt
    assert "child detail URL from another route family" in prompt
    assert "Phase 2" not in prompt
    assert "GitLab: generate issue-only" not in prompt


def test_render_generate_benign_tasks_prompt_uses_action_only_contract_for_paper_profile():
    plan = compile_capability_task_card_plan("tier2_pure_action_paper", sites={"reddit"})

    prompt = phase_1_generate_new_tasks.render_generate_benign_tasks_prompt(
        site_name="reddit",
        num_tasks=12,
        task_card_plan=plan,
    )

    assert "HostActionOnlyPlaceholder" in prompt
    assert "AgentResponseEvaluator" not in prompt
    assert "link/no-link" not in prompt
    assert "blank/populated" not in prompt
    assert "answer with exactly" not in prompt.casefold()
    assert '"id": "novel_reddit_1"' in prompt
    assert "{site_name}" not in prompt
    assert "{num_tasks}" not in prompt


def test_wrap_task_preserves_instantiation_dict():
    wrapped = phase_1_existing_tasks._wrap_task(
        _raw_task(instantiation_dict={"retrieved_data_format_spec": "Return postcode fields."})
    )

    assert wrapped["instantiation_dict"] == {
        "retrieved_data_format_spec": "Return postcode fields."
    }


@pytest.mark.asyncio
async def test_generate_new_tasks_for_site_passes_explicit_sandbox_model(monkeypatch, tmp_path):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json"
    profile_path.parent.mkdir()
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))
    seen: dict[str, str | None] = {"model": None}

    async def fake_run_claude_in_sandbox(**kwargs):
        seen["model"] = kwargs.get("model")
        return {
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps(_novel_task_list()),
            "_summary": None,
        }

    monkeypatch.setattr(
        phase_1_generate_new_tasks, "run_claude_in_sandbox", fake_run_claude_in_sandbox
    )

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=_profile(uncovered=["surface-1"]),
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="cache-fp",
        sandbox_model="claude-opus-4-6",
    )

    assert result.errors == []
    assert seen["model"] == "claude-opus-4-6"
