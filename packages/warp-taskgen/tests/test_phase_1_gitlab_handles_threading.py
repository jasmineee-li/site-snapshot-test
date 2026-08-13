"""Lock in that Phase 1 propagates Phase 0c gitlab handle lists into each task.

Phase 2's URL-shape resolver disambiguates ``/<segment>`` gitlab URLs as
user_profile vs group via ``agent_context.gitlab.{user_handles,group_handles}``.
The lists are produced by Phase 0c (see
:mod:`warp_taskgen.phases.phase_0c_handle_enrichment`) and threaded through
Phase 1's existing ``_attach_agent_context_to_tasks`` flow. If anyone
later rewrites that flow to drop unknown agent_context keys, the
resolver silently degrades for every gitlab new_task entry. This test
guards against that regression.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from warp_taskgen.phases import phase_1_generate_new_tasks


def _profile(uncovered: list[str]) -> dict:
    return {
        "site_name": "shopping",
        "verification_capabilities": [
            {"eval_type": "AgentResponseEvaluator", "description": "Default."},
        ],
        "data_model": {"entities": [{"name": "Order", "fields": [{"name": "id"}]}]},
        "agent_context": {
            "response_format": {
                "requires_structured_output": True,
                "output_schema": {"task_type": "string", "status": "string"},
                "description": "Test contract.",
            },
            "authentication": {"pre_authenticated": False, "description": "Test."},
            "site_context": {"platform_name": "Shopping", "description": "Test."},
        },
        "injection_surface": [
            {"id": surface, "category": "test", "compatible_concealments": ["plaintext"]}
            for surface in uncovered
        ],
        "existing_task_coverage": {
            "injection_surfaces_with_task_coverage": [],
            "injection_surfaces_without_task_coverage": uncovered,
        },
    }


def _novel_task(task_id: str) -> dict:
    return {
        "id": task_id,
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": "test",
        "start_urls": ["__SHOPPING__/products/1"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "null"},
                    "expected": {
                        "task_type": "navigate",
                        "status": "SUCCESS",
                        "retrieved_data": None,
                    },
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_phase_1_threads_gitlab_handle_lists_into_each_task(monkeypatch, tmp_path):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    profile_path = tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json"
    profile_path.parent.mkdir()
    profile = _profile(uncovered=["surface-1"])
    profile_path.write_text(json.dumps(profile))

    agent_context = dict(profile["agent_context"])
    agent_context["gitlab"] = {
        "user_handles": ["root", "byteblaze", "primer"],
        "group_handles": ["a11yproject", "design"],
    }
    (profile_path.parent / "AGENT_CONTEXT_shopping.json").write_text(json.dumps(agent_context))

    generated_tasks = [_novel_task(f"novel_shopping_{i}") for i in range(1, 4)]
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
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "validate_generated_novel_tasks",
        lambda tasks, **kwargs: (tasks, []),
    )

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="shopping",
            profile_path=profile_path,
            profile=profile,
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="threading-test-fingerprint",
        novel_tasks_per_site=3,
    )

    assert result.errors == []
    assert len(result.benign_tasks) == 3
    for task in result.benign_tasks:
        ctx = task["agent_context"]
        assert ctx["gitlab"] == {
            "user_handles": ["root", "byteblaze", "primer"],
            "group_handles": ["a11yproject", "design"],
        }
