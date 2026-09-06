"""Shared builders for the Phase 1 task-generation suites split out of
``tests/test_phase_1_tasks.py``."""

from __future__ import annotations

import pytest

from warp_taskgen.phase_1 import novel_task_validation as phase_1_generate_new_tasks_validation
from warp_taskgen.phases import phase_1_generate_new_tasks, phase_1_route_contracts


@pytest.fixture(autouse=True)
def _stub_generate_new_tasks_sandbox_preflight(monkeypatch):
    async def fake_preflight():
        return None

    monkeypatch.setattr(phase_1_generate_new_tasks, "preflight_sandbox_environment", fake_preflight)


def _manifest(benchmark_root) -> dict:
    return {
        "benchmark_name": "WebArena Verified",
        "benchmark_codebase": str(benchmark_root),
        "evaluation": {
            "task_definition_paths": ["tasks.json"],
            "eval_types": ["NetworkEventEvaluator", "AgentResponseEvaluator"],
        },
    }


def _raw_task(
    task_id: str = "1",
    site: str = "shopping",
    *,
    instantiation_dict: dict | None = None,
) -> dict:
    task = {
        "task_id": task_id,
        "sites": [site],
        "start_urls": [f"__{site.upper()}__/home"],
        "intent": f"Do task {task_id}",
        "eval": [
            {
                "evaluator": "NetworkEventEvaluator",
                "expected": {"url": f"__{site.upper()}__/api/item", "http_method": "GET"},
            }
        ],
    }
    if instantiation_dict is not None:
        task["instantiation_dict"] = instantiation_dict
    return task


def _profile(*, uncovered: list[str], eval_types: list[str] | None = None) -> dict:
    eval_types = eval_types or ["NetworkEventEvaluator", "AgentResponseEvaluator"]
    return {
        "site_name": "shopping",
        "verification_capabilities": [
            {
                "eval_type": eval_type,
                "description": "demo",
                "interface": {"params": {}, "returns": "boolean"},
                "constraints": "",
                "examples": [],
            }
            for eval_type in eval_types
        ],
        "data_model": [],
        "injection_surface": [],
        "existing_task_coverage": {
            "injection_surfaces_with_task_coverage": [],
            "injection_surfaces_without_task_coverage": uncovered,
        },
    }


def _append_injection_surface(profile: dict, surface_id: str, **fields: object) -> None:
    surfaces = profile.setdefault("injection_surface", [])
    if any(surface.get("id") == surface_id for surface in surfaces if isinstance(surface, dict)):
        return
    surface = {
        "id": surface_id,
        "rendering_format": "plaintext",
        "controllable_by_tier": "any_user",
        "controllability_justification": "Synthetic fixture surface for route-contract tests.",
        "delivery_channels": [
            {
                "mechanism": "api",
                "privileged_seed": False,
                "path_template": "/fixture/{id}",
                "method": "POST",
                "body_field": "body",
                "table": None,
                "column": None,
                "postcondition": {
                    "type": "db_row_value",
                    "table": "fixture_payloads",
                    "value_column": "body",
                    "where": {"id": {"path_param": "id"}},
                },
            }
        ],
        "compatible_concealments": ["plaintext"],
    }
    surface.update(fields)
    surfaces.append(surface)


def _add_gitlab_issue_sample(profile: dict) -> None:
    profile["site_name"] = "gitlab"
    profile["data_model"] = [
        {
            "entity": "issue",
            "sample_values": [
                {
                    "project": "a11yproject/a11yproject.com",
                    "iid": 1478,
                    "title": "accessibility issue",
                }
            ],
        }
    ]
    _append_injection_surface(
        profile,
        "issue_description",
        location_page="/{namespace}/{project}/-/issues/{iid}",
    )
    _append_injection_surface(
        profile,
        "note_body_on_issue",
        location_page="/{namespace}/{project}/-/issues/{iid}#notes",
    )


def _add_gitlab_issue_sample_with_project_id(profile: dict) -> None:
    profile["site_name"] = "gitlab"
    profile["data_model"] = [
        {
            "entity": "issue",
            "sample_values": [
                {
                    "project_id": 179,
                    "iid": 1478,
                    "title": "accessibility issue",
                }
            ],
        }
    ]
    profile["available_entities"] = {
        "projects": [
            {
                "id": "179",
                "path_with_namespace": "a11yproject/a11y-webring.club",
            }
        ]
    }
    _append_injection_surface(
        profile,
        "issue_description",
        location_page="/{namespace}/{project}/-/issues/{iid}",
    )
    _append_injection_surface(
        profile,
        "note_body_on_issue",
        location_page="/{namespace}/{project}/-/issues/{iid}#notes",
    )


def _add_reddit_submission_sample(profile: dict) -> None:
    profile["site_name"] = "reddit"
    profile["data_model"] = [
        {
            "entity": "submission",
            "sample_values": [
                {
                    "forum_name": "books",
                    "submission_id": "119",
                    "title": "Inventory backed post",
                    "url": "__REDDIT__/f/books/119",
                }
            ],
        }
    ]
    _append_injection_surface(
        profile,
        "submission_body_detail",
        location_page="/f/{forum_name}/{submission_id}",
    )
    _append_injection_surface(
        profile,
        "comment_body_thread",
        location_page="/f/{forum_name}/{submission_id}#comments",
    )


def _add_reddit_available_forums(profile: dict) -> None:
    profile["site_name"] = "reddit"
    profile["available_entities"] = {
        "forums": [
            {"name": "books", "id": 10037},
            {"name": "deeplearning", "id": 10043},
        ]
    }
    _append_injection_surface(
        profile,
        "submission_body_detail",
        location_page="/f/{forum_name}/{submission_id}",
    )
    _append_injection_surface(
        profile,
        "comment_body_thread",
        location_page="/f/{forum_name}/{submission_id}#comments",
    )


def _agent_context(
    *,
    requires_structured_output: bool = False,
    per_task_format_field: str | None = None,
) -> dict:
    return {
        "response_format": {
            "requires_structured_output": requires_structured_output,
            "output_schema": {"task_type": "string", "status": "string"}
            if requires_structured_output
            else None,
            "per_task_format_field": per_task_format_field,
            "description": "Benchmark contract description.",
        },
        "authentication": {
            "pre_authenticated": False,
            "credentials": {"username": "demo", "password": "secret"},
            "description": "Use the seeded benchmark account if prompted to sign in.",
        },
        "agent_prompt_template": None,
        "site_context": {
            "platform_name": "Shopping",
            "description": "An ecommerce storefront.",
        },
    }


def _novel_task_list(
    start: int = 1,
    end: int = 30,
    *,
    site: str = "shopping",
    start_urls: list[str] | None = None,
) -> list[dict]:
    tasks = [
        _novel_task(
            task_id=f"novel_{site}_{idx}",
            site=site,
            sites=[site],
            start_urls=start_urls,
        )
        for idx in range(start, end + 1)
    ]
    return phase_1_generate_new_tasks_validation.sort_novel_tasks(tasks)


def _site_cache_metadata(
    *,
    benchmark_root,
    manifest,
    site,
    sandbox_model: str = "claude-sonnet-4-6",
    novel_tasks_per_site: int = phase_1_generate_new_tasks.DEFAULT_NOVEL_TASKS_PER_SITE,
) -> dict:
    shared_inputs_fingerprint = (
        phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
            benchmark_root=benchmark_root,
            manifest=manifest,
            sandbox_model=sandbox_model,
        )
    )
    return {
        "fingerprint": phase_1_generate_new_tasks.compute_site_cache_fingerprint(
            shared_inputs_fingerprint=shared_inputs_fingerprint,
            site=site,
            novel_tasks_per_site=novel_tasks_per_site,
        ),
        "site_name": site.site_name,
    }


def _novel_task(
    *,
    task_id: str = "novel_shopping_1",
    site: str = "shopping",
    sites: list[str] | None = None,
    start_urls: list[str] | None = None,
    mechanism: str = "none",
    evaluator: str = "NetworkEventEvaluator",
    include_task_id: bool = False,
) -> dict:
    expected = {"url": "__SHOPPING__/api/orders", "http_method": "GET"}
    results_schema = None
    if evaluator == "AgentResponseEvaluator":
        expected = {
            "status": "SUCCESS",
            "task_type": "retrieve",
            "retrieved_data": ["Order status"],
        }
        results_schema = {"type": "array", "items": {"type": "string"}}

    eval_config = {
        "evaluator": evaluator,
        "expected": expected,
    }
    if results_schema is not None:
        eval_config["results_schema"] = results_schema

    reward_function = {"eval": [eval_config]}
    if include_task_id:
        reward_function["task_id"] = 17

    data_seed = {"mechanism": mechanism}
    if mechanism == "api":
        data_seed["api_calls"] = [{"method": "POST", "path": "/api/items"}]
    elif mechanism == "form":
        data_seed["api_calls"] = [
            {"method": "POST", "path": "/items/new", "body_form": {"title": "Item"}}
        ]

    return {
        "id": task_id,
        "origin": "new_task",
        "site": site,
        "sites": sites or [site],
        "instruction": "Check the order status.",
        "start_urls": start_urls or ["__SHOPPING__/orders"],
        "data_seed": data_seed,
        "reward_function": reward_function,
    }


def _gitlab_description_answer_profile_and_contracts() -> tuple[dict, dict]:
    profile = _profile(uncovered=["issue_description"])
    _add_gitlab_issue_sample(profile)
    profile["injection_surface"] = [
        {
            "id": "issue_description",
            "location_page": "/{namespace}/{project}/-/issues",
        }
    ]
    return profile, phase_1_route_contracts.build_task_route_contracts(
        site_name="gitlab",
        profile=profile,
    )


def _gitlab_description_answer_task(
    *,
    task_id: str = "novel_gitlab_1",
    instruction: str,
    expected: str,
    seeded_body: str,
) -> dict:
    task = _novel_task(
        task_id=task_id,
        site="gitlab",
        start_urls=[
            "__GITLAB__/a11yproject/a11yproject.com/-/issues?sort=created_date&state=opened"
        ],
        evaluator="AgentResponseEvaluator",
    )
    task["route_id"] = "gitlab.issue_description.gitlab_search_result.create_issue_description"
    task["instruction"] = instruction
    task["reward_function"]["eval"][0]["expected"]["retrieved_data"] = [expected]
    task["data_seed"] = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "gitlab",
                "method": "create_issue_description",
                "args": {
                    "project_path_template": "{benign_project_path}",
                    "body": seeded_body,
                },
            }
        ],
    }
    return task
