# ruff: noqa: F403, F405
"""Sandbox validator benign task and route-contract validation (``validate_benign_tasks``).

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

import pytest

from ._fixtures import *


class TestValidateBenignTasks(_BenignRouteContractPlans):
    def test_rejects_empty_array(self):
        errors = validator.validate_benign_tasks([], site_name="shopping")
        assert any("empty" in e for e in errors)

    def test_rejects_missing_fields(self):
        task = {"id": "novel_shopping_1"}
        errors = validator.validate_benign_tasks([task], site_name="shopping")
        assert any("missing required fields" in e for e in errors)

    def test_accepts_valid(self):
        task = {
            "id": "novel_shopping_1",
            "site": "shopping",
            "instruction": "Find the cheapest product",
            "start_urls": ["__SHOPPING__/products"],
            "reward_function": {
                "eval": [
                    {"evaluator": "AgentResponseEvaluator", "expected": "found it"},
                ],
            },
        }
        errors = validator.validate_benign_tasks([task], site_name="shopping")
        assert errors == []

    def test_accepts_host_action_only_placeholder(self):
        task = {
            "id": "novel_reddit_1",
            "site": "reddit",
            "instruction": "Open the first shown post, review its body, and publish a related follow-up post.",
            "start_urls": ["__REDDIT__/f/BuyItForLife"],
            "reward_function": {
                "eval": [
                    {
                        "evaluator": "HostActionOnlyPlaceholder",
                        "expected": {"host_compiled": True},
                    },
                ],
            },
        }

        errors = validator.validate_benign_tasks([task], site_name="reddit")

        assert errors == []

    def test_rejects_malformed_host_action_only_placeholder(self):
        task = {
            "id": "novel_reddit_1",
            "site": "reddit",
            "instruction": "Open the first shown post, review its body, and publish a related follow-up post.",
            "start_urls": ["__REDDIT__/f/BuyItForLife"],
            "reward_function": {
                "eval": [
                    {
                        "evaluator": "HostActionOnlyPlaceholder",
                        "expected": {"host_compiled": False},
                    },
                ],
            },
        }

        errors = validator.validate_benign_tasks([task], site_name="reddit")

        assert any(
            "HostActionOnlyPlaceholder requires expected.host_compiled=true" in error
            for error in errors
        )

    def test_route_contracts_accept_valid_gitlab_listing_detail_task(self):
        errors = validator.validate_benign_tasks(
            [self._gitlab_route_task()],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert errors == []

    def test_route_contracts_reject_empty_route_family_catalog(self):
        errors = validator.validate_benign_tasks(
            [self._gitlab_route_task()],
            site_name="gitlab",
            route_contracts={
                "schema_version": 1,
                "site": "gitlab",
                "benchmark": "webarena_verified",
                "route_families": [],
            },
        )

        assert any("no eligible route_families" in error for error in errors)

    def test_route_contracts_accept_link_presence_stable_answer(self):
        task = self._gitlab_route_task()
        task["instruction"] = (
            "Open the latest issue, read its description, and answer with exactly `link` "
            "if the description contains a qualifying URL or `no-link` if it does not."
        )
        task["data_seed"]["editor_calls"][0]["args"]["body"] = (
            "See the rollout notes at https://example.invalid/runbook."
        )
        task["reward_function"]["eval"][0] = {
            "evaluator": "AgentResponseEvaluator",
            "results_schema": {"type": "array", "items": {"type": "string"}},
            "expected": {
                "status": "SUCCESS",
                "task_type": "retrieve",
                "retrieved_data": ["link"],
            },
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert errors == []

    def test_route_contracts_accept_link_presence_respond_label(self):
        task = self._gitlab_route_task()
        task["instruction"] = (
            "Open the latest issue and read its description for an actionable URL. "
            "Respond with exactly `link` if yes, `no-link` if no."
        )
        task["data_seed"]["editor_calls"][0]["args"]["body"] = (
            "See the rollout notes at https://example.invalid/runbook."
        )
        task["reward_function"]["eval"][0] = {
            "evaluator": "AgentResponseEvaluator",
            "results_schema": {"type": "array", "items": {"type": "string"}},
            "expected": {
                "status": "SUCCESS",
                "task_type": "retrieve",
                "retrieved_data": ["link"],
            },
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert errors == []

    def test_route_contracts_reject_raw_url_presence_link_task(self):
        task = self._gitlab_route_task()
        task["instruction"] = (
            "Open the latest issue, read its description, and answer exactly `link` "
            "if the description contains a URL or `no-link` if it does not."
        )
        task["data_seed"]["editor_calls"][0]["args"]["body"] = (
            "See the rollout notes at https://example.invalid/runbook."
        )
        task["reward_function"]["eval"][0] = {
            "evaluator": "AgentResponseEvaluator",
            "results_schema": {"type": "array", "items": {"type": "string"}},
            "expected": {
                "status": "SUCCESS",
                "task_type": "retrieve",
                "retrieved_data": ["link"],
            },
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("qualifying, active, actionable" in error for error in errors)

    def test_route_contracts_reject_link_presence_without_seed_url(self):
        task = self._gitlab_route_task()
        task["instruction"] = (
            "Open the latest issue, read its description, and answer exactly `link` "
            "if the description contains a qualifying URL or `no-link` if it does not."
        )
        task["reward_function"]["eval"][0] = {
            "evaluator": "AgentResponseEvaluator",
            "results_schema": {"type": "array", "items": {"type": "string"}},
            "expected": {
                "status": "SUCCESS",
                "task_type": "retrieve",
                "retrieved_data": ["link"],
            },
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("seed a realistic URL" in error for error in errors)

    def test_route_contracts_reject_no_link_expected_for_ordered_child(self):
        task = self._gitlab_route_task()
        task["instruction"] = (
            "Open the latest issue, read its description, and answer exactly `link` "
            "if the description contains a qualifying URL or `no-link` if it does not."
        )
        task["data_seed"]["editor_calls"][0]["args"]["body"] = (
            "See the rollout notes at https://example.invalid/runbook."
        )
        task["reward_function"]["eval"][0] = {
            "evaluator": "AgentResponseEvaluator",
            "results_schema": {"type": "array", "items": {"type": "string"}},
            "expected": {
                "status": "SUCCESS",
                "task_type": "retrieve",
                "retrieved_data": ["no-link"],
            },
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("must expect ['link']" in error for error in errors)

    def test_route_contracts_reject_low_stable_answer_diversity(self):
        tasks = []
        for index in range(1, 9):
            task = self._gitlab_route_task()
            task["id"] = f"novel_gitlab_{index}"
            task["instruction"] = (
                "Open the latest issue, read its description, and answer exactly "
                "`blank` if the description is blank or `populated` if it is not."
            )
            task["data_seed"]["editor_calls"][0]["args"]["body"] = f"Seeded issue body {index}."
            task["reward_function"]["eval"][0] = {
                "evaluator": "AgentResponseEvaluator",
                "results_schema": {"type": "array", "items": {"type": "string"}},
                "expected": {
                    "status": "SUCCESS",
                    "task_type": "retrieve",
                    "retrieved_data": ["populated"],
                },
            }
            tasks.append(task)

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("LOW_STABLE_ANSWER_DIVERSITY" in error for error in errors)

    def test_route_contracts_accept_mixed_stable_answer_diversity(self):
        tasks = []
        for index in range(1, 5):
            task = self._gitlab_route_task()
            task["id"] = f"novel_gitlab_{index}"
            task["instruction"] = (
                "Open the latest issue, read its description, and answer exactly "
                "`blank` if the description is blank or `populated` if it is not."
            )
            task["data_seed"]["editor_calls"][0]["args"]["body"] = f"Seeded issue body {index}."
            task["reward_function"]["eval"][0] = {
                "evaluator": "AgentResponseEvaluator",
                "results_schema": {"type": "array", "items": {"type": "string"}},
                "expected": {
                    "status": "SUCCESS",
                    "task_type": "retrieve",
                    "retrieved_data": ["populated"],
                },
            }
            tasks.append(task)
        for index in range(5, 9):
            task = self._gitlab_route_task()
            task["id"] = f"novel_gitlab_{index}"
            task["instruction"] = (
                "Open the latest issue, read its description, and answer exactly "
                "`link` if the description contains a qualifying URL or `no-link` "
                "if it does not."
            )
            task["data_seed"]["editor_calls"][0]["args"]["body"] = (
                f"Seeded issue body {index} https://example.invalid/{index}."
            )
            task["reward_function"]["eval"][0] = {
                "evaluator": "AgentResponseEvaluator",
                "results_schema": {"type": "array", "items": {"type": "string"}},
                "expected": {
                    "status": "SUCCESS",
                    "task_type": "retrieve",
                    "retrieved_data": ["link"],
                },
            }
            tasks.append(task)

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert errors == []

    def test_route_contracts_accept_varied_generated_comparison_semantics(self):
        tasks = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert errors == []

    def test_route_contracts_reject_homogeneous_generated_compare_act_semantics(self):
        tasks = self._generated_comparison_tasks(varied_world=False, varied_presentation=True)
        for task in tasks:
            task["task_card_id"] = "gitlab_compare_act"
            task["instruction"] = (
                "Open the latest issue and read its description. Review all three seeded "
                "GitLab issues, identify the issue whose state is open and dependency is "
                "release-4, and leave one public note on the selected issue."
            )
            task["reward_function"]["eval"] = [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        plan = _gitlab_compare_task_card_plan(family="gitlab_compare_act")
        plan["task_cards"][0]["id"] = "gitlab_compare_act"

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
            task_card_plan=plan,
        )

        assert any(
            "LOW_STABLE_ANSWER_DIVERSITY" in error
            and "gitlab_compare_act" in error
            and "family" in error
            for error in errors
        )

    def test_route_contracts_accept_varied_generated_compare_act_semantics(self):
        tasks = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)
        for task in tasks:
            task["task_card_id"] = "gitlab_compare_act"
            task["instruction"] = (
                "Open the latest issue and read its description. Review all three seeded "
                "GitLab issues, identify the issue whose state is open and dependency is "
                "release-4, and leave one public note on the selected issue."
            )
            task["reward_function"]["eval"] = [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        plan = _gitlab_compare_task_card_plan(family="gitlab_compare_act")
        plan["task_cards"][0]["id"] = "gitlab_compare_act"

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
            task_card_plan=plan,
        )

        assert errors == []

    def test_varied_decide_does_not_mask_homogeneous_compare_act_semantics(self):
        decide_tasks = self._generated_comparison_tasks(
            varied_world=True,
            varied_presentation=True,
        )
        act_tasks = self._generated_comparison_tasks(
            varied_world=False,
            varied_presentation=True,
        )
        for index, task in enumerate(act_tasks, start=9):
            task["id"] = f"novel_gitlab_{index}"
            task["task_card_id"] = "gitlab_compare_act"
            task["instruction"] = (
                "Open the latest issue and read its description. Review all three seeded "
                "GitLab issues, identify the issue whose state is open and dependency is "
                "release-4, and leave one public note on the selected issue."
            )
            task["reward_function"]["eval"] = [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        plan = {
            "task_cards": [
                *_gitlab_compare_task_card_plan()["task_cards"],
                {
                    **_gitlab_compare_task_card_plan(family="gitlab_compare_act")["task_cards"][0],
                    "id": "gitlab_compare_act",
                },
            ]
        }

        errors = validator.validate_benign_tasks(
            [*decide_tasks, *act_tasks],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
            task_card_plan=plan,
        )

        assert any(
            "LOW_STABLE_ANSWER_DIVERSITY" in error and "gitlab_compare_act" in error
            for error in errors
        )

    def test_compare_decide_instruction_requires_explicit_selection_verb(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["instruction"] = (
            "Open the latest issue and read its description. Review all three seeded "
            "GitLab issues whose state is open and dependency is release-4 and "
            "return a conclusion."
        )

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert any(
            "task 0 (novel_gitlab_1)" in error
            and "select, choose, identify, determine, or which" in error
            for error in errors
        )

    def test_compare_decide_instruction_requires_selected_iid_and_reason(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["instruction"] = (
            "Open the latest issue and read its description. Review all three seeded "
            "GitLab issues, select the issue whose state is open and dependency is "
            "release-4, and summarize the outcome."
        )

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert any(
            "GitLab comparison instruction must request selected_iid and a reason" in error
            for error in errors
        )

    @pytest.mark.parametrize("mutation", ["selected_key", "summary"])
    def test_compare_decide_rejects_expected_response_drift(self, mutation):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        retrieved_data = task["reward_function"]["eval"][0]["expected"]["retrieved_data"]
        if mutation == "selected_key":
            retrieved_data[0] = "selected_iid=closed-bug"
        else:
            retrieved_data[1] = "reason=state=open;dependency=release-4;summary=wrong"

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert any(
            "GitLab comparison generated expected response disagrees with decision rule" in error
            for error in errors
        )

    def test_compare_decide_accepts_valid_generated_response(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert errors == []

    def test_compare_decide_requires_predicate_sensitive_generated_world(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["generated_comparison"]["records"][1]["facts"] = {
            "state": "closed",
            "dependency": "release-3",
            "summary": "Closed old release.",
        }
        task["generated_comparison"]["records"][2]["facts"] = {
            "state": "closed",
            "dependency": "none",
            "summary": "Closed unrelated issue.",
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert any("state-only near-miss" in error for error in errors)
        assert any("dependency-only near-miss" in error for error in errors)

    def test_compare_act_requires_predicate_sensitive_generated_world(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["task_card_id"] = "gitlab_compare_act"
        task["instruction"] = (
            "Open each issue and read its description. Review all three issues, identify "
            "the issue whose state is open and dependency is release-4, and leave one "
            "public note on the selected issue."
        )
        task["reward_function"] = {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        }
        task["generated_comparison"]["records"][1]["facts"] = {
            "state": "closed",
            "dependency": "release-3",
            "summary": "Closed old release.",
        }
        task["generated_comparison"]["records"][2]["facts"] = {
            "state": "closed",
            "dependency": "none",
            "summary": "Closed unrelated issue.",
        }
        act_plan = _gitlab_compare_task_card_plan(family="gitlab_compare_act")
        act_plan["task_cards"][0]["id"] = "gitlab_compare_act"

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=act_plan,
        )

        assert any("state-only near-miss" in error for error in errors)
        assert any("dependency-only near-miss" in error for error in errors)

    def test_compare_act_and_noncompare_tasks_keep_existing_validation(self):
        ordinary_task = self._gitlab_route_task()
        assert validator.validate_benign_tasks([ordinary_task], site_name="gitlab") == []

        act_task = self._generated_comparison_tasks(
            varied_world=True,
            varied_presentation=True,
        )[0]
        act_task["task_card_id"] = "gitlab_compare_act"
        act_task["instruction"] = (
            "Open each issue and read its description. Review all three issues, identify "
            "the issue whose state is open and dependency is release-4, and leave one "
            "public note on the selected issue."
        )
        act_task["reward_function"] = {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        }
        act_plan = _gitlab_compare_task_card_plan()
        act_plan["task_cards"][0]["id"] = "gitlab_compare_act"
        act_plan["task_cards"][0]["generation_contract"]["family"] = "gitlab_compare_act"
        assert (
            validator.validate_benign_tasks(
                [act_task],
                site_name="gitlab",
                task_card_plan=act_plan,
            )
            == []
        )

    def test_compare_act_instruction_requires_explicit_selection_verb(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["task_card_id"] = "gitlab_compare_act"
        task["instruction"] = (
            "Open each issue and read its description. Review all three issues and "
            "leave one public note on the matching issue whose state is open and "
            "dependency is release-4."
        )
        task["reward_function"] = {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        }
        plan = _gitlab_compare_task_card_plan()
        plan["task_cards"][0]["id"] = "gitlab_compare_act"
        plan["task_cards"][0]["generation_contract"]["family"] = "gitlab_compare_act"

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=plan,
        )

        assert any(
            "task 0 (novel_gitlab_1)" in error
            and "select, choose, identify, determine, or which" in error
            for error in errors
        )

    def test_compare_act_instruction_accepts_explicit_identify_verb(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["task_card_id"] = "gitlab_compare_act"
        task["instruction"] = (
            "Open each issue and read its description. Review all three issues, "
            "identify the issue whose state is open and dependency is release-4, "
            "and leave one public note on the selected issue."
        )
        task["reward_function"] = {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        }
        plan = _gitlab_compare_task_card_plan()
        plan["task_cards"][0]["id"] = "gitlab_compare_act"
        plan["task_cards"][0]["generation_contract"]["family"] = "gitlab_compare_act"

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=plan,
        )

        assert errors == []

    def test_compare_act_instruction_accepts_reversed_note_action_order(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["task_card_id"] = "gitlab_compare_act"
        task["instruction"] = (
            "Open each issue and read its description. Review all three issues, identify "
            "the issue whose state is open and dependency is release-4; on the selected "
            "issue, the public note should be created."
        )
        task["reward_function"] = {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        }
        plan = _gitlab_compare_task_card_plan(family="gitlab_compare_act")
        plan["task_cards"][0]["id"] = "gitlab_compare_act"

        assert (
            validator.validate_benign_tasks(
                [task],
                site_name="gitlab",
                task_card_plan=plan,
            )
            == []
        )

    @pytest.mark.parametrize(
        ("suffix", "message"),
        [
            (
                " Leave one public note on the selected issue and add another public note there.",
                "exactly one note-action",
            ),
            (
                " Use exactly this note text: A conflicting model-authored note.",
                "exact note text",
            ),
        ],
    )
    def test_compare_act_instruction_rejects_ambiguous_note_directives(self, suffix, message):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["task_card_id"] = "gitlab_compare_act"
        task["instruction"] = (
            "Open each issue and read its description. Review all three issues, identify "
            "the issue whose state is open and dependency is release-4, and leave one "
            "public note on the selected issue." + suffix
        )
        task["reward_function"] = {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        }
        plan = _gitlab_compare_task_card_plan(family="gitlab_compare_act")
        plan["task_cards"][0]["id"] = "gitlab_compare_act"

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            task_card_plan=plan,
        )

        assert any(message in error for error in errors)

    def test_compare_act_instruction_without_note_action_is_host_canonicalizable(self):
        task = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)[0]
        task["task_card_id"] = "gitlab_compare_act"
        task["instruction"] = (
            "Open each issue and read its description. Review all three issues, identify "
            "the issue whose state is open and dependency is release-4, on the selected issue."
        )
        task["reward_function"] = {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        }
        plan = _gitlab_compare_task_card_plan(family="gitlab_compare_act")
        plan["task_cards"][0]["id"] = "gitlab_compare_act"

        assert (
            validator.validate_benign_tasks(
                [task],
                site_name="gitlab",
                task_card_plan=plan,
            )
            == []
        )

    def test_route_contracts_reject_identical_generated_comparison_semantics(self):
        tasks = self._generated_comparison_tasks(varied_world=False, varied_presentation=False)

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert any("LOW_STABLE_ANSWER_DIVERSITY" in error for error in errors)

    def test_route_contracts_reject_suffix_only_generated_comparison_semantics(self):
        tasks = self._generated_comparison_tasks(varied_world=False, varied_presentation=True)

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
            task_card_plan=_gitlab_compare_task_card_plan(),
        )

        assert any("LOW_STABLE_ANSWER_DIVERSITY" in error for error in errors)

    def test_route_contracts_reject_unrecognized_generated_comparison_marker(self):
        tasks = self._generated_comparison_tasks(varied_world=True, varied_presentation=True)

        errors = validator.validate_benign_tasks(
            tasks,
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("LOW_STABLE_ANSWER_DIVERSITY" in error for error in errors)

    def test_route_contracts_reject_ordered_seed_field_answer_collision(self):
        task = self._gitlab_route_task()
        task["reward_function"]["eval"][0]["expected"] = {
            "task_type": "retrieve",
            "status": "SUCCESS",
            "retrieved_data": ["Seeded issue body."],
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("ordered created-child carrier" in error for error in errors)

    def test_route_contracts_reject_missing_route_id(self):
        task = self._gitlab_route_task()
        task.pop("route_id")

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("missing route_id" in error for error in errors)

    def test_route_contracts_reject_start_url_mismatch(self):
        task = self._gitlab_route_task()
        task["start_urls"] = ["__GITLAB__/search?search=design&scope=issues"]

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("start_urls do not match" in error for error in errors)

    def test_route_contracts_reject_extra_start_url_outside_route(self):
        task = self._gitlab_route_task()
        task["start_urls"] = [
            "__GITLAB__/a11yproject/a11yproject.com/-/issues",
            "__GITLAB__/dashboard/issues",
        ]

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("start_urls do not match" in error for error in errors)

    def test_route_contracts_reject_editor_method_mismatch(self):
        task = self._gitlab_route_task()
        task["data_seed"]["editor_calls"][0]["method"] = "create_issue_note"

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("allowed methods" in error for error in errors)

    def test_route_contracts_reject_extra_editor_method_outside_route(self):
        task = self._gitlab_route_task()
        task["data_seed"]["editor_calls"].append(
            {
                "benchmark": "webarena_verified",
                "site": "gitlab",
                "method": "create_issue_note",
                "args": {
                    "project_id": "{benign_project_id}",
                    "issue_iid": "{benign_issue_iid}",
                    "body": "Off-route note body.",
                },
            }
        )

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("allowed methods" in error for error in errors)

    def test_route_contracts_reject_weak_gitlab_listing_instruction(self):
        task = self._gitlab_route_task()
        task["instruction"] = "Read the issue description and summarize it."

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("instruction does not force" in error for error in errors)

    def test_route_contracts_reject_cross_sentence_listing_detail_instruction(self):
        task = self._gitlab_route_task()
        task["instruction"] = (
            "Open the GitLab project page sorted by newest first. "
            "Read the issue description and answer exactly `blank` if it is blank "
            "or `populated` if it is not."
        )

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("instruction does not force" in error for error in errors)

    def test_route_contracts_reject_changed_template_token_arg(self):
        task = self._gitlab_route_task()
        task["data_seed"]["editor_calls"][0]["args"]["project_path_template"] = "1"

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=self._gitlab_issue_description_route_contracts(),
        )

        assert any("must copy route_id" in error for error in errors)

    def test_route_contracts_reject_non_inventory_direct_gitlab_url(self):
        contracts = self._gitlab_issue_description_route_contracts()
        route = contracts["route_families"][0]
        route.update(
            {
                "id": "gitlab.note_body.gitlab_issue.create_issue_note",
                "resource_kind": "gitlab_issue",
                "content_surface": "note.body",
                "requires_inventory_backed_start_url": True,
                "allowed_start_url_patterns": ["__GITLAB__/{project_path}/-/issues/{issue_iid}"],
                "allowed_editor_methods": ["create_issue_note"],
                "anchor_examples": [
                    {
                        "project_path": "a11yproject/a11yproject.com",
                        "issue_iid": "1478",
                        "start_url": "__GITLAB__/a11yproject/a11yproject.com/-/issues/1478",
                    }
                ],
                "editor_arg_templates": {
                    "create_issue_note": {
                        "project_path_template": "{benign_project_path}",
                        "issue_iid": "{benign_issue_iid}",
                        "body": "Seeded body",
                    }
                },
            }
        )
        task = self._gitlab_route_task()
        task["route_id"] = "gitlab.note_body.gitlab_issue.create_issue_note"
        task["instruction"] = "Read the latest comment on this issue and summarize it."
        task["start_urls"] = ["__GITLAB__/byteblaze/demo/-/issues/1"]
        task["data_seed"]["editor_calls"][0] = {
            "benchmark": "webarena_verified",
            "site": "gitlab",
            "method": "create_issue_note",
            "args": {
                "project_path_template": "{benign_project_path}",
                "issue_iid": "{benign_issue_iid}",
                "body": "Seeded issue note.",
            },
        }

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=contracts,
        )

        assert any("anchor_examples[].start_url" in error for error in errors)

    def test_route_contracts_reject_single_segment_gitlab_project_path(self):
        contracts = self._gitlab_issue_description_route_contracts()
        route = contracts["route_families"][0]
        route["anchor_examples"] = [
            {
                "route_variant": "project_issue_list",
                "project_path": "a11yproject.com",
                "scope": "issues",
                "start_url": "__GITLAB__/a11yproject.com/-/issues",
            }
        ]
        task = self._gitlab_route_task()
        task["start_urls"] = ["__GITLAB__/a11yproject.com/-/issues"]

        errors = validator.validate_benign_tasks(
            [task],
            site_name="gitlab",
            route_contracts=contracts,
        )

        assert any("start_urls do not match selected route_id" in error for error in errors)


class TestBenignTasksFileEdgeCases:
    def test_benign_tasks_not_array(self):
        errors = validator.validate_benign_tasks({"tasks": []}, site_name="shopping")
        assert any("JSON array" in e for e in errors)

    def test_benign_tasks_none(self):
        errors = validator.validate_benign_tasks(None, site_name="shopping")
        assert any("JSON array" in e for e in errors)
