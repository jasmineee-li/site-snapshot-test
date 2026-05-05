from __future__ import annotations

import importlib


def test_legacy_phase_runner_imports_delegate_to_canonical_modules() -> None:
    cases = [
        ("worldsim.phases.phase_2_injections", "worldsim.phase_2.runner", "run"),
        (
            "worldsim.phases.phase_2_target_resolver",
            "worldsim.phase_2.target_resolution.runner",
            "resolve_tasks",
        ),
        ("worldsim.phases.phase_4_adversarial", "worldsim.phase_4.runner", "run"),
    ]

    for legacy_name, canonical_name, delegated_name in cases:
        canonical = importlib.import_module(canonical_name)
        legacy = importlib.import_module(legacy_name)

        assert legacy is canonical
        assert getattr(legacy, delegated_name) is getattr(canonical, delegated_name)


def test_legacy_phase_helper_imports_delegate_to_canonical_functions() -> None:
    # `phase_2_injections_api` remains the canonical Shape-C streaming L3
    # implementation on feat/worldsim-v5; the PR #11 rename to
    # `worldsim.phase_2.runner_api` was deferred to a later migration cycle,
    # so only the importability of the legacy module is asserted here.
    legacy_runner_api = importlib.import_module("worldsim.phases.phase_2_injections_api")
    assert hasattr(legacy_runner_api, "generate_phase_2a_plans_api")

    output = importlib.import_module("worldsim.phase_2.output")
    legacy_output = importlib.import_module("worldsim.phases.phase_2_output")
    artifacts = importlib.import_module("worldsim.phase_2.phase_2c.artifacts")
    legacy_artifacts = importlib.import_module("worldsim.phases.phase_2c_artifacts")
    config = importlib.import_module("worldsim.phase_2.phase_2c.config")
    legacy_config = importlib.import_module("worldsim.phases.phase_2c_config")

    assert legacy_output._sanitize_task_for_output is output._sanitize_task_for_output
    assert legacy_artifacts._write_phase_2c_artifacts is artifacts._write_phase_2c_artifacts
    assert legacy_config._extract_instances_list is config._extract_instances_list


def test_legacy_target_resolver_private_helpers_remain_reexported() -> None:
    legacy = importlib.import_module("worldsim.phases.phase_2_target_resolver")

    helper_modules = {
        "_literalize_regex_value": "worldsim.phase_2.target_resolution.url_matching",
        "_project_item_to_record": "worldsim.phase_2.target_resolution.reconstruction",
        "_reconstruct_start_url_from_anchors": "worldsim.phase_2.target_resolution.reconstruction",
        "_canonicalize_project_path": "worldsim.phase_2.target_resolution.url_matching",
        "_normalise_sort_direction": "worldsim.phase_2.target_resolution.http_probes",
        "_default_listing_probe": "worldsim.phase_2.target_resolution.listing_probes",
        "_gitlab_issue_listing_intent": "worldsim.phase_2.target_resolution.listing_intent",
        "_route_evidence_flags": "worldsim.phase_2.target_resolution.encounter",
    }

    for helper_name, canonical_module_name in helper_modules.items():
        canonical_module = importlib.import_module(canonical_module_name)
        assert getattr(legacy, helper_name) is getattr(canonical_module, helper_name)
