from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

from scripts import readiness_audit
from worldsim.benchmark_capabilities import get_benchmark_capabilities


def test_legacy_phase_compatibility_wrappers_are_removed() -> None:
    importlib.invalidate_caches()

    retired_modules = readiness_audit.LEGACY_PHASE_IMPORT_MODULES - {
        "worldsim.phases.phase_2_injections_api"
    }
    for legacy_module in retired_modules:
        assert importlib.util.find_spec(legacy_module) is None


def test_phase_2_api_compatibility_import_delegates_to_canonical_module() -> None:
    canonical = importlib.import_module("worldsim.phase_2.runner_api")
    historical = importlib.import_module("worldsim.phases.phase_2_injections_api")

    assert historical is canonical
    assert sys.modules["worldsim.phases.phase_2_injections_api"] is canonical
    assert historical.generate_phase_2a_plans_api is canonical.generate_phase_2a_plans_api
    assert historical._EMIT_PLANS_TOOL is canonical._EMIT_PLANS_TOOL
    assert historical._build_messages is canonical._build_messages


def test_phase_1_validation_compatibility_import_delegates_to_canonical_modules() -> None:
    """The historical Phase 1 facade re-exports the feature-owned validators."""

    historical = importlib.import_module("worldsim.phases.phase_1_generate_new_tasks_validation")
    canonical = importlib.import_module("worldsim.phase_1.novel_task_validation")
    task_cards = importlib.import_module("worldsim.phase_1.novel_task_validation.task_cards")

    assert historical.validate_generated_novel_task is canonical.validate_generated_novel_task
    assert historical.validate_generated_novel_tasks is canonical.validate_generated_novel_tasks
    assert (
        historical.validate_generated_novel_tasks_detailed
        is canonical.validate_generated_novel_tasks_detailed
    )
    assert historical.sort_novel_tasks is canonical.sort_novel_tasks
    assert historical.merge_benign_tasks is canonical.merge_benign_tasks
    assert historical._validate_task_card_alignment is task_cards._validate_task_card_alignment


@pytest.mark.asyncio
async def test_text_fill_compat_facade_forwards_legacy_patch_points(monkeypatch) -> None:
    """The old facade remains patchable while callers move to feature modules."""

    legacy = importlib.import_module("worldsim.phases.phase_2_text_fill")
    api = importlib.import_module("worldsim.phase_2.text_fill.api")
    service = importlib.import_module("worldsim.phase_2.text_fill.service")

    sentinel_client = object()
    sentinel_instructor = object()
    with monkeypatch.context() as patch:
        # Register canonical attrs with the context before the facade mutates
        # them directly. Its teardown then restores the originals even when
        # an assertion or fake raises.
        patch.setattr(api, "get_client", api.get_client)
        patch.setattr(api, "instructor", api.instructor)
        patch.setattr(legacy, "get_client", lambda: sentinel_client)
        patch.setattr(legacy, "instructor", sentinel_instructor)

        api_seen: dict[str, object] = {}

        async def fake_api(*args, **kwargs):
            del args, kwargs
            api_seen["get_client"] = api.get_client()
            api_seen["instructor"] = api.instructor
            return ("payload", "compat")

        patch.setattr(api, "_call_text_fill_api", fake_api)
        assert await legacy._call_text_fill_api("prompt", "model") == ("payload", "compat")
        assert api_seen == {"get_client": sentinel_client, "instructor": sentinel_instructor}

    sentinel_render = object()
    sentinel_call = object()
    with monkeypatch.context() as patch:
        patch.setattr(service, "render_fill_prompt", service.render_fill_prompt)
        patch.setattr(service, "_call_text_fill_api", service._call_text_fill_api)
        patch.setattr(legacy, "render_fill_prompt", sentinel_render)
        patch.setattr(legacy, "_call_text_fill_api", sentinel_call)
        service_seen: dict[str, object] = {}

        async def fake_generate(*args, **kwargs):
            del args, kwargs
            service_seen["render_fill_prompt"] = service.render_fill_prompt
            service_seen["_call_text_fill_api"] = service._call_text_fill_api
            return (None, {"status": "compat"})

        patch.setattr(service, "_generate_single_payload", fake_generate)
        assert await legacy._generate_single_payload("task") == (None, {"status": "compat"})
        assert service_seen == {
            "render_fill_prompt": sentinel_render,
            "_call_text_fill_api": sentinel_call,
        }

    sentinel_generate = object()
    with monkeypatch.context() as patch:
        patch.setattr(service, "_generate_single_payload", service._generate_single_payload)
        patch.setattr(legacy, "_generate_single_payload", sentinel_generate)
        fill_seen: dict[str, object] = {}

        async def fake_fill(*args, **kwargs):
            del args, kwargs
            fill_seen["_generate_single_payload"] = service._generate_single_payload
            return (None, {"status": "compat"})

        patch.setattr(service, "_fill_one_task", fake_fill)
        assert await legacy._fill_one_task("task") == (None, {"status": "compat"})
        assert fill_seen == {"_generate_single_payload": sentinel_generate}


def test_phase_2c_compat_facade_forwards_canonical_public_surface() -> None:
    """The historical Phase 2c module remains a forwarding facade only."""

    historical = importlib.import_module("worldsim.phases.phase_2_feasibility")
    canonical = importlib.import_module("worldsim.phase_2.phase_2c")
    implementation = importlib.import_module("worldsim.phase_2.phase_2c._impl")
    assert historical._legacy_impl is implementation
    assert historical.FeasibilityReport is canonical.FeasibilityReport
    assert historical.verify_feasibility is not implementation.verify_feasibility
    assert historical._legacy_impl.verify_feasibility is canonical.verify_feasibility
    assert historical.get_benchmark_capabilities is get_benchmark_capabilities


@pytest.mark.asyncio
async def test_phase_2c_compat_probe_patch_isolated_by_test_context(monkeypatch) -> None:
    """Legacy probe patches forward while scoped canonical destinations are restored."""

    historical = importlib.import_module("worldsim.phases.phase_2_feasibility")
    probes = importlib.import_module("worldsim.phase_2.phase_2c.probes")
    implementation = importlib.import_module("worldsim.phase_2.phase_2c._impl")
    original_verify_seed_renders = probes.verify_seed_renders
    original_impl_verify_seed_renders = implementation.verify_seed_renders
    original_impl_logger = implementation.logger

    async def failing_verify_seed_renders(**_kwargs):
        raise RuntimeError("compatibility probe failure")

    with monkeypatch.context() as patch:
        # Register the canonical destination with the context so the test
        # itself is exception-safe even while the facade synchronizes it.
        patch.setattr(probes, "verify_seed_renders", original_verify_seed_renders)
        patch.setattr(implementation, "verify_seed_renders", original_impl_verify_seed_renders)
        patch.setattr(implementation, "logger", original_impl_logger)
        patch.setattr(historical, "verify_seed_renders", failing_verify_seed_renders)

        outcome = await historical._run_render_check(
            browser=object(),
            render_semaphore=None,
            seed={
                "editor_calls": [
                    {
                        "site": "gitlab",
                        "method": "create_issue_note",
                        "args": {"note_body": "compatibility probe body"},
                    }
                ]
            },
            metadata={
                "read_surface_urls": [
                    "https://gitlab.example/project/-/issues/1",
                ]
            },
            instance={
                "site_name": "gitlab",
                "site_url": "https://gitlab.example",
                "benchmark": "webarena_verified",
            },
        )

        assert not outcome.ok
        assert "compatibility probe failure" in outcome.detail

    assert probes.verify_seed_renders is original_verify_seed_renders
    assert implementation.verify_seed_renders is original_impl_verify_seed_renders
    assert implementation.logger is original_impl_logger


def test_exposure_contract_compat_facade_forwards_legacy_patch_points(monkeypatch) -> None:
    """The exposure facade keeps old monkeypatch seams during migration."""

    legacy = importlib.import_module("worldsim.phases.phase_2_exposure_contract")
    impl = importlib.import_module("worldsim.phase_2.exposure_contract._impl")
    with monkeypatch.context() as patch:
        # The facade synchronizes its legacy patch into the implementation;
        # register that destination with the context so teardown restores it.
        patch.setattr(impl, "iter_specs", impl.iter_specs)
        patch.setattr(legacy, "iter_specs", lambda *, site, benchmark: [])

        contract = legacy.build_exposure_contract(
            benign_task_id="compat-exposure",
            site="gitlab",
            benchmark="webarena_verified",
            benign_target_resource={
                "kind": "gitlab_issue",
                "anchors": {"project_id": "22", "issue_iid": "7"},
                "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
            },
        )

    assert contract["seed_capability"] == {
        "status": "unsupported",
        "reason": "no_viable_editor_method_under_anchors",
    }
    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "no_viable_editor_method_under_anchors",
    }


def test_exposure_contract_compat_facade_forwards_owner_helpers(monkeypatch) -> None:
    """Legacy builder and seed calls still honor patched owner helpers."""

    legacy = importlib.import_module("worldsim.phases.phase_2_exposure_contract")
    impl = importlib.import_module("worldsim.phase_2.exposure_contract._impl")
    helper_names = (
        "iter_specs",
        "available_tokens_for_kind",
        "canonical_core_surface",
        "is_active_carrier_surface",
        "is_core_surface",
        "active_carrier_ineligible_reason",
    )
    calls = {name: 0 for name in helper_names}

    with monkeypatch.context() as patch:
        for name in helper_names:
            patch.setattr(impl, name, getattr(impl, name))
            original = getattr(legacy, name)

            def spy(*args, _name=name, _original=original, **kwargs):
                calls[_name] += 1
                return _original(*args, **kwargs)

            patch.setattr(legacy, name, spy)

        contract = legacy.build_exposure_contract(
            benign_task_id="compat-exposure-owner-helpers",
            site="gitlab",
            benchmark="webarena_verified",
            benign_target_resource={
                "kind": "gitlab_issue",
                "anchors": {"project_id": "22", "issue_iid": "7"},
                "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
                "exact_comment_region_forced_by_task": True,
            },
        )
        seed = legacy.materialize_seed_template_from_contract(contract)
        assert (
            legacy.active_carrier_ineligible_reason("gitlab", "issue.title")
            == "retired_title_carrier_surface"
        )

    assert contract["eligibility"] == {"status": "eligible"}
    assert seed["editor_calls"][0]["args"]["body"] == "{{PAYLOAD_TEXT}}"
    assert all(calls.values())
