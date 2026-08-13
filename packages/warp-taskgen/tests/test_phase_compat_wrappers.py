from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

from scripts import readiness_audit


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
