from __future__ import annotations

import importlib
import importlib.util
import sys

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
