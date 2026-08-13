from __future__ import annotations

import importlib
import importlib.util

from scripts import readiness_audit

RETIRED_FEATURE_MODULES = frozenset(
    {
        "worldsim.phases.phase_2_injections_api",
        "worldsim.phases.phase_2_text_fill",
        "worldsim.phases.phase_2_exposure_contract",
        "worldsim.phases.phase_1_generate_new_tasks_validation",
        "worldsim.phases.phase_2_feasibility",
    }
)


def test_retired_feature_facades_are_absent() -> None:
    importlib.invalidate_caches()

    for module_name in RETIRED_FEATURE_MODULES:
        assert module_name in readiness_audit.LEGACY_PHASE_IMPORT_MODULES
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        assert spec is None


def test_canonical_feature_modules_own_the_retired_surfaces() -> None:
    canonical_modules = {
        "warp_taskgen.phase_2.runner_api": "runner_api.py",
        "warp_taskgen.phase_2.text_fill": "text_fill/__init__.py",
        "warp_taskgen.phase_2.exposure_contract": "exposure_contract/__init__.py",
        "warp_taskgen.phase_1.novel_task_validation": "novel_task_validation/__init__.py",
        "warp_taskgen.phase_2.phase_2c": "phase_2c/__init__.py",
    }

    for module_name, suffix in canonical_modules.items():
        spec = importlib.util.find_spec(module_name)
        assert spec is not None
        assert spec.origin is not None
        assert spec.origin.endswith(suffix)


def test_tracked_source_has_no_retired_feature_imports() -> None:
    audit = readiness_audit.build_audit()

    assert audit.legacy_phase_imports == []
    assert audit.active_facade_imports == []
