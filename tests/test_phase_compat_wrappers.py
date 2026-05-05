from __future__ import annotations

import importlib
import importlib.util

from scripts import readiness_audit


def test_legacy_phase_compatibility_wrappers_are_removed() -> None:
    importlib.invalidate_caches()

    for legacy_module in readiness_audit.LEGACY_PHASE_IMPORT_MODULES:
        assert importlib.util.find_spec(legacy_module) is None


def test_tracked_source_has_no_legacy_phase_imports() -> None:
    assert readiness_audit.build_audit().legacy_phase_imports == []
