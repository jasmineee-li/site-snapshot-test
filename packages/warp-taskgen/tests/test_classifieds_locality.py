from __future__ import annotations

from pathlib import Path

from warp_taskgen.phase_2.phase_2c.policy import default_feasibility_policy_catalog
from warp_taskgen.seeding.site_contracts import default_seed_registry
from warp_taskgen.sites import SiteCatalog

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_classifieds_remains_out_of_production_defaults() -> None:
    assert SiteCatalog().sites == ("gitlab", "reddit")
    assert ("visualwebarena", "classifieds") not in default_seed_registry().registrations
    assert default_feasibility_policy_catalog().get("visualwebarena", "classifieds") is None


def test_generic_phase_modules_do_not_branch_on_classifieds() -> None:
    generic_roots = (
        PACKAGE_ROOT / "warp_taskgen" / "phases",
        PACKAGE_ROOT / "warp_taskgen" / "phase_4",
    )
    offenders: list[str] = []
    for root in generic_roots:
        for path in root.rglob("*.py"):
            if "classifieds" in path.read_text(encoding="utf-8").casefold():
                offenders.append(str(path.relative_to(PACKAGE_ROOT)))

    assert offenders == []
