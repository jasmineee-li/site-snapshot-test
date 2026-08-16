"""Proofs for the diagnostic composition package boundary.

These tests deliberately stay above the implementation details of the
composition validator.  They protect the two boundaries a future Site
definition must not cross: generic Phase code must remain Site-neutral, and
the diagnostic module must be importable from the package rather than relying
on a repository checkout or process-wide Site registration.
"""

from __future__ import annotations

import importlib
import importlib.resources
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from warp_taskgen.site_composition import (
    ActiveSitePolicy,
    CapabilityReference,
    OperationalEvidence,
    SiteBenchmarkBinding,
    SiteDefinition,
    SiteDoctorRequest,
    compile_site_definitions,
    default_site_definitions,
)
from warp_taskgen.sites import SiteCatalog

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "warp_taskgen"

# ``proof_forum`` is a test-only identity used by the composition conformance
# fixture.  Keep the check narrow: ``forum`` is already a legitimate Reddit
# domain noun in existing generic compatibility code.
TEST_SITE_NOUNS = ("proof_forum", "proof_forum.thread_reply")
GENERIC_FEATURES = (
    "phase_1",
    "phase_2",
    "phase_2c",
    "phase_4",
    "rewards",
    "seeding",
)


def _generic_production_sources() -> list[Path]:
    return [path for feature in GENERIC_FEATURES for path in (SOURCE_ROOT / feature).rglob("*.py")]


def test_test_only_site_nouns_stay_out_of_generic_phase_code() -> None:
    """A test Site is local to its definition and never a generic branch."""

    offenders = {
        path.relative_to(SOURCE_ROOT).as_posix(): noun
        for path in _generic_production_sources()
        for noun in TEST_SITE_NOUNS
        if noun in path.read_text(encoding="utf-8")
    }

    assert offenders == {}, (
        f"test-only Site nouns leaked into generic Phase/reward/seeding code: {offenders!r}"
    )


def test_composition_import_does_not_mutate_the_runtime_site_catalog() -> None:
    """Diagnostic registration is not production Site activation."""

    from warp_taskgen.editors import EDITOR_REGISTRY

    before = SiteCatalog().sites
    editor_before = dict(EDITOR_REGISTRY)
    importlib.import_module("warp_taskgen.site_composition")
    default_site_definitions()
    assert SiteCatalog().sites == before
    assert dict(EDITOR_REGISTRY) == editor_before


def test_removed_definition_fails_closed_without_a_stale_edge() -> None:
    """Deleting a definition changes only its explicit composition result."""

    request = SiteDoctorRequest(
        site="proof_forum",
        benchmark="webarena_verified",
        use_case="ugc_reply",
    )
    policy = ActiveSitePolicy()
    evidence = OperationalEvidence()
    absent = CapabilityReference("missing", None, ("test.definition",))
    registered = compile_site_definitions(
        (
            SiteDefinition(
                site="proof_forum",
                bindings=(
                    SiteBenchmarkBinding(
                        benchmark="webarena_verified",
                        targeting=absent,
                        profile=absent,
                        editor_specs=absent,
                        seed=absent,
                        feasibility=absent,
                        read_surface=absent,
                        readback=absent,
                        final_state=absent,
                        action_cards=absent,
                    ),
                ),
                provenance=("test.definition",),
            ),
        ),
        request,
        active_policy=policy,
        operational_evidence=evidence,
    )
    removed = compile_site_definitions(
        (),
        request,
        active_policy=policy,
        operational_evidence=evidence,
    )

    assert registered.static_status == "incomplete"
    assert removed.static_status == "invalid"
    assert registered.definition_digest != removed.definition_digest
    assert "test.definition" not in removed.to_json()
    assert all("test.definition" not in str(finding.detail) for finding in removed.findings)


def test_removing_unrelated_definition_preserves_active_site_diagnostics() -> None:
    """Removing one Site cannot change another Site's diagnostic report."""

    definitions = tuple(default_site_definitions())
    gitlab_definitions = tuple(
        definition for definition in definitions if definition.site == "gitlab"
    )
    request = SiteDoctorRequest(
        site="gitlab",
        benchmark="webarena_verified",
        use_case="phase_2_feasibility",
    )
    policy = ActiveSitePolicy()
    evidence = OperationalEvidence()
    complete_catalog_report = compile_site_definitions(
        definitions,
        request,
        active_policy=policy,
        operational_evidence=evidence,
    )
    gitlab_only_report = compile_site_definitions(
        gitlab_definitions,
        request,
        active_policy=policy,
        operational_evidence=evidence,
    )

    assert gitlab_definitions
    assert complete_catalog_report.static_status == gitlab_only_report.static_status
    assert complete_catalog_report.status == gitlab_only_report.status
    assert complete_catalog_report.definition_digest == gitlab_only_report.definition_digest
    assert [
        (finding.capability, finding.state, finding.detail)
        for finding in complete_catalog_report.findings
    ] == [
        (finding.capability, finding.state, finding.detail)
        for finding in gitlab_only_report.findings
    ]


def test_composition_module_is_in_the_canonical_distribution() -> None:
    """The wheel/sdist package declaration includes the diagnostic module."""

    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel_packages = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    sdist_include = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert wheel_packages == ["warp_taskgen"]
    assert "/warp_taskgen" in sdist_include
    assert (SOURCE_ROOT / "site_composition.py").is_file()


def test_composition_import_works_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    """Import and resource lookup do not depend on the checkout CWD."""

    code = """
import importlib.resources
import warp_taskgen.site_composition

resource = importlib.resources.files("warp_taskgen").joinpath("site_composition.py")
assert resource.is_file()
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(PACKAGE_ROOT),
        "PYTHON_DOTENV_DISABLED": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_composition_import_does_not_bootstrap_editor_registrations(tmp_path: Path) -> None:
    """Importing the static compiler alone has no editor-package side effect."""

    code = """
import sys

assert "warp_taskgen.editors" not in sys.modules
import warp_taskgen.site_composition
assert "warp_taskgen.editors" not in sys.modules
assert not any(name.startswith("warp_taskgen.editors.") for name in sys.modules)
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(PACKAGE_ROOT),
        "PYTHON_DOTENV_DISABLED": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
