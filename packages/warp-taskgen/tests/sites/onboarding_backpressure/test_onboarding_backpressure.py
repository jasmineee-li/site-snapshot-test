"""Focused backpressure for the canonical Site-onboarding slice.

These tests deliberately stay at public seams.  The source scans are the two
architectural exceptions: they keep removed Python names and the synthetic
Site noun from leaking into generic production modules.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PACKAGE_ROOT / "warp_taskgen"

REMOVED_PYTHON_NAMES = (
    "CapabilityReference",
    "OperationalEvidence",
    "OperationalState",
    "SiteBenchmarkBinding",
    "SiteDefinition",
    "SiteDoctorReport",
    "SiteDoctorRequest",
    "ActiveSitePolicy",
)
GENERIC_SOURCE_ROOTS = (
    "phase_1",
    "phase_2",
    "phase_2c",
    "phase_4",
    "rewards",
    "seeding",
)


def _canonical_api() -> tuple[object, ...]:
    contracts = importlib.import_module("warp_taskgen.site_composition_contracts")
    compiler = importlib.import_module("warp_taskgen.site_composition")
    defaults = importlib.import_module("warp_taskgen.site_composition_defaults")
    default_factory = getattr(compiler, "default_site_compositions", None)
    if not callable(default_factory):
        default_factory = defaults.default_site_compositions
    return (
        compiler.SiteComposition,
        compiler.SiteBenchmarkComposition,
        compiler.SiteOwnerDeclaration,
        compiler.SiteCompositionCheckRequest,
        getattr(compiler, "SiteCompositionCheckReport", contracts.SiteCompositionCheckReport),
        compiler.check_site_composition,
        default_factory,
    )


def _classifieds_capability_values() -> tuple[str, str]:
    """Read the explicit POC card rather than duplicating its action identity."""

    composition = next(item for item in _default_compositions() if item.site == "classifieds")
    binding = composition.benchmark_compositions[0]
    return binding.supported_carriers[0], binding.supported_action_kinds[0]


def _classifieds_request(*, site: str = "classifieds", benchmark: str = "visualwebarena"):
    request_type = _canonical_api()[3]
    carrier, action_kind = _classifieds_capability_values()
    return request_type(
        site=site,
        benchmark=benchmark,
        use_case="public_reply",
        carrier=carrier,
        action_kind=action_kind,
    )


def _default_compositions() -> tuple[object, ...]:
    return tuple(_canonical_api()[-1]())


def _check(compositions: tuple[object, ...], request: object) -> object:
    return _canonical_api()[-2](compositions, request)


def _production_sources() -> tuple[Path, ...]:
    return tuple(sorted(SOURCE_ROOT.rglob("*.py")))


def _composition_sources() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in _production_sources()
            if (
                "site_composition" in path.name
                or "site_compositions" in path.parts
                or path.name in {"site_doctor.py", "site_check.py"}
            )
        )
    )


def _generic_sources() -> tuple[Path, ...]:
    paths: set[Path] = set()
    for root in GENERIC_SOURCE_ROOTS:
        paths.update((SOURCE_ROOT / root).rglob("*.py"))
    # Generic contract modules are a second, explicit locality boundary.  Do
    # not scan Site-owned modules because the synthetic noun is local there.
    paths.update(path for path in SOURCE_ROOT.glob("*.py") if "contract" in path.stem.casefold())
    return tuple(sorted(paths))


def _catalog_snapshot() -> dict[str, object]:
    from warp_taskgen.phase_2.phase_2c.policy import default_feasibility_policy_catalog
    from warp_taskgen.rewards.final_state_catalog import default_final_state_evaluator_catalog
    from warp_taskgen.seeding.site_contracts import default_seed_registry
    from warp_taskgen.sites import SiteCatalog

    return {
        "sites": SiteCatalog().sites,
        "editors": tuple(sorted(default_seed_registry().registrations)),
        "feasibility": tuple(sorted(default_feasibility_policy_catalog().policies)),
        "final_state": default_final_state_evaluator_catalog().bindings,
    }


def test_removed_python_names_are_absent_from_canonical_source() -> None:
    offenders = {
        path.relative_to(SOURCE_ROOT).as_posix(): name
        for path in _production_sources()
        for name in REMOVED_PYTHON_NAMES
        if name in path.read_text(encoding="utf-8")
    }

    assert offenders == {}, f"removed Site Python names remain in production: {offenders!r}"
    assert not (SOURCE_ROOT / "cli" / "site_doctor.py").exists()


def test_site_composition_uses_its_own_digest_and_public_reply_identity() -> None:
    offenders = {
        path.relative_to(SOURCE_ROOT).as_posix(): token
        for path in _composition_sources()
        for token in ("definition_digest", "ugc_reply")
        if token in path.read_text(encoding="utf-8")
    }

    assert offenders == {}, f"legacy Site Composition vocabulary remains: {offenders!r}"


def test_run_definition_vocabulary_remains_run_owned() -> None:
    from warp_taskgen.run_definition import define_run

    run_definition_source = (SOURCE_ROOT / "run_definition.py").read_text(encoding="utf-8")
    contract_source = (SOURCE_ROOT / "run_definition_contracts.py").read_text(encoding="utf-8")
    assert "definition_digest" in run_definition_source
    assert "definition_digest" in contract_source
    assert "site_composition_digest" not in run_definition_source
    assert "site_composition_digest" not in contract_source

    definition = define_run({"benchmark_name": "webarena_verified"})
    assert set(definition.to_dict()) == {
        "schema_version",
        "run_id",
        "source_run_id",
        "definition_digest",
        "contributions",
        "legacy",
    }
    assert "site_composition_digest" not in json.dumps(definition.to_dict(), sort_keys=True)


def test_synthetic_site_noun_stays_out_of_generic_production_modules() -> None:
    offenders = {
        path.relative_to(SOURCE_ROOT).as_posix(): "synthetic_discussion_forum"
        for path in _generic_sources()
        if "synthetic_discussion_forum" in path.read_text(encoding="utf-8")
    }

    assert offenders == {}, f"synthetic Site leaked into generic code: {offenders!r}"


def test_composition_import_and_projection_do_not_mutate_default_catalogs() -> None:
    before = _catalog_snapshot()
    _default_compositions()
    importlib.import_module("warp_taskgen.site_composition")
    importlib.import_module("warp_taskgen.site_composition_defaults")
    after = _catalog_snapshot()

    assert after == before


def test_default_cohort_is_unchanged_and_classifieds_is_explicit_diagnostic_poc() -> None:
    from warp_taskgen.phase_2.phase_2c.policy import default_feasibility_policy_catalog
    from warp_taskgen.rewards.final_state_catalog import default_final_state_evaluator_catalog
    from warp_taskgen.seeding.site_contracts import default_seed_registry
    from warp_taskgen.sites import SiteCatalog

    assert SiteCatalog().sites == ("gitlab", "reddit")
    assert tuple(sorted(default_seed_registry().registrations)) == (
        ("webarena_verified", "gitlab"),
        ("webarena_verified", "reddit"),
    )
    assert tuple(sorted(default_feasibility_policy_catalog().policies)) == (
        ("webarena_verified", "gitlab"),
        ("webarena_verified", "reddit"),
    )
    assert default_final_state_evaluator_catalog().bindings == (
        ("webarena_verified", "gitlab"),
        ("webarena_verified", "reddit"),
    )

    compositions = {composition.site: composition for composition in _default_compositions()}
    assert set(compositions) == {"gitlab", "reddit", "classifieds"}
    classifieds = compositions["classifieds"]
    assert tuple(benchmark.benchmark for benchmark in classifieds.benchmark_compositions) == (
        "visualwebarena",
    )
    binding = classifieds.benchmark_compositions[0]
    # Declarations may mark this host seam unsupported; the public_reply
    # use-case catalog excludes it and therefore the compiler derives the
    # non-applicable finding without mutating the declaration.
    assert binding.final_state_evaluation.state == "unsupported"
    report = _check(_default_compositions(), _classifieds_request())
    assert report.finding("final_state_evaluation").state == "not_applicable"
    assert binding.supported_action_kinds == (_classifieds_capability_values()[1],)


def test_unknown_removed_and_comparison_only_sites_fail_closed() -> None:
    compositions = _default_compositions()

    unknown = _check(compositions, _classifieds_request(site="site_that_is_not_declared"))
    removed = _check((), _classifieds_request(site="synthetic_discussion_forum"))
    comparison = _check(compositions, _classifieds_request(benchmark="wasp"))

    assert unknown.static_status == "invalid"
    assert removed.static_status == "invalid"
    assert comparison.static_status == "invalid"


def test_duplicate_and_malformed_compositions_fail_closed() -> None:
    composition_type = _canonical_api()[0]
    first = _default_compositions()[0]

    with pytest.raises(ValueError):
        composition_type(
            site=first.site,
            benchmark_compositions=first.benchmark_compositions + first.benchmark_compositions,
        )
    with pytest.raises((TypeError, ValueError)):
        composition_type(site="", benchmark_compositions=first.benchmark_compositions)

    request_type = _canonical_api()[3]
    with pytest.raises((TypeError, ValueError)):
        request_type(site="", benchmark="webarena_verified", use_case="public_reply")


def test_comparison_only_benchmarks_never_gain_warp_capabilities() -> None:
    from warp_taskgen.benchmark_capabilities import get_benchmark_capabilities

    for benchmark in ("wasp", "stwebagentbench", "doomarena"):
        capabilities = get_benchmark_capabilities(benchmark)
        assert capabilities.is_comparison_only is True
        assert capabilities.warp_phase_admission == ()
        assert capabilities.supports("comparison_ingestion") is True
        for capability in (
            "phase_1_generation",
            "phase_2_generation",
            "phase_2_feasibility",
            "phase_4_execution",
            "warp_evaluation",
        ):
            assert capabilities.supports(capability) is False
            with pytest.raises(ValueError):
                capabilities.require(capability)


def test_site_composition_digest_is_order_stable_and_not_a_run_definition_digest() -> None:
    compositions = _default_compositions()
    request = _classifieds_request()
    first = _check(compositions, request)
    reversed_order = _check(tuple(reversed(compositions)), request)

    assert first.static_status == reversed_order.static_status == "complete"
    assert first.site_composition_digest == reversed_order.site_composition_digest
    assert first.site_composition_digest.startswith("sha256:")
    assert len(first.site_composition_digest) == len("sha256:") + 64
    assert not hasattr(first, "definition_digest")
    payload = first.to_dict()
    assert payload["site_composition_digest"] == first.site_composition_digest
    assert "definition_digest" not in payload


def _digest_subprocess(seed: str, *, reverse_imports: bool = False, cwd: Path) -> str:
    if reverse_imports:
        imports = (
            "import importlib\n"
            "composition = importlib.import_module('warp_taskgen.site_composition')\n"
            "defaults = importlib.import_module('warp_taskgen.site_composition_defaults')\n"
        )
    else:
        imports = (
            "import importlib\n"
            "defaults = importlib.import_module('warp_taskgen.site_composition_defaults')\n"
            "composition = importlib.import_module('warp_taskgen.site_composition')\n"
        )
    carrier, action_kind = _classifieds_capability_values()
    code = (
        imports
        + "default_site_compositions = getattr(composition, 'default_site_compositions', defaults.default_site_compositions)\n"
        + "request = composition.SiteCompositionCheckRequest(site='classifieds', benchmark='visualwebarena', "
        + f"use_case='public_reply', carrier={carrier!r}, "
        + f"action_kind={action_kind!r})\n"
        + "print(composition.check_site_composition(default_site_compositions(), request).site_composition_digest)\n"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(PACKAGE_ROOT),
        "PYTHONHASHSEED": seed,
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONNOUSERSITE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_site_composition_digest_is_stable_across_hash_seeds_and_import_order(
    tmp_path: Path,
) -> None:
    digests = {
        _digest_subprocess(seed, reverse_imports=reverse, cwd=tmp_path)
        for seed in ("0", "1", "42")
        for reverse in (False, True)
    }

    assert len(digests) == 1


def test_canonical_cli_requires_exact_carrier_and_action_and_says_static_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from warp_taskgen.cli import build_parser, main

    argv = [
        "site",
        "composition",
        "check",
        "classifieds",
        "--benchmark",
        "visualwebarena",
        "--use-case",
        "public_reply",
        "--carrier",
        _classifieds_capability_values()[0],
        "--action-kind",
        _classifieds_capability_values()[1],
        "--json",
    ]
    parsed = build_parser().parse_args(argv)
    assert parsed.site == "classifieds"
    assert parsed.benchmark == "visualwebarena"
    assert parsed.use_case == "public_reply"
    assert parsed.carrier == _classifieds_capability_values()[0]
    assert parsed.action_kind == _classifieds_capability_values()[1]

    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["static_status"] == "complete"
    assert report["site_composition_digest"]
    assert report["scope"] == "static_site_composition_only"
    assert report["readiness_status"] == "blocked"
    assert report["readiness_blockers"] == [
        "active_policy_not_checked",
        "live_evidence_not_checked",
    ]
    assert report["active_policy_checked"] is False
    assert report["live_evidence_checked"] is False

    wrong_action = list(argv)
    wrong_action[wrong_action.index("--action-kind") + 1] = "not-a-classifieds-action"
    assert main(wrong_action) == 1
    incomplete = json.loads(capsys.readouterr().out)
    assert incomplete["static_status"] == "incomplete"

    human_argv = [item for item in argv if item != "--json"]
    assert main(human_argv) == 0
    human = capsys.readouterr().out.casefold()
    assert "static" in human
    assert "active policy" in human
    assert "live evidence" in human
    assert "not checked" in human


def test_canonical_cli_unknown_site_uses_invalid_exit_code_without_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from warp_taskgen.cli import main

    result = main(
        [
            "site",
            "composition",
            "check",
            "removed_site",
            "--benchmark",
            "visualwebarena",
            "--use-case",
            "public_reply",
            "--carrier",
            _classifieds_capability_values()[0],
            "--action-kind",
            _classifieds_capability_values()[1],
            "--json",
        ]
    )

    assert result == 2
    report = json.loads(capsys.readouterr().out)
    assert report["static_status"] == "invalid"
