# ruff: noqa: F403, F405
"""Parity between the stdlib-only sandbox validator and ``warp_taskgen.seed_contracts``,
including the ``_CORE_SURFACE_ALIASES`` table mirrored from the Site-owned carrier policies.

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

import pytest

from warp_taskgen.sites import default_catalog

from ._fixtures import *


def test_core_surface_aliases_match_site_owned_carrier_policies() -> None:
    """The stdlib-only sandbox copy must equal the Site-owned alias tables."""
    policies = {
        site: default_catalog().bind(benchmark="webarena_verified", site=site).carrier_policy()
        for site in ("gitlab", "reddit")
    }
    assert validator._CORE_SURFACE_ALIASES == {
        site: dict(policy.surface_aliases) for site, policy in policies.items()
    }
    for site, policy in policies.items():
        for raw in policy.surface_aliases:
            assert validator._canonical_core_surface(site, raw) == policy.canonical_surface(raw)


class TestProfileValidationParity:
    """Compare sandbox and orchestrator profile validation coverage."""

    def test_both_catch_site_name_mismatch(self):
        profile = {
            "site_name": "shopping",
            "data_model": [],
            "injection_surface": [],
        }
        # Sandbox
        sandbox_errors = validator.validate_profile(profile, site_name="gitlab")
        assert any("mismatch" in e for e in sandbox_errors)

        # Orchestrator
        from warp_taskgen.profile_validation import validate_profile as orch_validate

        with pytest.raises(ValueError, match="mismatch"):
            orch_validate("gitlab", profile)

    def test_both_catch_unknown_entity(self):
        profile = {
            "site_name": "shopping",
            "data_model": [
                {"entity": "Product", "fields": [{"name": "title"}]},
            ],
            "injection_surface": [
                {"id": "s1", "source_field": "Review.body"},
            ],
        }
        sandbox_errors = validator.validate_profile(profile, site_name="shopping")
        assert any("unknown entity" in e for e in sandbox_errors)

        from warp_taskgen.profile_validation import validate_profile as orch_validate

        with pytest.raises(ValueError, match="unknown entity"):
            orch_validate("shopping", profile)

    def test_both_catch_unknown_field(self):
        profile = {
            "site_name": "shopping",
            "data_model": [
                {"entity": "Product", "fields": [{"name": "title"}]},
            ],
            "injection_surface": [
                {"id": "s1", "source_field": "Product.nonexistent"},
            ],
        }
        sandbox_errors = validator.validate_profile(profile, site_name="shopping")
        assert any("unknown field" in e for e in sandbox_errors)

        from warp_taskgen.profile_validation import validate_profile as orch_validate

        with pytest.raises(ValueError, match="unknown field"):
            orch_validate("shopping", profile)

    def test_orchestrator_checks_eval_types_sandbox_does_not(self):
        """Orchestrator validates verification_capabilities vs manifest eval types.

        The sandbox validator does NOT check this because it doesn't have
        access to the manifest at profile-validation time. This is a known
        and acceptable gap.
        """
        profile = {
            "site_name": "shopping",
            "data_model": [],
            "injection_surface": [],
            "verification_capabilities": [
                {"eval_type": "CustomEvaluator"},
            ],
        }
        # Sandbox: no error (doesn't check eval types)
        sandbox_errors = validator.validate_profile(profile, site_name="shopping")
        assert sandbox_errors == []

        # Orchestrator: error when manifest doesn't have that eval type
        from warp_taskgen.profile_validation import validate_profile as orch_validate

        with pytest.raises(ValueError, match="eval types absent"):
            orch_validate(
                "shopping",
                profile,
                manifest_eval_types=["AgentResponseEvaluator"],
            )


class TestDataSeedValidationParity:
    """Compare sandbox and orchestrator data seed validation."""

    def test_both_reject_non_dict(self):
        # Sandbox returns errors list
        sandbox_errors = validator.validate_data_seed("not a dict")
        assert any("object" in e for e in sandbox_errors)

        # Orchestrator raises ValueError
        from warp_taskgen.seeding import validate_data_seed as orch_validate

        with pytest.raises(ValueError, match="object"):
            orch_validate("not a dict")

    def test_both_reject_empty_mechanism(self):
        seed = {"foo": "bar"}
        sandbox_errors = validator.validate_data_seed(seed)
        assert any("mechanism" in e for e in sandbox_errors)

        from warp_taskgen.seeding import validate_data_seed as orch_validate

        with pytest.raises(ValueError, match="mechanism"):
            orch_validate(seed)

    def test_both_reject_sql_mechanism(self):
        seed = {"mechanism": "sql", "statements": ["INSERT INTO t (c) VALUES ('v')"]}
        sandbox_errors = validator.validate_data_seed(seed)
        assert any("unknown" in e for e in sandbox_errors)

        from warp_taskgen.seeding import validate_data_seed as orch_validate

        with pytest.raises(ValueError, match="unknown"):
            orch_validate(seed)

    def test_both_reject_unknown_mechanism(self):
        seed = {"mechanism": "carrier_pigeon"}
        sandbox_errors = validator.validate_data_seed(seed)
        assert any("unknown" in e for e in sandbox_errors)

        from warp_taskgen.seeding import validate_data_seed as orch_validate

        with pytest.raises(ValueError, match="unknown"):
            orch_validate(seed)

    def test_both_reject_none_mechanism_with_editor_calls(self):
        seed = {
            "mechanism": "none",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue_title",
                    "args": {"project_id": "1", "title": "Seeded title"},
                }
            ],
        }
        sandbox_errors = validator.validate_data_seed(seed, allow_none=True)
        assert any("must not include editor_calls" in e for e in sandbox_errors)

        from warp_taskgen.seeding import validate_data_seed as orch_validate

        with pytest.raises(ValueError, match="must not include editor_calls"):
            orch_validate(seed, allow_none=True)

    def test_both_reject_unsupported_editor_method_for_known_site(self):
        seed = {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "destroy_everything",
                    "args": {"name": "x"},
                }
            ],
        }
        sandbox_errors = validator.validate_data_seed(seed)
        assert any("not supported" in e for e in sandbox_errors)

        from warp_taskgen.seeding import validate_data_seed as orch_validate

        with pytest.raises(ValueError, match="not supported"):
            orch_validate(seed)
