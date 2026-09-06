# ruff: noqa: F403, F405
"""Sandbox validator profile validation (``validate_profile``).

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

from ._fixtures import *


class TestValidateProfile:
    def test_rejects_site_name_mismatch(self):
        profile = {
            "site_name": "shopping",
            "data_model": [],
            "injection_surface": [],
        }
        errors = validator.validate_profile(profile, site_name="gitlab")
        assert any("mismatch" in e for e in errors)

    def test_rejects_unknown_entity_ref(self):
        profile = {
            "site_name": "shopping",
            "data_model": [
                {
                    "entity": "Product",
                    "fields": [{"name": "title"}],
                }
            ],
            "injection_surface": [
                {
                    "id": "surf-1",
                    "source_field": "Review.body",
                }
            ],
        }
        errors = validator.validate_profile(profile, site_name="shopping")
        assert any("unknown entity" in e for e in errors)

    def test_rejects_unknown_field_ref(self):
        profile = {
            "site_name": "shopping",
            "data_model": [
                {
                    "entity": "Product",
                    "fields": [{"name": "title"}],
                }
            ],
            "injection_surface": [
                {
                    "id": "surf-1",
                    "source_field": "Product.nonexistent_field",
                }
            ],
        }
        errors = validator.validate_profile(profile, site_name="shopping")
        assert any("unknown field" in e for e in errors)

    def test_rejects_field_from_wrong_entity(self):
        profile = {
            "site_name": "shopping",
            "data_model": [
                {
                    "entity": "Product",
                    "fields": [{"name": "title"}],
                },
                {
                    "entity": "Review",
                    "fields": [{"name": "body"}],
                },
            ],
            "injection_surface": [
                {
                    "id": "surf-1",
                    "source_field": "Product.body",
                }
            ],
        }
        errors = validator.validate_profile(profile, site_name="shopping")
        assert any("unknown field" in e for e in errors)

    def test_accepts_valid_profile(self):
        profile = {
            "site_name": "shopping",
            "data_model": [
                {
                    "entity": "Product",
                    "fields": [{"name": "title"}, {"name": "description"}],
                }
            ],
            "injection_surface": [
                {
                    "id": "surf-1",
                    "source_field": "Product.description",
                }
            ],
        }
        errors = validator.validate_profile(profile, site_name="shopping")
        assert errors == []


class TestProfileInjectionSurfaceFieldMismatch:
    """injection_surface references an entity that exists but a field that doesn't."""

    def test_entity_exists_field_missing(self):
        profile = {
            "site_name": "shopping",
            "data_model": [
                {
                    "entity": "Product",
                    "fields": [{"name": "title"}, {"name": "price"}],
                }
            ],
            "injection_surface": [
                {
                    "id": "surf-1",
                    "source_field": "Product.nonexistent_field",
                }
            ],
        }
        errors = validator.validate_profile(profile, site_name="shopping")
        # Entity "Product" exists, so no entity error
        assert not any("unknown entity" in e for e in errors)
        # But "nonexistent_field" doesn't match any field name
        assert any("unknown field" in e for e in errors)


class TestProfileEmptyDataModel:
    """Profile with empty data_model but non-empty injection_surface."""

    def test_empty_data_model_no_entity_or_field_errors(self):
        """When data_model is empty, known_entities and known_fields are empty.

        The validator guards with 'and known_entities' / 'and known_fields',
        so no errors are raised for unknown references when there's nothing
        to cross-reference against.
        """
        profile = {
            "site_name": "shopping",
            "data_model": [],
            "injection_surface": [
                {
                    "id": "surf-1",
                    "source_field": "Ghost.phantom_field",
                }
            ],
        }
        errors = validator.validate_profile(profile, site_name="shopping")
        # With empty data_model, the guard clauses (known_entities/known_fields)
        # prevent any errors since there's nothing to validate against.
        assert errors == []

    def test_missing_data_model_key(self):
        """Profile without data_model key at all."""
        profile = {
            "site_name": "shopping",
            "injection_surface": [
                {
                    "id": "surf-1",
                    "source_field": "Product.title",
                }
            ],
        }
        errors = validator.validate_profile(profile, site_name="shopping")
        # data_model defaults to [] via .get("data_model", [])
        assert errors == []
