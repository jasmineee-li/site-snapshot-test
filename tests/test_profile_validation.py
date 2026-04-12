from __future__ import annotations

import pytest

from worldsim.profile_validation import load_and_validate_profile, validate_profile


def test_load_and_validate_profile_rejects_invalid_json(tmp_path):
    profile_path = tmp_path / "BENCHMARK_PROFILE_shopping.json"
    profile_path.write_text("{not-json")

    with pytest.raises(ValueError, match="invalid profile JSON"):
        load_and_validate_profile("shopping", profile_path)


def test_validate_profile_rejects_eval_types_absent_from_manifest():
    profile = {
        "data_model": [],
        "injection_surface": [],
        "verification_capabilities": [{"eval_type": "db_query_match"}],
    }

    with pytest.raises(
        ValueError,
        match="verification capabilities reference eval types absent from manifest",
    ):
        validate_profile("shopping", profile, manifest_eval_types=["string_match"])
