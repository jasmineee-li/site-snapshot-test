from __future__ import annotations

import json

import pytest

from warp_taskgen.adversarial_actions.capability_adapters import (
    available_capability_adapter_profiles,
    capability_adapters_for_profile,
)
from warp_taskgen.adversarial_actions.capability_task_cards import (
    compile_capability_task_card_plan,
)
from warp_taskgen.agent_config import execution_instance_dict
from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.phases.phase_1_task_cards import validate_task_card_plan
from warp_taskgen.resume_metadata import fingerprint_payload, instance_identity
from warp_taskgen.sites.classifieds_reader import (
    preflight_classifieds_reader,
)


def _instance(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "site_name": "classifieds",
        "site_url": "https://classifieds.test",
        "reader_auth": {"type": "none"},
    }
    value.update(overrides)
    return value


def test_classifieds_reader_requires_explicit_anonymous_auth() -> None:
    result = preflight_classifieds_reader(_instance())

    assert result.ok is True
    assert result.reason is None
    assert result.browser_context_kwargs == {}
    assert result.reader_auth == {"type": "none"}
    assert result.fresh_context_required is True


def test_classifieds_reader_fails_closed_without_explicit_none_auth() -> None:
    missing = preflight_classifieds_reader(_instance(reader_auth=None))
    wrong = preflight_classifieds_reader(
        _instance(reader_auth={"type": "storage_state", "path": "writer.json"})
    )

    assert missing.ok is False
    assert missing.reason == "missing_reader_auth"
    assert wrong.ok is False
    assert wrong.reason == "non_anonymous_reader_auth"


def test_reader_auth_survives_instance_serialization_and_affects_identity() -> None:
    instance = BenchmarkInstance(
        site_name="classifieds",
        site_url="https://classifieds.test",
        reader_auth={"type": "none"},
    )
    bound = execution_instance_dict(instance, {"_worldsim_runtime": {}})
    serialized = instance.model_dump(mode="json")
    identity = instance_identity(instance)
    changed = BenchmarkInstance(
        site_name="classifieds",
        site_url="https://classifieds.test",
        reader_auth={"type": "unknown", "notes": "deliberately rejected"},
    )

    assert bound["reader_auth"] == {"type": "none"}
    assert serialized["reader_auth"] == {"type": "none"}
    json.dumps(serialized)
    assert identity["reader_auth"] == {"type": "none"}
    assert fingerprint_payload(identity) != fingerprint_payload(instance_identity(changed))


def test_absent_reader_auth_preserves_existing_instance_identity() -> None:
    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="https://gitlab.test",
    )

    assert "reader_auth" not in instance.model_dump(mode="json")
    assert "reader_auth" not in instance_identity(instance)


def test_classifieds_reader_rejects_context_auth_and_writer_reuse() -> None:
    context_auth = preflight_classifieds_reader(
        _instance(),
        reader_context_kwargs={"storage_state": {"cookies": []}},
    )
    empty_storage_state = preflight_classifieds_reader(
        _instance(),
        reader_context_kwargs={"storage_state": {}},
    )
    headers = preflight_classifieds_reader(
        _instance(),
        reader_context_kwargs={"extra_http_headers": {"Authorization": "Bearer secret"}},
    )
    same_context = object()
    reused = preflight_classifieds_reader(
        _instance(),
        writer_context=same_context,
        reader_context=same_context,
    )
    cookies = preflight_classifieds_reader(_instance(), reader_cookies=({"name": "session"},))

    assert context_auth.ok is False
    assert context_auth.reason == "reader_context_auth"
    assert empty_storage_state.ok is False
    assert empty_storage_state.reason == "reader_context_auth"
    assert headers.ok is False
    assert headers.reason == "reader_context_auth"
    assert reused.ok is False
    assert reused.reason == "writer_context_reused"
    assert cookies.ok is False
    assert cookies.reason == "reader_context_has_cookies"


def test_classifieds_reader_contract_does_not_expose_runtime_internals() -> None:
    result = preflight_classifieds_reader(_instance())

    assert result.to_metadata() == {
        "reader_auth": {"type": "none"},
        "fresh_context_required": True,
        "writer_context_reuse_forbidden": True,
    }
    assert "storage_state" not in result.to_metadata()
    assert "cookies" not in result.to_metadata()


def test_named_classifieds_profile_has_one_safe_natural_semantic_card() -> None:
    assert "classifieds_listing_reply_poc" in available_capability_adapter_profiles()
    adapters = capability_adapters_for_profile("classifieds_listing_reply_poc")
    assert len(adapters) == 1
    adapter = adapters[0]
    assert adapter.site == "classifieds"
    assert adapter.action_kind == "answer_opposite_binary_label"
    assert adapter.benign_reward_shape is None
    assert adapter.scenario_template is not None

    plan = compile_capability_task_card_plan("classifieds_listing_reply_poc")
    validate_task_card_plan(plan)
    card = plan["task_cards"][0]
    assert card["site"] == "classifieds"
    assert card["compatible_action_kinds"] == ["answer_opposite_binary_label"]
    assert card["benign_reward_shape"] == "agent_response_binary"
    assert card["scenario_template"]["workflow_family"] == "read_only_control"
    public_template = card["scenario_template"]
    assert "title" not in str(public_template).casefold()
    assert "description" not in str(public_template).casefold()
    assert all(
        forbidden not in str(public_template).casefold()
        for forbidden in (
            "attacker.invalid",
            "selector",
            "credential",
            "reset",
            "reward_logic",
            "payload",
            "listing_id",
        )
    )


def test_classifieds_profile_rejects_non_visualwebarena_benchmarks() -> None:
    for benchmark_family in ("webarena_verified", "unknown"):
        with pytest.raises(ValueError, match="no cards"):
            capability_adapters_for_profile(
                "classifieds_listing_reply_poc",
                benchmark_family=benchmark_family,
            )


def test_existing_profile_defaults_remain_webarena_verified() -> None:
    adapters = capability_adapters_for_profile("semantic_minval")

    assert adapters
    assert {adapter.benchmark_family for adapter in adapters} == {"webarena_verified"}
