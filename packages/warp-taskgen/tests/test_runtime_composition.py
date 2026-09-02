from __future__ import annotations

import pytest

from warp_taskgen.runtime_composition import (
    CLASSIFIEDS_LISTING_REPLY_POC,
    ROCKET_CHAT_CONVERSATION_DECISION_POC,
    RuntimeComposition,
    classifieds_listing_reply_poc,
    rocket_chat_conversation_decision_poc,
    runtime_composition_for_name,
)


def test_classifieds_runtime_composition_is_explicit_and_isolated() -> None:
    composition = classifieds_listing_reply_poc()

    assert composition.name == CLASSIFIEDS_LISTING_REPLY_POC
    assert composition.site_catalog.sites == ("classifieds",)
    assert composition.seed_registry.get("visualwebarena", "classifieds") is not None
    assert composition.feasibility_policy_catalog.get("visualwebarena", "classifieds") is not None
    assert composition.strict_seed_cleanup is True


def test_rocket_chat_runtime_composition_is_explicit_and_non_default() -> None:
    composition = rocket_chat_conversation_decision_poc()

    assert composition.name == ROCKET_CHAT_CONVERSATION_DECISION_POC
    assert composition.site_catalog.sites == ("rocketchat",)
    assert composition.seed_registry.get("theagentcompany", "rocketchat") is not None
    assert composition.feasibility_policy_catalog.get("theagentcompany", "rocketchat") is not None
    assert composition.strict_seed_cleanup is True
    assert runtime_composition_for_name(ROCKET_CHAT_CONVERSATION_DECISION_POC).name == composition.name


def test_runtime_composition_defaults_to_none_and_unknown_fails_closed() -> None:
    assert runtime_composition_for_name(None) is None
    assert runtime_composition_for_name("") is None

    with pytest.raises(ValueError, match="unknown runtime composition"):
        runtime_composition_for_name("classifieds")


def test_runtime_composition_is_frozen() -> None:
    composition = classifieds_listing_reply_poc()

    with pytest.raises((AttributeError, TypeError)):
        composition.name = "other"  # type: ignore[misc]


def test_runtime_composition_rejects_wrong_catalog_types() -> None:
    composition = classifieds_listing_reply_poc()

    with pytest.raises(TypeError, match="site_catalog"):
        RuntimeComposition(
            name=composition.name,
            site_catalog=object(),  # type: ignore[arg-type]
            seed_registry=composition.seed_registry,
            feasibility_policy_catalog=composition.feasibility_policy_catalog,
        )
