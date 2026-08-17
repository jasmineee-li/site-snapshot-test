"""Assertions for the host-owned action-card owner seam."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest


def assert_action_card_behavior(
    card: Any,
    *,
    expected_site: str,
    expected_benchmark: str,
    expected_carrier: str,
    expected_action_kind: str,
) -> None:
    """Check route/target identity and malformed or duplicate route rejection."""

    assert card.site == expected_site
    assert card.benchmark_family == expected_benchmark
    assert card.action_kind == expected_action_kind
    assert getattr(card, "carrier_surface", None) == expected_carrier
    assert card.route_ids
    card.validate()

    for route_ids in ((), (card.route_ids[0], card.route_ids[0])):
        malformed = replace(card, route_ids=route_ids)
        with pytest.raises(ValueError):
            malformed.validate()
