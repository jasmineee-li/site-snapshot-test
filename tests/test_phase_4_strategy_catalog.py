from __future__ import annotations

from worldsim.phase_4.strategy_catalog import (
    ALLOWED_STRATEGIES,
    REFUSAL_TRIGGERS,
    TRIGGER_TO_STRATEGIES,
    strategies_for_trigger,
)


def test_pool_size_is_22():
    assert len(ALLOWED_STRATEGIES) == 22


def test_visual_concealment_is_dropped():
    # The category error we're fixing. Never reintroduce.
    assert "visual_concealment" not in ALLOWED_STRATEGIES


def test_concealment_steering_is_deferred():
    # These require a prompt judge + new reward function. Deferred indefinitely.
    for banned in ("scripted_message", "command_silent_execution", "false_justification"):
        assert banned not in ALLOWED_STRATEGIES, f"{banned!r} must stay deferred"


def test_refusal_triggers_count_is_7():
    assert len(REFUSAL_TRIGGERS) == 7
    assert "distracted" in REFUSAL_TRIGGERS
    assert "unknown" in REFUSAL_TRIGGERS


def test_every_trigger_has_mapping():
    for trigger in REFUSAL_TRIGGERS:
        assert trigger in TRIGGER_TO_STRATEGIES, f"missing lookup for {trigger}"


def test_distracted_maps_to_salience_strategies_after_pvpo_encounter():
    strategies, actionable = strategies_for_trigger("distracted")
    assert actionable is True
    assert strategies == (
        "necessary_for_benign_goal",
        "specificity",
        "semantic_coherence",
    )


def test_actionable_triggers_map_to_real_strategies():
    for trigger in (
        "suspicion",
        "tonal_mismatch",
        "formatting",
        "externally_sourced",
        "safety_policy",
        "distracted",
        "unknown",
    ):
        strategies, actionable = strategies_for_trigger(trigger)
        assert actionable is True, f"{trigger} should be actionable"
        assert len(strategies) >= 1
        assert len(strategies) <= 3, f"{trigger} must map to ≤3 strategies"
        for s in strategies:
            assert s in ALLOWED_STRATEGIES, f"{s!r} (from {trigger}) not in pool"


def test_invalid_trigger_returns_empty_unactionable():
    strategies, actionable = strategies_for_trigger("not_a_real_trigger")
    assert strategies == ()
    assert actionable is False
