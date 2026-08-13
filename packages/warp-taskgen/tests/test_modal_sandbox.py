from __future__ import annotations

from warp_taskgen import modal_sandbox


def test_update_watchdog_state_tracks_rejected_rate_limit_deadline():
    state = modal_sandbox.SandboxWatchdogState(
        last_event_at=10.0,
        last_non_rate_limit_progress_at=10.0,
        last_liveness_log_at=10.0,
    )

    modal_sandbox._update_watchdog_state(
        state,
        {
            "type": "rate_limit",
            "status": "rejected",
            "resets_at": 120.0,
            "retry_after_seconds": 15.0,
        },
        now_monotonic=20.0,
        now_wall=100.0,
    )

    assert state.last_event_at == 20.0
    assert state.last_non_rate_limit_progress_at == 10.0
    assert state.blocked_until == 120.0
    assert state.consecutive_rejected_rate_limits == 1


def test_update_watchdog_state_resets_after_non_rate_limit_progress():
    state = modal_sandbox.SandboxWatchdogState(
        last_event_at=20.0,
        last_non_rate_limit_progress_at=10.0,
        last_liveness_log_at=20.0,
        blocked_until=120.0,
        consecutive_rejected_rate_limits=2,
    )

    modal_sandbox._update_watchdog_state(
        state,
        {"type": "tool_call", "tool": "Read"},
        now_monotonic=25.0,
        now_wall=105.0,
    )

    assert state.last_event_at == 25.0
    assert state.last_non_rate_limit_progress_at == 25.0
    assert state.blocked_until is None
    assert state.consecutive_rejected_rate_limits == 0


def test_watchdog_timeout_reason_waits_for_backoff_deadline():
    state = modal_sandbox.SandboxWatchdogState(
        last_event_at=40.0,
        last_non_rate_limit_progress_at=30.0,
        last_liveness_log_at=40.0,
        blocked_until=200.0,
        consecutive_rejected_rate_limits=1,
    )

    assert (
        modal_sandbox._watchdog_timeout_reason(
            state,
            now_monotonic=100.0,
            now_wall=250.0,
            rate_limit_grace_seconds=60,
        )
        is None
    )

    reason = modal_sandbox._watchdog_timeout_reason(
        state,
        now_monotonic=130.0,
        now_wall=261.0,
        rate_limit_grace_seconds=60,
    )

    assert reason is not None
    assert "rejected rate limit backoff" in reason


def test_watchdog_timeout_reason_detects_general_silence():
    state = modal_sandbox.SandboxWatchdogState(
        last_event_at=0.0,
        last_non_rate_limit_progress_at=0.0,
        last_liveness_log_at=0.0,
    )

    reason = modal_sandbox._watchdog_timeout_reason(
        state,
        now_monotonic=1_250.0,
        now_wall=1_250.0,
        silence_seconds=1_200,
    )

    assert reason == "no sandbox events for 1250s"


def test_watchdog_liveness_message_throttles_until_interval():
    state = modal_sandbox.SandboxWatchdogState(
        last_event_at=10.0,
        last_non_rate_limit_progress_at=10.0,
        last_liveness_log_at=100.0,
    )

    assert (
        modal_sandbox._watchdog_liveness_message(
            state,
            now_monotonic=150.0,
            now_wall=150.0,
            tool_calls=4,
            liveness_log_seconds=120,
        )
        is None
    )
    assert state.last_liveness_log_at == 100.0


def test_watchdog_liveness_message_reports_quiet_tool_execution():
    state = modal_sandbox.SandboxWatchdogState(
        last_event_at=10.0,
        last_non_rate_limit_progress_at=5.0,
        last_liveness_log_at=20.0,
    )

    message = modal_sandbox._watchdog_liveness_message(
        state,
        now_monotonic=145.0,
        now_wall=145.0,
        tool_calls=56,
        liveness_log_seconds=120,
    )

    assert message is not None
    assert "quiet=135s" in message
    assert "non_rate_limit_quiet=140s" in message
    assert "tool_calls=56" in message
    assert state.last_liveness_log_at == 145.0


def test_watchdog_liveness_message_reports_rate_limit_backoff():
    state = modal_sandbox.SandboxWatchdogState(
        last_event_at=10.0,
        last_non_rate_limit_progress_at=5.0,
        last_liveness_log_at=20.0,
        blocked_until=200.0,
        consecutive_rejected_rate_limits=2,
    )

    message = modal_sandbox._watchdog_liveness_message(
        state,
        now_monotonic=145.0,
        now_wall=175.0,
        tool_calls=12,
        liveness_log_seconds=120,
    )

    assert message is not None
    assert "rate_limit_backoff_remaining=25s" in message
    assert "rejected_events=2" in message
