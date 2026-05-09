from __future__ import annotations

from types import SimpleNamespace

from worldsim import _sandbox_runner


def test_rate_limit_event_payload_includes_structured_fields():
    message = SimpleNamespace(
        rate_limit_info=SimpleNamespace(
            status="rejected",
            rate_limit_type="five_hour",
            resets_at=1_700_000_120,
            utilization=SimpleNamespace(used=7, limit=10),
        )
    )

    payload = _sandbox_runner._rate_limit_event_payload(message, now_ts=1_700_000_000)

    assert payload == {
        "type": "rate_limit",
        "status": "rejected",
        "rate_limit_type": "five_hour",
        "resets_at": 1_700_000_120.0,
        "retry_after_seconds": 120.0,
        "utilization": {"used": 7, "limit": 10},
    }


def test_rate_limit_event_payload_clamps_negative_retry_after():
    message = SimpleNamespace(
        rate_limit_info=SimpleNamespace(
            status="allowed_warning",
            type="seven_day",
            resets_at=50,
            utilization=None,
        )
    )

    payload = _sandbox_runner._rate_limit_event_payload(message, now_ts=75)

    assert payload["retry_after_seconds"] == 0.0
    assert payload["rate_limit_type"] == "seven_day"
