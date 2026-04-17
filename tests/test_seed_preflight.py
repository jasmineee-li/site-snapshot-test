from __future__ import annotations

import requests

from worldsim.phases import phase_4_adversarial
from worldsim.config import BenchmarkInstance
from worldsim.instance_selection import select_task_site_instance


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, text: str = "ok") -> None:
        self.status_code = status_code
        self.text = text


def test_probe_seed_base_state_uses_cache(monkeypatch):
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    cache: dict[tuple[str, str, str], phase_4_adversarial.BaseStateProbeResult] = {}
    instance = {
        "site_name": "gitlab",
        "site_url": "http://gitlab.test",
        "auth": {"type": "bearer_token", "token": "token"},
    }

    first = phase_4_adversarial._probe_seed_base_state(instance, cache=cache)
    second = phase_4_adversarial._probe_seed_base_state(instance, cache=cache)

    assert first.ok is True
    assert second.ok is True
    assert calls == ["http://gitlab.test/api/v4/user"]


def test_probe_seed_base_state_cache_is_scoped_by_auth(monkeypatch):
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(kwargs["headers"]["Authorization"])
        return _FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    cache: dict[tuple[str, str, str], phase_4_adversarial.BaseStateProbeResult] = {}

    first = phase_4_adversarial._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "token-a"},
        },
        cache=cache,
    )
    second = phase_4_adversarial._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "token-b"},
        },
        cache=cache,
    )

    assert first.ok is True
    assert second.ok is True
    assert calls == ["Bearer token-a", "Bearer token-b"]


def test_probe_seed_base_state_reports_auth_missing(monkeypatch):
    def fake_get(url, **kwargs):
        return _FakeResponse(status_code=401)

    monkeypatch.setattr(requests, "get", fake_get)

    result = phase_4_adversarial._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "bad-token"},
        }
    )

    assert result.ok is False
    assert result.mismatch is not None
    assert result.mismatch.kind == "auth_missing"
    assert result.mismatch.site == "gitlab"


def test_probe_seed_base_state_treats_redirect_as_auth_failure(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _FakeResponse(status_code=302))

    result = phase_4_adversarial._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "token"},
        }
    )

    assert result.ok is False
    assert result.mismatch is not None
    assert result.mismatch.kind == "auth_missing"
    assert "HTTP 302 redirect" in result.mismatch.detail


def test_probe_seed_base_state_for_task_targets_only_probes_selected_instance(monkeypatch):
    probed: list[str] = []
    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-a.test",
            reset_endpoint="http://gitlab-a.test/reset",
            replica_index=0,
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-b.test",
            reset_endpoint="http://gitlab-b.test/reset",
            replica_index=1,
        ),
    ]
    task = {
        "id": "adv-gitlab-1",
        "site": "gitlab",
        "adversarial_data_seed": {
            "mechanism": "api",
            "api_calls": [
                {"target": {"site": "gitlab", "resource_type": "user_status", "update": {}}}
            ],
        },
    }
    expected = select_task_site_instance(task, "gitlab", instances)

    def fake_probe(instance, cache=None):
        probed.append(instance["site_url"])
        return phase_4_adversarial.BaseStateProbeResult(ok=True)

    monkeypatch.setattr(phase_4_adversarial, "_probe_seed_base_state", fake_probe)

    errors = phase_4_adversarial._probe_seed_base_state_for_task_targets([task], instances)

    assert errors == []
    assert probed == [expected.site_url]


def test_preflight_adversarial_seed_preserves_structured_mismatch_fields(monkeypatch):
    monkeypatch.setattr(
        phase_4_adversarial,
        "_probe_seed_base_state",
        lambda instance, cache=None: phase_4_adversarial.BaseStateProbeResult(ok=True),
    )
    monkeypatch.setattr(
        phase_4_adversarial,
        "preflight_http_seed_calls",
        lambda seed, instance: ["api_calls[0]: resolver exploded"],
    )

    report = __import__("asyncio").run(
        phase_4_adversarial.preflight_adversarial_seed(
            {
                "mechanism": "api",
                "api_calls": [
                    {
                        "target": {
                            "site": "gitlab",
                            "resource_type": "group",
                            "create": {"group": {"name_template": "seed-group"}},
                        },
                        "body": {"description": "payload"},
                    }
                ],
            },
            {"site_name": "gitlab", "site_url": "http://gitlab.test"},
        )
    )

    assert report.ok is False
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.call_index == 0
    assert mismatch.site == "gitlab"
    assert mismatch.resource_type == "group"
    assert mismatch.kind == "resolver_error"
