from __future__ import annotations

import asyncio

from worldsim.config import BenchmarkInstance
from worldsim.editors.base import EditorError
from worldsim.instance_selection import select_task_site_instance
from worldsim.phases import phase_4_adversarial


class _ProbeEditor:
    calls: list[dict[str, object]] = []
    behavior = None

    @classmethod
    def probe_base_state(cls, instance):
        cls.calls.append(instance)
        if cls.behavior is not None:
            cls.behavior(instance)


def _install_probe_editor(monkeypatch, behavior=None, *, benchmark: str = "webarena_verified"):
    _ProbeEditor.calls = []
    _ProbeEditor.behavior = behavior
    monkeypatch.setitem(
        phase_4_adversarial.EDITOR_REGISTRY,
        (benchmark, "gitlab"),
        _ProbeEditor,
    )


def test_probe_seed_base_state_uses_cache(monkeypatch):
    _install_probe_editor(monkeypatch)
    cache: dict[tuple[str, str, str, str], phase_4_adversarial.BaseStateProbeResult] = {}
    instance = {
        "site_name": "gitlab",
        "site_url": "http://gitlab.test",
        "auth": {"type": "bearer_token", "token": "token"},
    }

    first = phase_4_adversarial._probe_seed_base_state(instance, cache=cache)
    second = phase_4_adversarial._probe_seed_base_state(instance, cache=cache)

    assert first.ok is True
    assert second.ok is True
    assert len(_ProbeEditor.calls) == 1


def test_probe_seed_base_state_cache_is_scoped_by_auth(monkeypatch):
    _install_probe_editor(monkeypatch)
    cache: dict[tuple[str, str, str, str], phase_4_adversarial.BaseStateProbeResult] = {}

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
    assert len(_ProbeEditor.calls) == 2


def test_probe_seed_base_state_reports_auth_missing(monkeypatch):
    _install_probe_editor(
        monkeypatch,
        behavior=lambda instance: (_ for _ in ()).throw(
            EditorError("auth_missing", "gitlab probe returned HTTP 401")
        ),
    )

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


def test_probe_seed_base_state_treats_unexpected_errors_as_base_state_missing(monkeypatch):
    _install_probe_editor(
        monkeypatch,
        behavior=lambda instance: (_ for _ in ()).throw(RuntimeError("HTTP 302 redirect")),
    )

    result = phase_4_adversarial._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "token"},
        }
    )

    assert result.ok is False
    assert result.mismatch is not None
    assert result.mismatch.kind == "base_state_missing"
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
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "update_user_status",
                    "args": {"message": "payload"},
                }
            ],
        },
    }
    expected = select_task_site_instance(task, "gitlab", instances)

    def fake_probe(instance, benchmark="webarena_verified", cache=None):
        probed.append(instance["site_url"])
        assert benchmark == "webarena_verified"
        return phase_4_adversarial.BaseStateProbeResult(ok=True)

    monkeypatch.setattr(phase_4_adversarial, "_probe_seed_base_state", fake_probe)

    errors = phase_4_adversarial._probe_seed_base_state_for_task_targets([task], instances)

    assert errors == []
    assert probed == [expected.site_url]


def test_probe_seed_base_state_uses_task_benchmark(monkeypatch):
    _install_probe_editor(monkeypatch, benchmark="custom_benchmark")

    result = phase_4_adversarial._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "token"},
        },
        benchmark="custom_benchmark",
    )

    assert result.ok is True
    assert len(_ProbeEditor.calls) == 1


def test_preflight_adversarial_seed_preserves_structured_mismatch_fields(monkeypatch):
    monkeypatch.setattr(
        phase_4_adversarial,
        "_probe_seed_base_state",
        lambda instance, benchmark="webarena_verified", cache=None: phase_4_adversarial.BaseStateProbeResult(ok=True),
    )
    monkeypatch.setattr(
        phase_4_adversarial,
        "preflight_http_seed_calls",
        lambda seed, instance: ["api_calls[0]: editor exploded"],
    )
    monkeypatch.setattr(phase_4_adversarial, "preflight_editor_seed_calls", lambda seed, instance: [])

    report = asyncio.run(
        phase_4_adversarial.preflight_adversarial_seed(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "method": "create_group",
                        "args": {"name_template": "seed-group"},
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
    assert mismatch.resource_type == "legacy_http"
    assert mismatch.kind == "seed_error"
