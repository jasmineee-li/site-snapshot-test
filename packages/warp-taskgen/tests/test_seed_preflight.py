from __future__ import annotations

from typing import ClassVar

from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.editors.base import EditorError
from warp_taskgen.instance_selection import select_task_site_instance
from warp_taskgen.phase_4 import preflight as phase_4_preflight
from warp_taskgen.seeding.site_contracts import (
    SeedSiteRegistration,
    SeedSiteRegistry,
    default_seed_registry,
)


class _ProbeEditor:
    calls: ClassVar[list[dict[str, object]]] = []
    behavior = None

    @classmethod
    def probe_base_state(cls, instance):
        cls.calls.append(instance)
        if cls.behavior is not None:
            cls.behavior(instance)


def _install_probe_editor(monkeypatch, behavior=None, *, benchmark: str = "webarena_verified"):
    _ProbeEditor.calls = []
    _ProbeEditor.behavior = behavior
    registrations = dict(default_seed_registry().registrations)
    replacement = SeedSiteRegistration(benchmark, "gitlab", _ProbeEditor)
    registrations[replacement.key] = replacement
    monkeypatch.setattr(
        phase_4_preflight,
        "default_seed_registry",
        lambda: SeedSiteRegistry(registrations),
    )


def test_probe_seed_base_state_uses_cache(monkeypatch):
    _install_probe_editor(monkeypatch)
    cache: dict[tuple[str, str, str, str], phase_4_preflight.BaseStateProbeResult] = {}
    instance = {
        "site_name": "gitlab",
        "site_url": "http://gitlab.test",
        "auth": {"type": "bearer_token", "token": "token"},
    }

    first = phase_4_preflight._probe_seed_base_state(instance, cache=cache)
    second = phase_4_preflight._probe_seed_base_state(instance, cache=cache)

    assert first.ok is True
    assert second.ok is True
    assert len(_ProbeEditor.calls) == 1


def test_probe_seed_base_state_cache_is_scoped_by_auth(monkeypatch):
    _install_probe_editor(monkeypatch)
    cache: dict[tuple[str, str, str, str], phase_4_preflight.BaseStateProbeResult] = {}

    first = phase_4_preflight._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "token-a"},
        },
        cache=cache,
    )
    second = phase_4_preflight._probe_seed_base_state(
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

    result = phase_4_preflight._probe_seed_base_state(
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

    result = phase_4_preflight._probe_seed_base_state(
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
        return phase_4_preflight.BaseStateProbeResult(ok=True)

    monkeypatch.setattr(phase_4_preflight, "_probe_seed_base_state", fake_probe)

    errors = phase_4_preflight._probe_seed_base_state_for_task_targets([task], instances)

    assert errors == []
    assert probed == [expected.site_url]


def test_probe_seed_base_state_uses_task_benchmark(monkeypatch):
    _install_probe_editor(monkeypatch, benchmark="stwebagentbench")

    result = phase_4_preflight._probe_seed_base_state(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"type": "bearer_token", "token": "token"},
        },
        benchmark="stwebagentbench",
    )

    assert result.ok is True
    assert len(_ProbeEditor.calls) == 1
