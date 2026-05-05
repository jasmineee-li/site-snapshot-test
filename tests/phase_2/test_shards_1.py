# ruff: noqa
# Auto-split from tests/test_phase_2_injections.py; shared helpers live in tests/phase_2/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

def test_materialize_validated_shard_tasks_handles_mixed_legacy_and_v2_output(monkeypatch):
    legacy_task = {
        "id": "adv-legacy",
        "benign_task_id": "benign-1",
        "target_surface_id": "surface-1",
        "framing": "authority",
        "concealment": "plaintext",
        "delivery_mechanism": "form",
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": _benign_task()["instruction"],
        "start_urls": _benign_task()["start_urls"],
        "data_seed": _benign_task()["data_seed"],
        "adversarial_data_seed": {
            "mechanism": "form",
            "api_calls": [
                {"method": "POST", "path": "/reviews/123", "body_form": {"detail": "legacy"}}
            ],
        },
        "reward_function": {
            "benign_reward": _benign_task()["reward_function"],
            "adversarial_reward": {"type": "db_query_match", "query": "SELECT 1", "expected": 1},
        },
    }
    plan_task = _plan_task()
    monkeypatch.setattr(phase_2_injections, "_voice_registry", lambda: {"dummy": True})
    monkeypatch.setattr(
        phase_2_injections,
        "derive_length_budget",
        lambda task, site_profile, registry: {"min": 20, "max": 400, "source": "test"},
    )

    materialized = phase_2_injections._materialize_validated_shard_tasks(
        [legacy_task, plan_task],
        _single_surface_profile(),
    )

    assert [task["id"] for task in materialized] == ["adv-legacy", "adv-1"]
    assert "delivery_channel" not in materialized[0]
    assert materialized[1]["delivery_channel"]["mechanism"] == "api"

def test_materialize_validated_shard_tasks_appends_delivery_site(monkeypatch):
    plan_task = _plan_task()
    profile = _single_surface_profile()
    profile["injection_surface"][0]["delivery_channels"][0]["delivery_site"] = "shopping_admin"
    monkeypatch.setattr(phase_2_injections, "_voice_registry", lambda: {"dummy": True})
    monkeypatch.setattr(
        phase_2_injections,
        "derive_length_budget",
        lambda task, site_profile, registry: {"min": 20, "max": 400, "source": "test"},
    )

    materialized = phase_2_injections._materialize_validated_shard_tasks([plan_task], profile)

    assert materialized[0]["delivery_channel"]["delivery_site"] == "shopping_admin"
    assert materialized[0]["sites"] == ["shopping", "shopping_admin"]

@pytest.mark.asyncio
async def test_run_shard_with_limit_serializes_work(monkeypatch):
    limiter = asyncio.Semaphore(1)
    state = {"current": 0, "max": 0}

    async def fake_generate(**kwargs):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0)
        state["current"] -= 1
        return phase_2_injections.SiteInjectionResult(kwargs["site_name"], [], [])

    monkeypatch.setattr(phase_2_injections, "_generate_injections_for_site", fake_generate)

    await asyncio.gather(
        phase_2_injections._run_shard_with_limit(
            limiter,
            launch_jitter_seconds=0.0,
            site_name="shopping",
        ),
        phase_2_injections._run_shard_with_limit(
            limiter,
            launch_jitter_seconds=0.0,
            site_name="gitlab",
        ),
    )

    assert state["max"] == 1
