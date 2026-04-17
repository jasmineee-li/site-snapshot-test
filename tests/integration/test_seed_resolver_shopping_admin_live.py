from __future__ import annotations

import asyncio

import pytest

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.phases.phase_4_adversarial import preflight_adversarial_seed

pytestmark = pytest.mark.integration


def test_current_shopping_admin_artifact_preflights_against_live_shopping(
    live_instance, phase_2_task
):
    task = phase_2_task("adv_shopping_admin_006")
    instance = live_instance("shopping")
    instance["seed_task"] = task
    errors = acquire_tokens_for_instances([instance])
    assert errors == []

    report = asyncio.run(preflight_adversarial_seed(task["adversarial_data_seed"], instance))

    assert report.ok, [m.message for m in report.mismatches]

