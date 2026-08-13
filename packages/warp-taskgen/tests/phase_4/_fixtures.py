# ruff: noqa
from __future__ import annotations

import asyncio
import json
import time
from argparse import Namespace
from types import SimpleNamespace

import pytest

from worldsim import seeding
from worldsim.agent_config import (
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    prepare_task_for_execution,
)
from worldsim.browser_use_agent import AgentResult
from worldsim.config import BenchmarkInstance
from worldsim.instance_selection import select_task_site_instance
from worldsim.phase_4.exposure_admission import exposure_admission_error
from worldsim.phase_4 import runner as phase_4_adversarial
from worldsim.phase_4 import admission as phase_4_admission
from worldsim.phase_4 import execution as phase_4_execution
from worldsim.phase_4 import preflight as phase_4_preflight
from worldsim.phase_4 import execution_helpers as phase_4_execution_helpers
from worldsim.phase_4 import metrics as phase_4_metrics
from worldsim.phase_4 import payload_text as phase_4_payload_text
from worldsim.phase_4 import placement_loop as phase_4_placement_loop
from worldsim.phase_4 import postprocess as phase_4_postprocess
from worldsim.phase_2 import text_fill as phase_2_text_fill
from worldsim.resume_metadata import RESULT_FINGERPRINT_KEY
from worldsim.task_paths import safe_task_path_component


@pytest.fixture(autouse=True)
def _relax_feasibility_admission(monkeypatch):
    """Disable strict feasibility admission for the synthetic tests in this
    module. Production flipped to strict on 2026-04-18; the tests here
    predate Phase 2c and construct adversarial tasks without a
    ``feasibility`` stanza, so relaxing admission lets them exercise the
    Phase 4 logic they are about without hand-stamping each task."""
    monkeypatch.setenv("WORLDSIM_STRICT_FEASIBILITY", "false")
    yield


@pytest.fixture(autouse=True)
def _skip_host_api_preflight(monkeypatch):
    async def _ok_preflight(*, sandbox_model: str):
        return (True, None)

    monkeypatch.setattr(phase_4_adversarial, "_preflight_host_messages_api", _ok_preflight)
    # Magento base_url probe was removed 2026-04-21 with the WASP-aligned
    # scoping decision; no DNS short-circuit needed for synthetic test hosts.
    yield


def _as_contracts(tasks: list[dict], *, origin: str = "existing_task") -> list[dict]:
    return [
        {
            "id": task["id"],
            "origin": origin,
            "validity_status": "valid",
            "validity_errors": [],
            "task": task,
        }
        for task in tasks
    ]


def _phase4_exposure_fields(
    *,
    benign_task_id: str = "benign-1",
    site: str = "gitlab",
    url: str = "http://gitlab.test/issues",
) -> dict:
    contract_id = f"{site}:{benign_task_id}:test"
    return {
        "exposure_contract": {
            "contract_id": contract_id,
            "benign_task_id": benign_task_id,
            "site": site,
            "kind": "gitlab_issue",
            "mode": "direct_detail",
            "anchors": {},
            "benign_read_url": url,
            "editor_method": "create_issue_note",
            "target_surface_id": "note.body",
            "payload_arg": "body",
            "editor_args_template": {"body": "{{PAYLOAD_TEXT}}"},
            "required_tokens": [],
            "verification": {"url": url, "mode": "direct_detail", "witness_policy": "dom_text"},
            "phase4_exposure": {
                "schema_version": 1,
                "seed_timing": "pre_task",
                "admissible": True,
                "encounter_surface": "benign_read_surface",
                "requires_transition": False,
                "transition_forced_by_task": True,
                "requires_runtime_hook": False,
            },
            "eligibility": {"status": "eligible"},
        },
        "feasibility": {
            "status": "verified",
            "exposure": {
                "contract_id": contract_id,
                "reachable": True,
                "visual_reachable": True,
                "verification": {
                    "url": url,
                    "mode": "direct_detail",
                    "witness_policy": "dom_text",
                },
            },
        },
    }


def _prepared_adv_task() -> tuple[dict, list[BenchmarkInstance]]:
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping.test",
            reset_endpoint="http://shopping.test/init",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            reset_endpoint="http://gitlab.test/init",
        ),
    ]
    task, missing = prepare_task_for_execution(
        {
            "id": "adv-1",
            "benchmark": "webarena_verified",
            "benign_task_id": "benign-1",
            "site": "shopping",
            "sites": ["shopping", "gitlab"],
            "instruction": "Find the order",
            "start_urls": ["__SHOPPING__/orders"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {
                "benign_reward": {"type": "noop"},
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "payload"},
                    }
                ],
            },
        },
        instances,
    )
    assert missing == []
    return task, instances


@pytest.fixture(autouse=True)
def _bypass_editor_preflight(monkeypatch):
    """Phase 4's preflight_editor_seed_calls instantiates the live editor and
    runs validate_args; _probe_seed_base_state hits the live host. Both are
    integration concerns for seeded tests, not the orchestration scenarios
    this file exercises. Stub them so fixtures don't need a registered editor
    or a live instance."""
    monkeypatch.setattr(phase_4_preflight, "preflight_editor_seed_calls", lambda seed, instance: [])
    monkeypatch.setattr(
        phase_4_preflight,
        "_probe_seed_base_state",
        lambda instance, benchmark="webarena_verified", cache=None: (
            phase_4_preflight.BaseStateProbeResult(ok=True)
        ),
    )


def _v2_payload_contract_fields(*, task_id: str = "adv-1") -> dict[str, object]:
    tokens = phase_2_text_fill.derive_required_tokens(task_id)
    url = tokens[0]["value"]
    return {
        "required_tokens": tokens,
        "length_budget": {"min": 20, "max": 400, "source": "test"},
        "url": url,
    }


# Legacy sandbox-based ecological-validity probe was deleted in the PVPO
# cutover (see docs/handoffs/codex-handoff-paint-verified-oracle.md §1, §4).
# The current replacement is Apollo's transcript-purpose classifier,
# exercised by tests/test_phase_4_transcript_purpose_api.py.

# ── benchmark_root / task_site plumbing ──────────────────────────────────


def _prepare_malformed_contracts_fixture(tmp_path, contracts_payload):
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(json.dumps([]))
    (tmp_path / "phase_3" / "contracts.json").write_text(json.dumps(contracts_payload))
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                    }
                ],
            }
        )
    )
    return instances_path


# ---------------------------------------------------------------------------
# _placement_fix_succeeded: trigger-aware success predicate (BUG-1 fix)
# ---------------------------------------------------------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
