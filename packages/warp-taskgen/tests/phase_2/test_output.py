# ruff: noqa
# Auto-split from tests/test_phase_2_injections.py; shared helpers live in tests/phase_2/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401
from warp_taskgen.phase_2 import output as phase_2_output
from warp_taskgen.phase_2.phase_2c import stage as phase_2_stage
from warp_taskgen.phase_2.phase_2c import artifacts as phase_2c_artifacts
from warp_taskgen.phase_2.phase_2c import config as phase_2c_config
from warp_taskgen.phase_2.phase_2c.types import FeasibilityReport
from warp_taskgen.phase_2 import eligibility
from warp_taskgen.phase_2 import target_stage


def test_merge_immutable_fields_redacts_agent_context_secrets():
    benign = _benign_task()
    benign["agent_context"] = {
        "authentication": {
            "credentials": {"username": "admin", "password": "admin1234"},
        },
        "auth_mechanism": {
            "type": "http_headers",
            "headers": {"X-M2-Admin-Auto-Login": "admin:admin1234"},
        },
        "description": "Use header X-M2-Admin-Auto-Login: admin:admin1234 for login.",
    }
    adv = {"id": "adv-1", "benign_task_id": benign["id"], "adversarial_reward": {"type": "noop"}}

    generation._merge_immutable_fields([adv], [benign])

    assert adv["agent_context"]["authentication"]["credentials"] == {
        "username": "<redacted>",
        "password": "<redacted>",
    }
    assert adv["agent_context"]["auth_mechanism"]["headers"] == {
        "X-M2-Admin-Auto-Login": "<redacted>"
    }
    assert "admin1234" not in adv["agent_context"]["description"]
    assert "<redacted>" in adv["agent_context"]["description"]


def test_merge_immutable_fields_redacts_email_password_examples_in_prose():
    benign = _benign_task()
    benign["agent_context"] = {
        "auth_mechanism": {
            "headers": {"X-M2-Customer-Auto-Login": "emma.lopez@gmail.com:Password.123"},
        },
        "description": (
            "Set the header value to 'email:password' format "
            "(e.g., 'emma.lopez@gmail.com:Password.123') on the browser context."
        ),
    }
    adv = {"id": "adv-1", "benign_task_id": benign["id"], "adversarial_reward": {"type": "noop"}}

    generation._merge_immutable_fields([adv], [benign])

    assert "emma.lopez@gmail.com:Password.123" not in adv["agent_context"]["description"]
    assert "email:password" in adv["agent_context"]["description"]
    assert "<redacted>" in adv["agent_context"]["description"]


def test_merge_immutable_fields_redacts_sensitive_benign_data_seed():
    benign = _benign_task()
    benign["data_seed"] = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "POST",
                "path": "/rest/V1/reviews",
                "body": {
                    "detail": "payload",
                    "token": "super-secret-token",
                },
                "headers": {"Authorization": "Bearer very-secret"},
            }
        ],
    }
    adv = {"id": "adv-1", "benign_task_id": benign["id"], "adversarial_reward": {"type": "noop"}}

    generation._merge_immutable_fields([adv], [benign])

    copied_seed = adv["data_seed"]
    assert copied_seed["api_calls"][0]["body"]["token"] == "<redacted>"
    assert copied_seed["api_calls"][0]["headers"]["Authorization"] == "<redacted>"


def test_merge_immutable_fields_copies_contract_bound_generation_metadata():
    benign = _benign_task()
    benign["contract_bound_generation"] = {
        "abstract_purpose_tag": "bug_followup",
        "style_tag": "bug_report",
    }
    benign["task_card_id"] = "gitlab_issue_description_public_followup_issue_paper"
    adv = {"id": "adv-1", "benign_task_id": benign["id"], "adversarial_reward": {"type": "noop"}}

    generation._merge_immutable_fields([adv], [benign])

    assert adv["contract_bound_generation"] == {
        "abstract_purpose_tag": "bug_followup",
        "style_tag": "bug_report",
    }
    assert adv["task_card_id"] == "gitlab_issue_description_public_followup_issue_paper"


def test_sanitize_task_for_output_redacts_data_seed_and_cookie_session_fields():
    task = _benign_task()
    task["data_seed"] = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "POST",
                "path": "/rest/V1/reviews",
                "body": {"detail": "payload"},
                "headers": {"Authorization": "Bearer very-secret"},
            }
        ],
    }
    task["agent_context"] = {
        "auth_mechanism": {
            "cookies": {"session": "cookie-secret"},
            "session_cookie": "session-secret",
        }
    }

    sanitized = phase_2_output._sanitize_task_for_output(task)

    assert sanitized["data_seed"]["api_calls"][0]["headers"]["Authorization"] == "<redacted>"
    assert sanitized["agent_context"]["auth_mechanism"]["cookies"] == {"session": "<redacted>"}
    assert sanitized["agent_context"]["auth_mechanism"]["session_cookie"] == "<redacted>"


def test_write_dropped_source_data_sidecar_clears_full_run_stale_records(tmp_path):
    path = tmp_path / "adversarial_tasks.dropped_source_data.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "old",
                    "site": "gitlab",
                    "source_data_issue": {"kind": "not_found"},
                }
            ]
        )
    )

    phase_2_stage._write_dropped_source_data_sidecar(path, [], sites_filter=None)

    assert json.loads(path.read_text()) == []


def test_write_dropped_source_data_sidecar_preserves_unfiltered_sites(tmp_path):
    path = tmp_path / "adversarial_tasks.dropped_source_data.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "old-gitlab",
                    "site": "gitlab",
                    "source_data_issue": {"kind": "not_found"},
                },
                {
                    "id": "old-reddit",
                    "site": "reddit",
                    "source_data_issue": {"kind": "gone"},
                },
            ]
        )
    )
    replacement = [
        {
            "id": "new-gitlab",
            "site": "gitlab",
            "source_data_issue": {"kind": "forbidden"},
        }
    ]

    merged = phase_2_stage._write_dropped_source_data_sidecar(
        path,
        replacement,
        sites_filter={"gitlab"},
    )

    records = json.loads(path.read_text())
    assert [record["id"] for record in records] == ["old-reddit", "new-gitlab"]
    assert merged == records


def test_write_dropped_source_data_sidecar_dedupes_by_site_and_id(tmp_path):
    path = tmp_path / "adversarial_tasks.dropped_source_data.json"
    duplicate = {
        "id": "same-id",
        "site": "gitlab",
        "source_data_issue": {"kind": "not_found"},
    }

    merged = phase_2_stage._write_dropped_source_data_sidecar(
        path,
        [duplicate, dict(duplicate)],
        sites_filter=None,
    )

    assert merged == [duplicate]
    assert json.loads(path.read_text()) == [duplicate]


def test_phase_2c_helpers_have_canonical_feature_owners():
    assert (
        phase_2c_config._feasibility_status({"feasibility": {"status": "verified"}}) == "verified"
    )
    for helper in (
        "_merged_dropped_source_data",
        "_phase_2c_report_summary_with_artifacts",
        "_validate_phase_2c_artifact_payloads",
        "_count_feasibility_status",
        "_count_idempotency_skipped",
        "_source_data_dropped_by_kind",
        "_phase_2c_per_site_counts",
    ):
        assert callable(getattr(phase_2c_artifacts, helper))
    for helper in (
        "_validate_phase_2c_instance_record",
        "_normalize_instance_record",
        "_benchmark_values_from_seed",
    ):
        assert callable(getattr(phase_2c_config, helper))
    for helper in ("_sanitize_agent_context_node", "_collect_agent_context_secrets"):
        assert callable(getattr(phase_2_output, helper))


def test_report_summary_can_count_merged_dropped_source_data():
    report = FeasibilityReport(
        verified=[],
        infeasible=[],
        skipped_already_verified=[],
        cleanup_warnings=[],
        host_fingerprint={},
        elapsed_seconds=0.0,
        per_site_counts={},
        dropped_source_data=[
            {"id": "current", "source_data_issue": {"kind": "not_found"}},
        ],
    )
    merged = [
        {"id": "preserved", "source_data_issue": {"kind": "gone"}},
        {"id": "current", "source_data_issue": {"kind": "not_found"}},
    ]

    summary = target_stage._report_summary_dict(
        report,
        instances_path="instances.scale.json",
        dropped_source_data=merged,
    )

    assert summary["source_data_dropped_count"] == 2
    assert summary["source_data_dropped_by_kind"] == {"gone": 1, "not_found": 1}


def test_dropped_source_sidecar_observes_facade_merge_monkeypatch(monkeypatch, tmp_path):
    path = tmp_path / "adversarial_tasks.dropped_source_data.json"

    def merged_sidecar(_path, _dropped_source_data, *, sites_filter):
        assert sites_filter == {"gitlab"}
        return [
            {
                "id": "patched",
                "site": "gitlab",
                "source_data_issue": {"kind": "not_found"},
            }
        ]

    monkeypatch.setattr(
        phase_2_stage,
        "_merged_dropped_source_data",
        merged_sidecar,
    )

    merged = phase_2_stage._write_dropped_source_data_sidecar(
        path,
        [],
        sites_filter={"gitlab"},
    )

    assert [record["id"] for record in merged] == ["patched"]
    assert json.loads(path.read_text()) == merged


@pytest.mark.asyncio
async def test_generate_injections_for_site_api_path_sanitizes_prompt_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    profile_path = tmp_path / "BENCHMARK_PROFILE_shopping.json"
    profile_path.write_text(json.dumps(_single_surface_profile()))
    agent_context_path = tmp_path / "AGENT_CONTEXT_shopping.json"
    agent_context_path.write_text(
        json.dumps(
            {
                "authentication": {
                    "credentials": {"username": "alice", "password": "secret-pass"},
                },
                "auth_mechanism": {
                    "headers": {"X-Test-Auto-Login": "alice:secret-pass"},
                },
            }
        )
    )

    benign = _benign_task()
    benign["agent_context"] = {
        "authentication": {
            "credentials": {"username": "alice", "password": "secret-pass"},
        }
    }
    benign["data_seed"] = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "POST",
                "path": "/rest/V1/reviews",
                "headers": {"Authorization": "Bearer very-secret"},
                "body": {"detail": "payload"},
            }
        ],
    }
    captured: dict[str, Any] = {}

    async def fake_generate_phase_2a_plans_api(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        runner_api,
        "generate_phase_2a_plans_api",
        fake_generate_phase_2a_plans_api,
    )
    monkeypatch.setattr(
        eligibility,
        "_phase_2a_eligible_tasks",
        lambda site_tasks, benign_target_resources, site_name: (site_tasks, []),
    )

    result = await generation._generate_injections_for_site(
        site_name="shopping",
        site_tasks=[benign],
        all_site_tasks=[benign],
        profile_path=profile_path,
        label="shopping",
        sandbox_model="claude-sonnet-4-6",
        instance=None,
    )

    assert result.adversarial_tasks == []
    assert captured["benign_tasks"][0]["agent_context"]["authentication"]["credentials"] == {
        "username": "<redacted>",
        "password": "<redacted>",
    }
    assert captured["agent_context"]["auth_mechanism"]["headers"] == {
        "X-Test-Auto-Login": "<redacted>"
    }
    assert (
        captured["benign_tasks"][0]["data_seed"]["api_calls"][0]["headers"]["Authorization"]
        == "<redacted>"
    )


@pytest.mark.asyncio
async def test_generate_injections_for_site_api_path_sanitizes_agent_context_cookies(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    profile_path = tmp_path / "BENCHMARK_PROFILE_shopping.json"
    profile_path.write_text(json.dumps(_single_surface_profile()))
    agent_context_path = tmp_path / "AGENT_CONTEXT_shopping.json"
    agent_context_path.write_text(
        json.dumps(
            {
                "authentication": {
                    "credentials": {"username": "alice", "password": "secret-pass"},
                },
                "auth_mechanism": {
                    "cookies": {"session": "cookie-secret"},
                    "headers": {"X-Test-Auto-Login": "alice:secret-pass"},
                },
            }
        )
    )

    benign = _benign_task()
    benign["agent_context"] = {
        "authentication": {
            "credentials": {"username": "alice", "password": "secret-pass"},
        }
    }
    benign["data_seed"] = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "POST",
                "path": "/rest/V1/reviews",
                "headers": {"Authorization": "Bearer very-secret"},
                "body": {"detail": "payload"},
            }
        ],
    }
    captured: dict[str, Any] = {}

    async def fake_generate_phase_2a_plans_api(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        runner_api,
        "generate_phase_2a_plans_api",
        fake_generate_phase_2a_plans_api,
    )
    monkeypatch.setattr(
        eligibility,
        "_phase_2a_eligible_tasks",
        lambda site_tasks, benign_target_resources, site_name: (site_tasks, []),
    )

    await generation._generate_injections_for_site(
        site_name="shopping",
        site_tasks=[benign],
        all_site_tasks=[benign],
        profile_path=profile_path,
        label="shopping",
        sandbox_model="claude-sonnet-4-6",
        instance=None,
    )

    assert captured["benign_tasks"][0]["agent_context"]["authentication"]["credentials"] == {
        "username": "<redacted>",
        "password": "<redacted>",
    }
    assert (
        captured["benign_tasks"][0]["data_seed"]["api_calls"][0]["headers"]["Authorization"]
        == "<redacted>"
    )
    assert captured["agent_context"]["auth_mechanism"]["cookies"] == {"session": "<redacted>"}
    assert captured["agent_context"]["auth_mechanism"]["headers"] == {
        "X-Test-Auto-Login": "<redacted>"
    }


def test_merge_preserving_unfiltered_sites_drops_quarantined_map_entries(tmp_path):
    path = tmp_path / "adversarial_tasks.json"
    path.write_text(
        json.dumps(
            [
                {"id": "map-1", "site": "map"},
                {"id": "shopping-1", "site": "shopping"},
            ]
        ),
        encoding="utf-8",
    )

    merged = phase_2_output._merge_preserving_unfiltered_sites(
        path,
        [{"id": "gitlab-1", "site": "gitlab"}],
        sites_filter={"gitlab"},
    )

    assert [item["id"] for item in merged] == ["shopping-1", "gitlab-1"]
