"""Parser and dispatch contracts for the phase and resume command flags."""

from __future__ import annotations

import pytest

from warp_taskgen.cli import build_parser, dispatch, resume
from warp_taskgen.state import save_state

from ._fixtures import _stub_generate_new_tasks_sandbox_preflight  # noqa: F401


def _subparser(parser, name: str):
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and name in choices:
            return choices[name]
    raise AssertionError(f"subparser {name!r} not found")


def test_build_parser_accepts_generate_novel_flag():
    parser = build_parser()

    args = parser.parse_args(["phase", "1", "--generate-novel"])

    assert args.generate_novel is True


def test_build_parser_accepts_phase_0_host_inventory_instances(tmp_path):
    parser = build_parser()
    inventory_path = tmp_path / "instances.scale.json"

    args = parser.parse_args(
        [
            "phase",
            "0",
            "--benchmark",
            "vendors/webarena-verified",
            "--host-inventory-instances",
            str(inventory_path),
        ]
    )

    assert args.host_inventory_instances == inventory_path


def test_build_parser_accepts_novel_tasks_per_site_aliases():
    parser = build_parser()

    args = parser.parse_args(["phase", "1", "--novel-tasks-per-site", "50"])
    alias_args = parser.parse_args(["phase", "1", "--new-tasks-per-site", "24"])

    assert args.novel_tasks_per_site == 50
    assert alias_args.novel_tasks_per_site == 24


def test_build_parser_accepts_phase_1_action_counts():
    parser = build_parser()

    args = parser.parse_args(
        [
            "phase",
            "1",
            "--phase-1-action-counts",
            "create_issue=20,create_post=20,create_issue_note=10",
        ]
    )

    assert args.phase_1_action_counts == "create_issue=20,create_post=20,create_issue_note=10"


def test_build_parser_accepts_phase_1_task_card_plan(tmp_path):
    parser = build_parser()
    plan_path = tmp_path / "task_cards.json"

    args = parser.parse_args(["phase", "1", "--task-card-plan", str(plan_path)])

    assert args.task_card_plan == plan_path


def test_build_parser_accepts_phase_1_task_capability_profile():
    parser = build_parser()

    args = parser.parse_args(["phase", "1", "--task-capability-profile", "tier3_repository_pilot"])

    assert args.task_capability_profile == "tier3_repository_pilot"


def test_build_parser_accepts_sandbox_model_flag_for_phase_3():
    parser = build_parser()

    args = parser.parse_args(
        ["phase", "3", "--instances", "instances.json", "--sandbox-model", "claude-opus-4-6"]
    )

    assert args.sandbox_model == "claude-opus-4-6"


def test_build_parser_rejects_removed_phase_2a_modal_flags():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "phase",
                "2",
                "--phase-2a-runtime",
                "modal",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "phase",
                "2",
                "--phase-2-sandbox-concurrency",
                "3",
            ]
        )


def test_build_parser_accepts_phase_2_text_fill_flags():
    parser = build_parser()

    args = parser.parse_args(
        [
            "phase",
            "2",
            "--phase-2b-texts-per-plan",
            "3",
            "--phase-2-text-fill-concurrency",
            "7",
            "--phase-2-text-model",
            "anthropic/claude-sonnet-4-6",
        ]
    )

    assert args.phase_2b_texts_per_plan == 3
    assert args.phase_2_text_fill_concurrency == 7
    assert args.phase_2_text_model == "anthropic/claude-sonnet-4-6"


def test_build_parser_accepts_phase_2a_action_policy():
    parser = build_parser()

    for policy in (
        "mutation_when_available",
        "mutation_only_when_available",
        "tier2_unaligned_control",
        "tier3_pilot",
        "tier3_unaligned_control",
    ):
        args = parser.parse_args(
            [
                "phase",
                "2",
                "--phase-2a-action-policy",
                policy,
            ]
        )

        assert args.phase_2a_action_policy == policy


def test_phase_2_help_mentions_sequential_2a_2b_stages():
    parser = build_parser()
    help_text = " ".join(_subparser(parser, "phase").format_help().split())
    assert "Phase 2 is one command with two internal model stages" in help_text
    assert "2a host-side API strategy planning, then 2b host-side text fill" in help_text
    assert "there are no separate --phase-2a-only or --phase-2b-only flags" in help_text


def test_resume_help_mentions_phase_2_stage_resume():
    parser = build_parser()
    resume_parser = _subparser(parser, "resume")
    help_text = " ".join(resume_parser.format_help().split())
    description = " ".join((resume_parser.description or "").split())
    assert "re-enters the saved internal sub-stage automatically" in description
    assert "2a planning or 2b text fill" in description
    assert "There are no separate --phase-2a-only or --phase-2b-only flags" in description
    assert "Override the saved Phase 2b text-fill model on resume." in help_text


def test_build_parser_accepts_resume_no_l3_l4_flag():
    parser = build_parser()

    args = parser.parse_args(["resume", "--no-l3-l4"])

    assert args.no_l3_l4 is True


def test_build_parser_accepts_resume_phase_4_timeout_overrides():
    parser = build_parser()

    args = parser.parse_args(
        [
            "resume",
            "--agent-llm-timeout",
            "240",
            "--agent-step-timeout",
            "300",
            "--agent-task-timeout",
            "900",
            "--phase-4-max-workers",
            "5",
        ]
    )

    assert args.agent_llm_timeout == 240
    assert args.agent_step_timeout == 300
    assert args.agent_task_timeout == 900
    assert args.phase_4_max_workers == 5


def test_build_parser_accepts_phase_4_task_timeout_override():
    parser = build_parser()

    args = parser.parse_args(
        ["phase", "4", "--agent-task-timeout", "900", "--phase-4-max-workers", "5"]
    )

    assert args.agent_task_timeout == 900
    assert args.phase_4_max_workers == 5


def test_dispatch_resume_preserves_saved_phase_2_l1_l2_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    save_state(
        "phase_2",
        status="running",
        phase_2_stage="planning",
        phase_2a_resolution_signature={
            "no_l3_l4": True,
            "instances_path": None,
            "instances_sha256": None,
        },
    )

    parser = build_parser()
    args = parser.parse_args(["resume"])
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        dispatch,
        "_install_verification_proxy_from_args",
        lambda synthetic: None,
    )

    def fake_dispatch_phase(synthetic):
        captured["args"] = synthetic
        return 0

    monkeypatch.setattr(dispatch, "_dispatch_phase", fake_dispatch_phase)

    rc = resume._dispatch_resume(args)

    assert rc == 0
    synthetic = captured["args"]
    assert synthetic.no_l3_l4 is True
    assert synthetic.feasibility_instances is None


@pytest.mark.parametrize(
    "argv",
    [
        ["phase", "3", "--max-tasks-per-site", "0"],
        ["phase", "4", "--max-tasks-per-site", "-1"],
        ["resume", "--max-tasks-per-site", "0"],
        ["phase", "2", "--phase-2-sandbox-concurrency", "0"],
        ["resume", "--phase-2-launch-jitter-ms", "0"],
        ["phase", "2", "--phase-2b-texts-per-plan", "0"],
        ["resume", "--phase-2-text-fill-concurrency", "0"],
    ],
)
def test_build_parser_rejects_non_positive_max_tasks_per_site(argv):
    parser = build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(argv)
