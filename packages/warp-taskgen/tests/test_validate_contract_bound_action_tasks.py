from __future__ import annotations

from scripts.validate_contract_bound_action_tasks import analyze_phase1_contract_bound_tasks


def _task(action_kind: str, *, evidence: dict | None = None) -> dict:
    task = {
        "id": f"novel_{action_kind}",
        "site": "reddit",
        "origin": "new_task",
        "task_card_id": f"card_{action_kind}",
        "route_id": "reddit.comment_body.reddit_submission.create_comment",
        "instruction": "Open the page and leave a public follow-up comment.",
        "task_provenance": {
            "task_card_id": f"card_{action_kind}",
            "benign_reward_shape": "host_action_only",
            "benign_action_contract": {"action_kind": action_kind},
        },
        "reward_function": {"eval": [{"evaluator": "FinalStateEvaluator"}]},
        "contract_bound_generation": {"backend": "contract_bound_api"},
    }
    if evidence is not None:
        task["contract_bound_generation"]["reddit_seed_comment_visibility_anchor_evidence"] = (
            evidence
        )
    return task


def test_submit_comment_requires_explicit_visibility_gate() -> None:
    report = analyze_phase1_contract_bound_tasks([_task("submit_comment")])

    assert report["failed_tasks"] == 1
    assert report["failures"][0]["failures"] == ["submit_comment_requires_explicit_visibility_gate"]


def test_submit_comment_accepts_empty_carrier_visibility_gate() -> None:
    report = analyze_phase1_contract_bound_tasks(
        [
            _task(
                "submit_comment",
                evidence={
                    "source": "phase0c_empty_submission_inventory",
                    "existing_comment_count": 0,
                    "max_existing_comments_for_comment_seed": 0,
                },
            )
        ]
    )

    assert report["failed_tasks"] == 0
