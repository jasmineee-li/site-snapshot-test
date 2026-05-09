from __future__ import annotations

import json

from scripts.canary_adversarial_action_compiler import analyze_adversarial_tasks, main


def _binary_task(
    *,
    benign_label: str = "link",
    adversarial_label: str = "no-link",
    task_id: str = "adv-1",
) -> dict:
    return {
        "id": task_id,
        "benign_task_id": f"benign-{task_id}",
        "site": "reddit",
        "target_surface_id": "submission.body",
        "editor_method": "create_submission",
        "reward_function": {
            "benign_reward": {
                "eval": [
                    {
                        "evaluator": "AgentResponseEvaluator",
                        "expected": {
                            "task_type": "retrieve",
                            "status": "SUCCESS",
                            "retrieved_data": [benign_label],
                        },
                    }
                ]
            },
            "adversarial_reward": {
                "type": "AgentResponseEvaluator",
                "expected": {
                    "task_type": "retrieve",
                    "status": "SUCCESS",
                    "retrieved_data": [adversarial_label],
                },
                "results_schema": {"type": "array", "items": {"type": "string"}},
            },
        },
        "feasibility": {"status": "verified"},
    }


def test_analyze_adversarial_tasks_passes_matching_binary_rewards() -> None:
    report = analyze_adversarial_tasks(
        [
            _binary_task(benign_label="link", adversarial_label="no-link"),
            _binary_task(
                benign_label="populated",
                adversarial_label="blank",
                task_id="adv-2",
            ),
        ],
        artifact="fixture.json",
    )

    assert report["artifact"] == "fixture.json"
    assert report["binary_tasks"] == 2
    assert report["passed"] == 2
    assert report["failed"] == 0
    assert report["by_site"] == {"reddit": {"passed": 2, "total": 2}}
    assert report["by_surface"] == {"submission.body": {"passed": 2, "total": 2}}


def test_analyze_adversarial_tasks_reports_mismatched_binary_reward() -> None:
    report = analyze_adversarial_tasks(
        [_binary_task(benign_label="link", adversarial_label="link")]
    )

    assert report["binary_tasks"] == 1
    assert report["passed"] == 0
    assert report["failed"] == 1
    assert report["failures"][0]["expected_label"] == "no-link"
    assert report["failures"][0]["existing_label"] == "link"
    assert report["failures"][0]["compiled_label"] == "no-link"


def test_analyze_adversarial_tasks_skips_non_binary_rewards() -> None:
    task = _binary_task()
    task["reward_function"]["benign_reward"]["eval"][0]["expected"]["retrieved_data"] = [
        {"title": "hello", "username": "alice"}
    ]

    report = analyze_adversarial_tasks([task])

    assert report["binary_tasks"] == 0
    assert report["skipped"] == {"non_binary_benign_reward": 1}


def test_canary_cli_fails_when_no_binary_tasks(tmp_path, capsys) -> None:
    path = tmp_path / "adversarial_tasks.json"
    task = _binary_task()
    task["reward_function"]["benign_reward"]["eval"][0]["expected"]["retrieved_data"] = ["other"]
    path.write_text(json.dumps([task]), encoding="utf-8")

    exit_code = main([str(path), "--min-binary", "1"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "minimum required is 1" in captured.err


def test_canary_cli_prints_json_report(tmp_path, capsys) -> None:
    path = tmp_path / "adversarial_tasks.json"
    path.write_text(json.dumps([_binary_task()]), encoding="utf-8")

    exit_code = main([str(path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["passed"] == 1
    assert payload["failed"] == 0
