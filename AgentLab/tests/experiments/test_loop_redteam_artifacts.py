from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import logging

import agentlab.experiments.loop as loop_module
import agentlab.benchmarks.redteam.app_artifacts as app_artifacts_module
from agentlab.benchmarks.redteam.config import RedteamEnvArgs
from agentlab.experiments.loop import ExpArgs, _move_old_exp, get_exp_result, yield_all_exp_results


def test_collect_redteam_artifacts_persists_final_state(tmp_path: Path) -> None:
    env_args = RedteamEnvArgs(
        task_name="task",
        target="target",
        attacker_model="",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="benign",
        doc="goal",
        app_dir=str(tmp_path / "app"),
    )
    exp_args = ExpArgs(agent_args=SimpleNamespace(agent_name="dummy"), env_args=env_args)
    exp_args.exp_dir = tmp_path / "exp"
    exp_args.exp_dir.mkdir()
    (exp_args.exp_dir / "server_events.log").write_text("{}", encoding="utf-8")

    class DummyEnv:
        def get_browser_state(self):
            return {"status": "done"}

    artifact_info = exp_args._collect_redteam_artifacts(DummyEnv())

    final_state_path = Path(artifact_info["final_state_path"])
    assert final_state_path.exists()
    assert json.loads(final_state_path.read_text(encoding="utf-8")) == {"status": "done"}
    assert artifact_info["server_events_path"].endswith("server_events.log")


def test_collect_redteam_artifacts_retries_final_state_fetch(tmp_path: Path, monkeypatch) -> None:
    env_args = RedteamEnvArgs(
        task_name="task",
        target="target",
        attacker_model="",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="benign",
        doc="goal",
        app_dir=str(tmp_path / "app"),
    )
    exp_args = ExpArgs(agent_args=SimpleNamespace(agent_name="dummy"), env_args=env_args)
    exp_args.exp_dir = tmp_path / "exp"
    exp_args.exp_dir.mkdir()
    (exp_args.exp_dir / "server_events.log").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_args, **_kwargs: None)

    class DummyEnv:
        def __init__(self):
            self.calls = 0

        def get_browser_state(self):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary miss")
            return {"status": "done"}

    env = DummyEnv()
    artifact_info = exp_args._collect_redteam_artifacts(env)

    final_state_path = Path(artifact_info["final_state_path"])
    assert env.calls == 3
    assert final_state_path.exists()
    assert json.loads(final_state_path.read_text(encoding="utf-8")) == {"status": "done"}


def test_collect_redteam_artifacts_redacts_sensitive_final_state(tmp_path: Path) -> None:
    env_args = RedteamEnvArgs(
        task_name="task",
        target="target",
        attacker_model="",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="benign",
        doc="goal",
        app_dir=str(tmp_path / "app"),
    )
    exp_args = ExpArgs(agent_args=SimpleNamespace(agent_name="dummy"), env_args=env_args)
    exp_args.exp_dir = tmp_path / "exp"
    exp_args.exp_dir.mkdir()
    (exp_args.exp_dir / "server_events.log").write_text("{}", encoding="utf-8")

    class DummyEnv:
        def get_browser_state(self):
            return {
                "status": "done",
                "password": "TEST_SECRET_123",
                "emails": [{"subject": "Hello", "body": "sensitive body"}],
            }

    artifact_info = exp_args._collect_redteam_artifacts(DummyEnv())

    final_state_path = Path(artifact_info["final_state_path"])
    saved = json.loads(final_state_path.read_text(encoding="utf-8"))
    assert saved["status"] == "done"
    assert saved["password"] == "[REDACTED]"
    assert saved["emails"][0]["body"] == "[REDACTED]"


def test_run_judge_evaluation_passes_live_server_port(tmp_path: Path, monkeypatch) -> None:
    env_args = RedteamEnvArgs(
        task_name="task",
        target="target",
        attacker_model="",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="benign",
        doc="goal",
        app_dir=str(tmp_path / "app"),
        judging={"submit": {"panel": []}},
    )
    exp_args = ExpArgs(agent_args=SimpleNamespace(agent_name="dummy"), env_args=env_args)
    exp_args.exp_dir = tmp_path / "exp"
    exp_args.exp_dir.mkdir()

    seen: dict[str, object] = {}

    class FakeJudgeRunner:
        def __init__(self, env_args_param):
            seen["env_args"] = env_args_param

        def evaluate(self, exp_dir, episode_info, server_port=None):
            seen["exp_dir"] = Path(exp_dir)
            seen["episode_info"] = episode_info
            seen["server_port"] = server_port
            return {"overall_success": True, "overall_score": 1.0}

        def save_result(self, exp_dir, result):
            seen["saved_exp_dir"] = Path(exp_dir)
            seen["saved_result"] = result

    monkeypatch.setattr(loop_module, "JudgeRunner", FakeJudgeRunner)

    class DummyEnv:
        server_port = 4321

    exp_args._run_judge_evaluation([], env=DummyEnv())

    assert seen["server_port"] == 4321
    assert seen["exp_dir"] == exp_args.exp_dir
    assert seen["saved_exp_dir"] == exp_args.exp_dir


def test_resolve_judge_runner_imports_redteam_judge_lazily() -> None:
    resolved = loop_module._resolve_judge_runner()

    assert resolved is not None
    assert resolved.__name__ == "JudgeRunner"


def test_save_experiment_results_archives_attack_metadata_and_records_summary_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_args = RedteamEnvArgs(
        task_name="task",
        target="target",
        attacker_model="",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="adversarial",
        doc="goal",
        app_dir=str(tmp_path / "app"),
        behavior_id="demo",
        variant="adversarial_demo_v0",
    )
    exp_args = ExpArgs(agent_args=SimpleNamespace(agent_name="dummy"), env_args=env_args)
    exp_args.exp_dir = tmp_path / "exp"
    exp_args.exp_dir.mkdir()
    (exp_args.exp_dir / "server_events.log").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        app_artifacts_module,
        "load_attack_metadata",
        lambda *args, **kwargs: {
            "test_mode": "adversarial",
            "metadata_source": "behavior_contract+variant_diff",
            "injection_locations": ["/mail/u/0/#/settings"],
            "placeholders": {"EMAILS:adv_e1": {"adversarial": ["archived"]}},
            "adversarial_content": {"EMAILS": [{"id": "adv_e1"}]},
            "attack_target_arrays": ["EMAILS"],
            "surface_summaries": ["/mail/u/0/#/settings - Settings"],
            "exposure_known": True,
        },
    )

    class DummyEnv:
        def get_browser_state(self):
            return {"status": "done"}

    exp_args._save_experiment_results(DummyEnv(), None, [], None, None)

    archive_path = exp_args.exp_dir / "attack_metadata.json"
    summary_path = exp_args.exp_dir / "summary_info.json"
    assert archive_path.exists()
    archived_payload = json.loads(archive_path.read_text(encoding="utf-8"))
    assert archived_payload["metadata_source"] == "behavior_contract+variant_diff"
    assert archived_payload["archive_context"]["behavior_id"] == "demo"
    assert archived_payload["archive_context"]["variant"] == "adversarial_demo_v0"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["attack_metadata_path"] == str(archive_path)


def test_run_adversarial_unsets_logging_file_handler(tmp_path: Path, monkeypatch) -> None:
    class DummyAgent:
        def __init__(self):
            self.action_set = SimpleNamespace(to_python_code=lambda *_args, **_kwargs: None)
            self.obs_preprocessor = None
            self.closed = False

        def set_task_name(self, _task_name):
            return None

        def close(self):
            self.closed = True

    class DummyEnv:
        def __init__(self):
            self.closed = False

        def reset(self, seed=0):
            return {}, {}

        def close(self):
            self.closed = True

    env_args = RedteamEnvArgs(
        task_name="task",
        target="target",
        attacker_model="",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="adversarial",
        doc="goal",
        app_dir=str(tmp_path / "app"),
    )
    env = DummyEnv()
    target_agent = DummyAgent()
    attacker_agent = DummyAgent()
    monkeypatch.setattr(env_args, "make_env", lambda **_kwargs: env)
    exp_args = ExpArgs(
        agent_args=SimpleNamespace(agent_name="dummy", make_agent=lambda: target_agent),
        env_args=env_args,
        attacker_agent_args=SimpleNamespace(make_agent=lambda: attacker_agent),
    )
    exp_args.exp_dir = tmp_path / "exp"
    exp_args.exp_dir.mkdir()
    monkeypatch.setattr(
        exp_args,
        "_run_single_turn_adversarial",
        lambda *args, **kwargs: [],
    )

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    exp_args._set_logger()
    file_handler = exp_args.logging_file_handler
    assert file_handler in root_logger.handlers

    try:
        exp_args._run_adversarial()

        assert file_handler not in root_logger.handlers
        assert target_agent.closed is True
        assert attacker_agent.closed is True
        assert env.closed is True
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)


def test_yield_all_exp_results_detects_nested_redteam_variant_groups(tmp_path: Path) -> None:
    parent_dir = tmp_path / "study" / "redteam" / "demo" / "agent-a" / "seed0_var0"
    benign_dir = parent_dir / "benign"
    adversarial_dir = parent_dir / "adversarial_demo_v0"
    benign_dir.mkdir(parents=True, exist_ok=True)
    adversarial_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "metadata.json").write_text(
        json.dumps(
            {
                "layout_kind": "redteam_variant_group",
                "layout_version": 2,
                "behavior_id": "demo",
                "agent_name": "agent-a",
                "run_group_id": "seed0_var0",
                "variants": ["benign", "adversarial_demo_v0"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for variant_dir, test_mode in ((benign_dir, "benign"), (adversarial_dir, "adversarial")):
        env_args = RedteamEnvArgs(
            task_name=f"redteam.demo.{variant_dir.name}",
            target="target",
            attacker_model="",
            pages=[],
            attack_type="direct",
            adversarial_mode="single_turn",
            test_mode=test_mode,
            doc="goal",
            behavior_id="demo",
            variant=variant_dir.name,
            active_variant="adversarial_demo_v0",
            parent_exp_dir=str(parent_dir),
            variant_name=variant_dir.name,
            run_group_id="seed0_var0",
            is_variant_run=True,
        )
        exp_args = ExpArgs(agent_args=SimpleNamespace(agent_name="dummy"), env_args=env_args)
        exp_args.exp_dir = variant_dir
        with open(variant_dir / "exp_args.pkl", "wb") as f:
            pickle.dump(exp_args, f)

    results = list(yield_all_exp_results(tmp_path / "study", progress_fn=None, use_cache=False))

    assert {result.exp_dir.name for result in results} == {"benign", "adversarial_demo_v0"}
    adversarial_result = get_exp_result(adversarial_dir)
    assert adversarial_result.is_variant_run is True
    assert adversarial_result.parent_dir == parent_dir
    assert adversarial_result.variant_name == "adversarial_demo_v0"


def test_move_old_exp_uses_unique_hidden_backup_names(tmp_path: Path) -> None:
    exp_dir = tmp_path / "seed0_var0"
    exp_dir.mkdir()
    (tmp_path / "_seed0_var0").mkdir()
    (tmp_path / "_seed0_var0_1").mkdir()

    _move_old_exp(exp_dir)

    assert not exp_dir.exists()
    assert (tmp_path / "_seed0_var0_2").exists()


def test_get_exp_result_cache_refreshes_when_exp_dir_is_reused(tmp_path: Path) -> None:
    exp_dir = tmp_path / "seed0_var0"
    env_args = RedteamEnvArgs(
        task_name="task",
        target="target",
        attacker_model="",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="benign",
        doc="goal",
    )
    exp_args = ExpArgs(agent_args=SimpleNamespace(agent_name="dummy"), env_args=env_args)

    exp_dir.mkdir()
    with open(exp_dir / "exp_args.pkl", "wb") as f:
        pickle.dump(exp_args, f)
    (exp_dir / "summary_info.json").write_text(
        json.dumps({"terminated": True, "cum_reward": 1}),
        encoding="utf-8",
    )

    cached = get_exp_result(exp_dir)
    assert cached.summary_info["cum_reward"] == 1

    _move_old_exp(exp_dir)

    exp_dir.mkdir()
    with open(exp_dir / "exp_args.pkl", "wb") as f:
        pickle.dump(exp_args, f)
    (exp_dir / "summary_info.json").write_text(
        json.dumps({"terminated": True, "cum_reward": 2}),
        encoding="utf-8",
    )

    refreshed = get_exp_result(exp_dir)
    assert refreshed.summary_info["cum_reward"] == 2
