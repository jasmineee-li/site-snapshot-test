import json
from pathlib import Path
from types import SimpleNamespace

from agentlab.benchmarks.redteam.config import RedteamEnvArgs
from agentlab.experiments.study import Study, _redteam_variant_sort_key


def test_redteam_variant_sort_key_orders_namespaced_rounds_numerically() -> None:
    variants = [
        "adversarial_demo_v10",
        "adversarial_demo_v2",
        "benign",
        "adversarial_demo_v1",
        "adversarial_other_v0",
    ]

    assert sorted(variants, key=_redteam_variant_sort_key) == [
        "benign",
        "adversarial_demo_v1",
        "adversarial_demo_v2",
        "adversarial_demo_v10",
        "adversarial_other_v0",
    ]


class _DummyAgentArgs:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def set_benchmark(self, benchmark, demo_mode: bool) -> None:
        return None


class _DummyBenchmark:
    name = "redteam"

    def __init__(self, env_args_list):
        self.env_args_list = env_args_list

    def dependency_graph_over_tasks(self):
        return {}

    def prepare_backends(self) -> None:
        return None


def _make_env_args(*, test_mode: str, task_name: str, active_variant: str = "") -> RedteamEnvArgs:
    return RedteamEnvArgs(
        task_name=task_name,
        target="target",
        attacker_model="attacker",
        pages=[],
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode=test_mode,
        doc="goal",
        app_dir="/tmp/app",
        app_id="shared-mail",
        behavior_id="demo",
        variation_seed=0,
        task_seed=0,
        active_variant=active_variant,
        variant=active_variant,
    )


def test_study_creates_distinct_redteam_parent_dirs_per_agent(tmp_path: Path) -> None:
    benchmark = _DummyBenchmark(
        [
            _make_env_args(test_mode="benign", task_name="redteam.demo.benign"),
            _make_env_args(
                test_mode="adversarial",
                task_name="redteam.demo.adversarial_demo_v0",
                active_variant="adversarial_demo_v0",
            ),
        ]
    )
    study = Study(
        agent_args=[_DummyAgentArgs("agent-a"), _DummyAgentArgs("agent-b")],
        benchmark=benchmark,
        dir=tmp_path / "study",
    )

    study._prepare_redteam_parent_dirs(tmp_path / "study")

    parent_dirs_by_agent = {
        agent_name: {
            Path(exp_args.env_args.parent_exp_dir)
            for exp_args in study.exp_args_list
            if exp_args.agent_args.agent_name == agent_name
        }
        for agent_name in ("agent-a", "agent-b")
    }

    assert len(parent_dirs_by_agent["agent-a"]) == 1
    assert len(parent_dirs_by_agent["agent-b"]) == 1
    assert parent_dirs_by_agent["agent-a"] != parent_dirs_by_agent["agent-b"]
    assert next(iter(parent_dirs_by_agent["agent-a"])).as_posix().endswith(
        "redteam/demo/agent-a__a0/seed0_var0"
    )


def test_study_uses_unique_agent_run_ids_for_duplicate_agent_names(tmp_path: Path) -> None:
    benchmark = _DummyBenchmark(
        [
            _make_env_args(test_mode="benign", task_name="redteam.demo.benign"),
            _make_env_args(
                test_mode="adversarial",
                task_name="redteam.demo.adversarial_demo_v0",
                active_variant="adversarial_demo_v0",
            ),
        ]
    )
    study = Study(
        agent_args=[_DummyAgentArgs("agent"), _DummyAgentArgs("agent")],
        benchmark=benchmark,
        dir=tmp_path / "study",
    )

    study._prepare_redteam_parent_dirs(tmp_path / "study")

    parent_dirs = {
        Path(exp_args.env_args.parent_exp_dir)
        for exp_args in study.exp_args_list
        if exp_args.env_args.test_mode == "benign"
    }

    assert len(parent_dirs) == 2
    assert {path.parent.name for path in parent_dirs} == {"agent__a0", "agent__a1"}

    metadata_by_parent = {
        parent_dir.parent.name: (parent_dir / "metadata.json").read_text(encoding="utf-8")
        for parent_dir in parent_dirs
    }
    assert '"agent_name": "agent"' in metadata_by_parent["agent__a0"]
    assert '"agent_run_id": "agent__a0"' in metadata_by_parent["agent__a0"]
    assert '"agent_run_id": "agent__a1"' in metadata_by_parent["agent__a1"]


def test_study_agents_on_benchmark_does_not_share_env_args_between_agents(tmp_path: Path) -> None:
    benchmark = _DummyBenchmark(
        [
            _make_env_args(test_mode="benign", task_name="redteam.demo.benign"),
            _make_env_args(
                test_mode="adversarial",
                task_name="redteam.demo.adversarial_demo_v0",
                active_variant="adversarial_demo_v0",
            ),
        ]
    )
    study = Study(
        agent_args=[_DummyAgentArgs("agent-a"), _DummyAgentArgs("agent-b")],
        benchmark=benchmark,
        dir=tmp_path / "study",
    )

    benign_runs = [
        exp_args
        for exp_args in study.exp_args_list
        if exp_args.env_args.test_mode == "benign"
    ]
    assert len(benign_runs) == 2
    assert benign_runs[0].env_args is not benign_runs[1].env_args

    benign_runs[0].env_args.parent_exp_dir = "mutated"
    assert benign_runs[1].env_args.parent_exp_dir == ""


def test_study_log_attack_success_rate_uses_adversarial_runs_only(tmp_path: Path, monkeypatch) -> None:
    parent_dir = tmp_path / "study" / "redteam" / "demo" / "agent__a0" / "seed0_var0"
    parent_dir.mkdir(parents=True, exist_ok=True)
    study = Study(
        agent_args=[_DummyAgentArgs("agent")],
        benchmark=_DummyBenchmark([]),
        dir=tmp_path / "study",
    )
    study.exp_args_list = [
        SimpleNamespace(agent_args=SimpleNamespace(), env_args=SimpleNamespace()),
        SimpleNamespace(agent_args=SimpleNamespace(), env_args=SimpleNamespace()),
    ]

    exp_results = [
        SimpleNamespace(exp_dir=parent_dir / "benign"),
        SimpleNamespace(exp_dir=parent_dir / "adversarial_demo_v0"),
    ]
    exp_payloads = iter(
        [
            {
                "behavior_id": "demo",
                "behavior_name": "demo",
                "test_mode": "benign",
                "judge_result": {"overall_success": True, "overall_score": 1.0},
                "token_usage": {"input": 1, "output": 2},
                "attacker": {"token_usage": {"total": 0}},
                "exp_dir": str(parent_dir / "benign"),
                "parent_exp_dir": str(parent_dir),
            },
            {
                "behavior_id": "demo",
                "behavior_name": "demo",
                "test_mode": "adversarial",
                "judge_result": {"overall_success": False, "overall_score": 0.0},
                "token_usage": {"input": 3, "output": 4},
                "attacker": {"token_usage": {"total": 5}},
                "exp_dir": str(parent_dir / "adversarial_demo_v0"),
                "parent_exp_dir": str(parent_dir),
            },
        ]
    )

    monkeypatch.setattr(
        "agentlab.experiments.study.inspect_results.yield_all_exp_results",
        lambda *_args, **_kwargs: iter(exp_results),
    )
    monkeypatch.setattr(study, "_extract_experiment_data", lambda exp_result: next(exp_payloads))

    study._save_run_log(result_df=None, summary_df=None, start_time=0.0)

    study_log = json.loads((study.dir / "study_log.json").read_text(encoding="utf-8"))
    behavior_log = json.loads((parent_dir / "behavior_log.json").read_text(encoding="utf-8"))

    assert study_log["summary"]["attack_success_rate"] == 0.0
    assert behavior_log["summary"]["attack_success_rate"] == 0.0
    assert behavior_log["summary"]["benign_success_rate"] == 1.0
    assert behavior_log["summary"]["adversarial_success_rate"] == 0.0


def test_study_log_attack_success_rate_is_none_without_adversarial_runs(tmp_path: Path, monkeypatch) -> None:
    parent_dir = tmp_path / "study" / "redteam" / "demo" / "agent__a0" / "seed0_var0"
    parent_dir.mkdir(parents=True, exist_ok=True)
    study = Study(
        agent_args=[_DummyAgentArgs("agent")],
        benchmark=_DummyBenchmark([]),
        dir=tmp_path / "study",
    )
    study.exp_args_list = [SimpleNamespace(agent_args=SimpleNamespace(), env_args=SimpleNamespace())]

    exp_results = [SimpleNamespace(exp_dir=parent_dir / "benign")]
    exp_payloads = iter(
        [
            {
                "behavior_id": "demo",
                "behavior_name": "demo",
                "test_mode": "benign",
                "judge_result": {"overall_success": True, "overall_score": 1.0},
                "token_usage": {"input": 1, "output": 2},
                "attacker": {"token_usage": {"total": 0}},
                "exp_dir": str(parent_dir / "benign"),
                "parent_exp_dir": str(parent_dir),
            },
        ]
    )

    monkeypatch.setattr(
        "agentlab.experiments.study.inspect_results.yield_all_exp_results",
        lambda *_args, **_kwargs: iter(exp_results),
    )
    monkeypatch.setattr(study, "_extract_experiment_data", lambda exp_result: next(exp_payloads))

    study._save_run_log(result_df=None, summary_df=None, start_time=0.0)

    study_log = json.loads((study.dir / "study_log.json").read_text(encoding="utf-8"))
    behavior_log = json.loads((parent_dir / "behavior_log.json").read_text(encoding="utf-8"))

    assert study_log["summary"]["attack_success_rate"] is None
    assert behavior_log["summary"]["attack_success_rate"] is None
    assert behavior_log["summary"]["benign_success_rate"] == 1.0
    assert behavior_log["summary"]["adversarial_success_rate"] is None
