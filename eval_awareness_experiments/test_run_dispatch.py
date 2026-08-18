"""Invariants the experiment dispatch in `run.py` relies on.

`_run_model` builds an experiment from `EXPERIMENT_CLASSES` through
`from_config` and then selects a driver with
`isinstance(experiment, PairwiseExperiment)`. mypy checks that the registry's
declared value type admits both capabilities, that the `else` branch is
exhaustive, and -- since ADR 0007 -- that each `from_config` constructs its own
experiment with arguments the constructor accepts. Four properties it cannot
check are the ones a future change is most likely to break:

* a registered class declares exactly one capability, so the branch has a case
  for it;
* the pairwise experiment cannot reach the per-sample driver, which opens the
  results file `run_pairs` resumes from in "w" mode;
* a *second* pairwise experiment is routed by registering it, with no edit to
  the branch. This is the case the previous `isinstance(..., ComparativeExperiment)`
  selection got wrong, and it is what ADR 0006 records as the reason for the split.
* every registered experiment is reachable through the same three-argument
  seam, so the runner needs no per-experiment branch to build one -- and a
  config value that a type checker cannot vouch for is rejected where it is
  read rather than iterated as if it were well formed.

These assert on classes wherever possible; the routing and configuration tests
construct, because `isinstance` and `from_config` are what the dispatch
actually evaluates.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from eval_awareness_experiments.experiments.base import (
    BaseExperiment,
    ExperimentConfig,
    PairwiseExperiment,
    PerSampleExperiment,
)
from eval_awareness_experiments.experiments.comparative import ComparativeExperiment
from eval_awareness_experiments.experiments.trajectory_awareness import (
    DEFAULT_JUDGES,
    TrajectoryAwarenessExperiment,
)
from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.run import EXPERIMENT_CLASSES
from eval_awareness_experiments.types import WebsiteExperimentResult, WebsiteSample


def _stub_model() -> LLM:
    """A stand-in for `LLM` carrying only `.model`.

    Constructing a real `LLM` builds an `AsyncOpenAI` client and therefore needs
    an API key, which would make these unit tests environment-dependent. Nothing
    under test calls the model. The cast keeps the stub while telling the type
    checker what it stands for.
    """
    return cast(LLM, SimpleNamespace(model="stub/model"))


def test_every_registered_experiment_declares_exactly_one_run_capability():
    for name, cls in EXPERIMENT_CLASSES.items():
        capabilities = [
            base.__name__
            for base in (PerSampleExperiment, PairwiseExperiment)
            if issubclass(cls, base)
        ]
        assert len(capabilities) == 1, (
            f"{name} -> {cls.__name__} declares {capabilities or 'no'} run capabilities; "
            "the dispatch in run.py has a branch for exactly one"
        )


def test_pairwise_experiment_cannot_reach_the_per_sample_driver():
    # `run` writes `{name}_results.jsonl` in "w" mode, which is the same file
    # `run_pairs` reads to resume. Inheriting it is what made a zero-row
    # truncation reachable; the split removes the method rather than guarding it.
    assert not hasattr(ComparativeExperiment, "run")
    assert not hasattr(ComparativeExperiment, "run_sample")
    assert hasattr(ComparativeExperiment, "run_pairs")


def test_per_sample_experiments_carry_the_shared_driver():
    per_sample = [c for c in EXPERIMENT_CLASSES.values() if issubclass(c, PerSampleExperiment)]
    assert per_sample, "expected at least one per-sample experiment in the registry"
    for cls in per_sample:
        assert hasattr(cls, "run")
        assert hasattr(cls, "run_sample")
        assert not hasattr(cls, "run_pairs")


def test_a_second_pairwise_experiment_is_routed_by_capability(tmp_path: Path):
    """The case the previous by-class selection got wrong.

    A new pairwise experiment registered under its own key must reach the
    pairwise branch. Selecting on `ComparativeExperiment` would have sent it to
    the `else`, and so to the truncating driver.
    """

    class SecondPairwiseExperiment(PairwiseExperiment):
        name = "second_pairwise"

        async def run_pairs(
            self,
            samples: list[WebsiteSample],
            format_type: str,
            max_per_side: int | None = None,
            seed: int = 42,
            cross_type: bool = False,
        ) -> list[WebsiteExperimentResult]:
            return []

    registry: dict[str, type[PerSampleExperiment] | type[PairwiseExperiment]] = {
        **EXPERIMENT_CLASSES,
        "second_pairwise": SecondPairwiseExperiment,
    }
    experiment = registry["second_pairwise"](model=_stub_model(), output_dir=tmp_path)

    assert isinstance(experiment, PairwiseExperiment)
    # The selection this replaces would have missed it and fallen to the driver.
    assert not isinstance(experiment, ComparativeExperiment)
    assert not isinstance(experiment, PerSampleExperiment)


def test_every_registered_experiment_is_built_through_the_same_seam(tmp_path: Path):
    """The runner supplies three arguments and names no experiment.

    An experiment that took a fourth would have to be recognized by the runner
    to be constructed, which is the string comparison ADR 0007 removes.
    """
    for name, cls in EXPERIMENT_CLASSES.items():
        assert list(inspect.signature(cls.from_config).parameters) == [
            "model",
            "output_dir",
            "config",
        ], f"{name} -> {cls.__name__}.from_config takes arguments the runner does not supply"

        experiment = cls.from_config(
            model=_stub_model(),
            output_dir=tmp_path / name,
            config={"experiment": name},
        )
        assert isinstance(experiment, cls)


def test_trajectory_awareness_reads_its_own_judges_setting(tmp_path: Path):
    configured: ExperimentConfig = {
        "experiment": "trajectory_awareness",
        "judges": ["verbalized_awareness"],
    }
    experiment = TrajectoryAwarenessExperiment.from_config(
        model=_stub_model(), output_dir=tmp_path / "configured", config=configured
    )
    assert experiment.judge_names == ["verbalized_awareness"]

    # An absent key leaves the experiment's own default, which is the behavior
    # the runner's `and config.get("judges")` guard used to produce.
    default = TrajectoryAwarenessExperiment.from_config(
        model=_stub_model(),
        output_dir=tmp_path / "default",
        config={"experiment": "trajectory_awareness"},
    )
    assert default.judge_names == DEFAULT_JUDGES


def test_a_judges_value_that_is_not_a_list_of_strings_is_rejected(tmp_path: Path):
    """The failure this seam exists to convert into a clean one.

    `judge_names` is consumed by iterating it, so `judges: verbalized_awareness`
    -- a bare string -- is truthy, survives `judge_names or DEFAULT_JUDGES`, and
    is then iterated character by character. Measured on the code this replaces:
    20 judge lookups for a 20-character string, each failing, each swallowed by
    `run_sample`'s `except Exception` into an "error" result row, and the run
    completing with results written. No caller sees an exception.

    The cast is what `yaml.safe_load` does: it hands the runner a mapping that
    nothing has checked. Every other caller of this constructor passes
    `judge_names` as a keyword mypy checks, which is why the check lives here,
    at the one boundary a type checker cannot reach, rather than in `__init__`.
    """
    for judges in ("verbalized_awareness", ["verbalized_awareness", 3], {"a": 1}):
        config = cast(ExperimentConfig, {"experiment": "trajectory_awareness", "judges": judges})
        with pytest.raises(TypeError, match="'judges' must be a list of strings"):
            TrajectoryAwarenessExperiment.from_config(
                model=_stub_model(), output_dir=tmp_path / "rejected", config=config
            )


def test_capability_classes_do_not_satisfy_their_own_contract(tmp_path: Path):
    # Each declares an abstract method, so neither can stand in for a concrete
    # experiment. `BaseExperiment` is deliberately not an ABC -- it declares no
    # way to run -- so it is excluded here; ADR 0006 records why.
    for cls in (PerSampleExperiment, PairwiseExperiment):
        with pytest.raises(TypeError):
            cls(model=_stub_model(), output_dir=tmp_path)  # type: ignore[abstract]

    assert not issubclass(BaseExperiment, (PerSampleExperiment, PairwiseExperiment))
