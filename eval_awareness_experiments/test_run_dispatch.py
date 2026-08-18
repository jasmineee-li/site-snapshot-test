"""Invariants the experiment dispatch in `run.py` relies on.

`_run_model` builds an experiment from `EXPERIMENT_CLASSES` and then selects a
driver with `isinstance(experiment, PairwiseExperiment)`. mypy checks that the
registry's declared value type admits both capabilities and that the `else`
branch is exhaustive, but three properties it cannot check are the ones a future
change is most likely to break:

* a registered class declares exactly one capability, so the branch has a case
  for it;
* the pairwise experiment cannot reach the per-sample driver, which opens the
  results file `run_pairs` resumes from in "w" mode;
* a *second* pairwise experiment is routed by registering it, with no edit to
  the branch. This is the case the previous `isinstance(..., ComparativeExperiment)`
  selection got wrong, and it is what ADR 0006 records as the reason for the split.

These assert on classes wherever possible; the routing test constructs, because
`isinstance` is what the dispatch actually evaluates.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from eval_awareness_experiments.experiments.base import (
    BaseExperiment,
    PairwiseExperiment,
    PerSampleExperiment,
)
from eval_awareness_experiments.experiments.comparative import ComparativeExperiment
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


def test_capability_classes_do_not_satisfy_their_own_contract(tmp_path: Path):
    # Each declares an abstract method, so neither can stand in for a concrete
    # experiment. `BaseExperiment` is deliberately not an ABC -- it declares no
    # way to run -- so it is excluded here; ADR 0006 records why.
    for cls in (PerSampleExperiment, PairwiseExperiment):
        with pytest.raises(TypeError):
            cls(model=_stub_model(), output_dir=tmp_path)  # type: ignore[abstract]

    assert not issubclass(BaseExperiment, (PerSampleExperiment, PairwiseExperiment))
