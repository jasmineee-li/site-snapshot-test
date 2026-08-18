"""Base class for eval awareness experiments."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Required, Self, TypedDict

from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import WebsiteExperimentResult, WebsiteSample

logger = logging.getLogger(__name__)


class ExperimentConfig(TypedDict, total=False):
    """One run's configuration, as `run.py` loads it from YAML or CLI flags.

    Every key the runner reads is named here, plus the per-experiment settings
    an experiment reads for itself in `from_config`. Naming them is what lets
    mypy check the runner's own accesses -- `config.get("max_per_side")` is an
    `int | None` against `run_pairs`, not an `Any` that fits anything.

    A TypedDict is a *static* description of a mapping. `yaml.safe_load`
    validates nothing, so the claims below hold for a config this tree
    constructs and are merely asserted for a config a person writes by hand.
    That is the split `require_str_list` exists for: a checked caller stays
    checked, and the one unchecked caller is validated where its value crosses
    into a constructor argument.
    """

    experiment: Required[str]
    name: str
    models: list[str]
    formats: list[str]
    manifest: str
    output_dir: str
    sources: list[str] | None
    website_types: list[str] | None
    ids: list[str] | None
    max_samples: int | None
    seed: int
    max_per_side: int | None
    cross_type: bool
    judges: list[str] | None


def require_str_list(value: object, key: str) -> list[str] | None:
    """Validate a config value declared as `list[str] | None`, or raise.

    Declared as taking `object` rather than the key's declared type on purpose:
    the check is against what a YAML file can actually deliver, not against
    what `ExperimentConfig` promises. Narrowing the parameter would make every
    branch below statically dead and buy nothing at runtime, which is the exact
    mistake this guards against.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise TypeError(f"config key {key!r} must be a list of strings, got {value!r}")
    return value


class BaseExperiment:
    """State and content loading shared by every eval awareness experiment.

    This class deliberately declares no way to *run* anything. Running is a
    capability, and an experiment picks exactly one of the two below:

    * `PerSampleExperiment` scores each sample on its own (`run_sample`),
      driven across samples x formats by the inherited `run`.
    * `PairwiseExperiment` scores samples against each other (`run_pairs`),
      which carries its own driver because a pair is not a sample.

    Splitting them is what lets `run.py` dispatch on the capability an
    experiment actually has rather than on the concrete class it happens to be.
    While both lived on one base class, the pairwise experiment carried a stub
    `run_sample` to satisfy the abstract contract and inherited a `run` that
    would open its results file in "w" mode and write zero rows — the same file
    `run_pairs` reads to resume. The dispatch's branch kept that path unreached,
    but it selected on `ComparativeExperiment` by name, so registering a second
    pairwise experiment under another key would have reached it. Selecting on
    the capability removes the stub and the inherited driver together, which
    makes the path unreachable rather than merely unvisited.

    This class is intentionally not an `ABC`: it now declares no abstract
    method, and an `ABC` without one does not block instantiation anyway, so
    the marker would assert an interface it no longer has. The two capability
    classes are the ABCs.
    """

    name: str = "base"

    def __init__(self, model: LLM, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, model: LLM, output_dir: Path, config: ExperimentConfig) -> Self:
        """Build this experiment from a run configuration.

        This is the single seam through which the CLI constructs a registered
        experiment, and the reason it exists is that the runner cannot type the
        alternative. The registry's declared value type is a capability class,
        whose `__init__` takes the model and the output directory and nothing
        else, so an experiment-specific keyword cannot be passed through it at
        all -- which is why the runner previously spread a `dict[str, Any]` and
        why mypy checked no part of that call, at any strictness tier.

        An experiment with settings of its own overrides this and reads them
        here, where `cls(...)` is an ordinary checked constructor call. The
        runner therefore never names an experiment to give it a setting, and
        adding a setting to one experiment does not touch `run.py`.

        The default reads nothing, which is correct for an experiment whose
        constructor takes only what every experiment takes.
        """
        return cls(model=model, output_dir=output_dir)

    def load_website_content(self, sample: WebsiteSample, format_type: str) -> str | None:
        """Load website content in the specified format.

        Args:
            sample: The website sample.
            format_type: One of "html", "axtree", "screenshot".

        Returns:
            Content string, or None if not available.
        """
        if format_type == "html" and sample.html_path:
            path = Path(sample.html_path)
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="replace")
                # Truncate very large HTML to avoid token limits
                if len(content) > 100_000:
                    content = content[:100_000] + "\n\n[... HTML truncated at 100k chars ...]"
                return content
        elif format_type == "axtree" and sample.axtree_path:
            path = Path(sample.axtree_path)
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")
        elif format_type == "screenshot" and sample.screenshot_path:
            path = Path(sample.screenshot_path)
            if path.exists():
                return str(path)  # Return path for image handling

        return None


class PerSampleExperiment(BaseExperiment, ABC):
    """An experiment that scores each sample on its own.

    Implement `run_sample`; `run` drives it across samples x formats.
    """

    @abstractmethod
    async def run_sample(
        self, sample: WebsiteSample, format_type: str
    ) -> list[WebsiteExperimentResult]:
        """Run experiment on a single sample.

        Args:
            sample: The website sample.
            format_type: One of "html", "axtree", "screenshot".

        Returns:
            List of experiment results.
        """
        pass

    async def run(
        self,
        samples: list[WebsiteSample],
        format_types: list[str],
    ) -> list[WebsiteExperimentResult]:
        """Run experiment on all samples and formats concurrently.

        Concurrency is bounded by the LLM client's semaphore.
        Saves results to JSONL when all tasks complete.
        """
        import asyncio

        output_file = self.output_dir / f"{self.name}_results.jsonl"

        async def _run_one(sample: WebsiteSample, fmt: str) -> list[WebsiteExperimentResult]:
            try:
                return await self.run_sample(sample, fmt)
            except Exception as e:
                logger.error(f"Error on {sample.id}/{fmt}: {e}")
                return [
                    WebsiteExperimentResult(
                        sample_id=sample.id,
                        experiment_type=self.name,
                        model=self.model.model,
                        format_used=fmt,
                        metadata={"error": str(e)},
                    )
                ]

        tasks = [_run_one(sample, fmt) for sample in samples for fmt in format_types]
        nested_results = await asyncio.gather(*tasks)
        all_results = [r for batch in nested_results for r in batch]

        # Save all results
        with open(output_file, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(
                    json.dumps(
                        {
                            "sample_id": r.sample_id,
                            "experiment_type": r.experiment_type,
                            "model": r.model,
                            "format_used": r.format_used,
                            "score": r.score,
                            "classification": r.classification,
                            "raw_response": r.raw_response,
                            "metadata": r.metadata,
                        }
                    )
                    + "\n"
                )

        print(f"  {self.name}: {len(all_results)} results saved to {output_file}")
        return all_results


class PairwiseExperiment(BaseExperiment, ABC):
    """An experiment that scores samples against each other rather than alone.

    Implement `run_pairs`. There is no shared driver: pairing, counterbalancing
    and resume are the pairwise experiment's own business, because the unit of
    work is a pair rather than a sample.
    """

    @abstractmethod
    async def run_pairs(
        self,
        samples: list[WebsiteSample],
        format_type: str,
        max_per_side: int | None = None,
        seed: int = 42,
        cross_type: bool = False,
    ) -> list[WebsiteExperimentResult]:
        """Run pairwise comparisons for one format.

        Args:
            samples: Candidate samples to pair.
            format_type: One of "html", "axtree", "screenshot".
            max_per_side: Cap on samples drawn per source, per bucket.
            seed: Random seed for sampling.
            cross_type: Pair across website types rather than within them.

        Returns:
            List of experiment results.
        """
