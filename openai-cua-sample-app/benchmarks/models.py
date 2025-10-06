"""Data models for benchmark definitions."""

from dataclasses import dataclass
from typing import List


@dataclass
class Page:
    """Definition of a page in a benchmark."""

    domain: str
    mode: str  # "synthetic" or "snapshot"

    def __post_init__(self):
        if self.mode not in ("synthetic", "snapshot"):
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'synthetic' or 'snapshot'"
            )


@dataclass
class BenchmarkCase:
    """A single test case within a benchmark."""

    behavior: str  # The task description
    pages: List[Page]
    tools: List[str]

    def __post_init__(self):
        if not self.behavior:
            raise ValueError("Behavior cannot be empty")
        if not self.pages:
            raise ValueError("Pages cannot be empty")
        if not self.tools:
            raise ValueError("Tools cannot be empty")


@dataclass
class Benchmark:
    """A collection of test cases."""

    name: str
    description: str
    cases: List[BenchmarkCase]

    def __post_init__(self):
        if not self.name:
            raise ValueError("Benchmark name cannot be empty")
        if not self.cases:
            raise ValueError("Benchmark must have at least one case")
