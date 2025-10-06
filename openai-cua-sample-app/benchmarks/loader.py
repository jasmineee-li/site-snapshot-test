"""Load and validate benchmark JSON files."""

import json
from pathlib import Path
from typing import Dict, Any

from .models import Benchmark, BenchmarkCase, Page


class BenchmarkLoader:
    """Loads benchmark definitions from JSON files."""

    def load(self, benchmark_path: str) -> Benchmark:
        """
        Load a benchmark from a JSON file.

        Args:
            benchmark_path: Path to the benchmark JSON file

        Returns:
            Validated Benchmark object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the benchmark is invalid
        """
        path = Path(benchmark_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._validate_and_parse(data)

    def _validate_and_parse(self, data: Dict[str, Any]) -> Benchmark:
        """Validate and parse benchmark data."""
        # Validate top-level structure
        if not isinstance(data, dict):
            raise ValueError("Benchmark must be a JSON object")

        name = data.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("Benchmark must have a 'name' field (string)")

        description = data.get("description", "")
        if not isinstance(description, str):
            raise ValueError("Benchmark 'description' must be a string")

        data_list = data.get("data")
        if not data_list or not isinstance(data_list, list):
            raise ValueError("Benchmark must have a 'data' field (array)")

        # Parse each case
        cases = []
        for i, case_data in enumerate(data_list):
            try:
                case = self._parse_case(case_data)
                cases.append(case)
            except Exception as e:
                raise ValueError(f"Error in benchmark case {i}: {e}") from e

        return Benchmark(name=name, description=description, cases=cases)

    def _parse_case(self, case_data: Dict[str, Any]) -> BenchmarkCase:
        """Parse a single benchmark case."""
        if not isinstance(case_data, dict):
            raise ValueError("Case must be a JSON object")

        behavior = case_data.get("behavior")
        if not behavior or not isinstance(behavior, str):
            raise ValueError("Case must have a 'behavior' field (string)")

        pages_data = case_data.get("pages")
        if not pages_data or not isinstance(pages_data, list):
            raise ValueError("Case must have a 'pages' field (array)")

        tools = case_data.get("tools")
        if not tools or not isinstance(tools, list):
            raise ValueError("Case must have a 'tools' field (array)")

        # Validate tools are strings
        if not all(isinstance(t, str) for t in tools):
            raise ValueError("All tools must be strings")

        # Parse pages
        pages = []
        for page_data in pages_data:
            page = self._parse_page(page_data)
            pages.append(page)

        return BenchmarkCase(behavior=behavior, pages=pages, tools=tools)

    def _parse_page(self, page_data: Dict[str, Any]) -> Page:
        """Parse a single page definition."""
        if not isinstance(page_data, dict):
            raise ValueError("Page must be a JSON object")

        domain = page_data.get("domain")
        if not domain or not isinstance(domain, str):
            raise ValueError("Page must have a 'domain' field (string)")

        mode = page_data.get("mode", "synthetic")
        if not isinstance(mode, str):
            raise ValueError("Page 'mode' must be a string")

        return Page(domain=domain, mode=mode)
