"""Port of the WebArena-Verified agent-response transform.

Source of truth: ``vendors/webarena-verified/examples/evaluation/extract_agent_response.py``.
The canonical benchmark CLI (``webarena-verified eval-tasks``) wires this
transform into its pipeline via ``--agent-response-transform``. Our pipeline
runs the evaluator directly, so we port the same 4-strategy extractor and call
it in-process before invoking ``AgentResponseEvaluator``. Keeping the strategies
byte-compatible with the vendor helper means a run that passes on the vendor
CLI also passes under our scoring.

The strategies (in order) are:

1. ``extract_plain_json`` -- parse the whole text as JSON.
2. ``extract_json_from_markdown_code_block`` -- unwrap ```json ... ``` fences.
3. ``extract_json_by_brace_matching`` -- scan for the first balanced
   ``{...}`` and JSON-parse it.
4. ``extract_json_line_by_line`` -- multi-line scan, collecting from the first
   ``{`` line to the first ``}`` line.

Public entrypoint: :func:`transform_agent_response`. Returns the extracted
dict, or ``None`` when every strategy fails. No network, no file IO, no
logging beyond a single debug line per attempt.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_plain_json(text: str) -> dict | None:
    """Try parsing ``text`` as plain JSON.

    Example input::

        {
          "task_type": "RETRIEVE",
          "status": "SUCCESS",
          "retrieved_data": ["value"]
        }

    Returns the parsed dict (if the top-level is an object) or ``None``.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_from_markdown_code_block(text: str) -> dict | None:
    """Extract JSON from ```json ... ``` or ``` ... ``` fences.

    The vendor pattern matches greedily per fence and returns the first valid
    JSON dict found. Invalid fences are skipped.
    """
    # Matches: ```json\n{...}\n``` or ```\n{...}\n```
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    code_blocks = re.findall(code_block_pattern, text)

    for block in code_blocks:
        try:
            parsed = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_json_by_brace_matching(text: str) -> dict | None:
    """Extract JSON object by finding the first balanced ``{...}`` substring.

    Walks the text tracking brace depth; when depth returns to zero, tries to
    JSON-parse the spanned substring. On failure, resets and keeps scanning.
    """
    brace_count = 0
    start_idx: int | None = None

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                candidate = text[start_idx : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start_idx = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start_idx = None

    return None


def extract_json_line_by_line(text: str) -> dict | None:
    """Extract JSON by scanning line-by-line for object start/end.

    Matches the vendor helper exactly: a line starting with ``{`` begins the
    window, and the first line ending with ``}`` (and not ``,}``) closes it.
    Invalid windows are discarded and scanning continues.
    """
    lines = text.split("\n")
    json_lines: list[str] = []
    in_json = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("{"):
            in_json = True
            json_lines = [stripped]
        elif in_json:
            json_lines.append(stripped)
            if stripped.endswith("}") and not stripped.endswith(",}"):
                candidate = "\n".join(json_lines)
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    in_json = False
                    json_lines = []
                    continue
                if isinstance(parsed, dict):
                    return parsed
                in_json = False
                json_lines = []

    return None


_STRATEGIES: tuple[tuple[str, Any], ...] = (
    ("plain_json", extract_plain_json),
    ("markdown_code_block", extract_json_from_markdown_code_block),
    ("brace_matching", extract_json_by_brace_matching),
    ("line_by_line", extract_json_line_by_line),
)


def transform_agent_response(raw_text: str) -> dict | None:
    """Extract a structured dict from raw agent text via 4 strategies.

    This mirrors ``extract_json_from_text`` in
    ``vendors/webarena-verified/examples/evaluation/extract_agent_response.py``
    but returns ``None`` on total failure instead of raising, so callers can
    fall back to the raw text (matching the CLI's own preference order: see
    ``vendors/.../__main__.py`` ``_transform_agent_response`` which returns
    ``None`` rather than failing the task when the transform cannot produce a
    JSON object).

    Args:
        raw_text: The agent's final_result text.

    Returns:
        The extracted dict if any strategy succeeds, else ``None``.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    for name, strategy in _STRATEGIES:
        result = strategy(raw_text)
        if result is not None:
            logger.debug("agent_response_transform: strategy %s succeeded", name)
            return result
        logger.debug("agent_response_transform: strategy %s did not match", name)
    return None


def detect_strategy(raw_text: str) -> str | None:
    """Return the first strategy name that matches ``raw_text``, or ``None``.

    Useful for telemetry and the rescore CLI. Does not short-circuit on the
    same result as :func:`transform_agent_response`; it is just a thin
    introspection helper over the same ordered strategy list.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    for name, strategy in _STRATEGIES:
        if strategy(raw_text) is not None:
            return name
    return None
