"""Tests for ``warp_taskgen.agent_response_transform``.

Covers the four extraction strategies and the negative cases so the
evaluator-parity transform stays byte-compatible with the vendor helper at
``vendors/webarena-verified/examples/evaluation/extract_agent_response.py``.
"""

from __future__ import annotations

import textwrap

from warp_taskgen.agent_response_transform import (
    detect_strategy,
    extract_json_by_brace_matching,
    extract_json_from_markdown_code_block,
    extract_json_line_by_line,
    extract_plain_json,
    transform_agent_response,
)


def test_plain_json_returns_same_dict() -> None:
    raw = '{"task_type": "retrieve", "status": "SUCCESS", "retrieved_data": [6]}'
    out = transform_agent_response(raw)
    assert out == {"task_type": "retrieve", "status": "SUCCESS", "retrieved_data": [6]}
    assert detect_strategy(raw) == "plain_json"


def test_plain_json_strategy_only_accepts_dicts() -> None:
    # Top-level arrays are not accepted because the evaluator expects an object.
    assert extract_plain_json("[1, 2, 3]") is None


def test_markdown_fenced_json_is_extracted() -> None:
    raw = textwrap.dedent(
        """\
        Here's the answer:

        ```json
        {
          "task_type": "retrieve",
          "status": "SUCCESS",
          "retrieved_data": ["42"]
        }
        ```
        """
    )
    out = transform_agent_response(raw)
    assert out == {
        "task_type": "retrieve",
        "status": "SUCCESS",
        "retrieved_data": ["42"],
    }
    assert detect_strategy(raw) == "markdown_code_block"


def test_markdown_fenced_without_lang_tag() -> None:
    raw = 'Plain fence:\n\n```\n{"a": 1, "b": 2}\n```\n'
    out = transform_agent_response(raw)
    assert out == {"a": 1, "b": 2}


def test_brace_matched_json_inside_prose() -> None:
    raw = 'The answer is {"task_type":"retrieve","status":"SUCCESS","retrieved_data":[6]}. Done.'
    out = transform_agent_response(raw)
    assert out == {
        "task_type": "retrieve",
        "status": "SUCCESS",
        "retrieved_data": [6],
    }
    # The plain-JSON strategy fails on this prose because the whole string
    # isn't JSON, so the transform falls through to brace matching.
    assert extract_plain_json(raw) is None
    assert extract_json_by_brace_matching(raw) == {
        "task_type": "retrieve",
        "status": "SUCCESS",
        "retrieved_data": [6],
    }
    assert detect_strategy(raw) == "brace_matching"


def test_bare_final_line_json_after_prose_via_line_scan() -> None:
    raw = textwrap.dedent(
        """\
        Reasoning about the task.
        More context here.
        {
          "task_type": "retrieve",
          "status": "SUCCESS"
        }
        Footer text.
        """
    )
    out = transform_agent_response(raw)
    assert out == {"task_type": "retrieve", "status": "SUCCESS"}
    # Brace matching also succeeds here because the object is well-formed in
    # place. Line-scan exists for cases where the brace walker gets confused
    # by stray braces earlier in the text.
    assert extract_json_line_by_line(raw) == {
        "task_type": "retrieve",
        "status": "SUCCESS",
    }


def test_pure_prose_returns_none() -> None:
    assert transform_agent_response("The answer is 6") is None
    assert detect_strategy("The answer is 6") is None


def test_malformed_json_inside_fences_returns_none() -> None:
    raw = "```json\n{task_type: retrieve, missing_quotes: true}\n```"
    assert extract_json_from_markdown_code_block(raw) is None
    assert transform_agent_response(raw) is None


def test_empty_input_returns_none() -> None:
    assert transform_agent_response("") is None
    assert transform_agent_response("   \n  ") is None
    assert transform_agent_response(None) is None  # type: ignore[arg-type]


def test_brace_walker_skips_non_json_braces_then_finds_real_object() -> None:
    # An agent writing prose that includes {stylistic braces} followed by a
    # real JSON object. The walker sees the stylistic brace is non-JSON and
    # should continue until it finds a parseable candidate.
    raw = (
        "Some prose with {stylistic} text and then the real answer "
        '{"task_type": "retrieve", "status": "SUCCESS", "retrieved_data": [1]}'
    )
    out = transform_agent_response(raw)
    assert out == {
        "task_type": "retrieve",
        "status": "SUCCESS",
        "retrieved_data": [1],
    }


def test_markdown_multiple_fences_returns_first_valid() -> None:
    raw = textwrap.dedent(
        """\
        ```
        not json at all
        ```

        ```json
        {"a": 1}
        ```
        """
    )
    # First fence fails to parse; second fence parses cleanly.
    assert transform_agent_response(raw) == {"a": 1}
