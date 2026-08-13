"""Byte-equivalence tests for :mod:`warp_taskgen.phase_4.needham_xml`.

The serializer is a verbatim port of Needham et al. 2025's
``pipeline/apply_pq.py::_format_xml`` (commit
``1543ffe644d3f8f3a43b5dfd7814dbe9b12f890d`` at
``github.com/jjpn97/eval_awareness``). These tests guard against both
accidental drift from upstream and "helpful" whitespace/grammar fixes.

The ``test_matches_upstream_*`` cases import upstream code directly when
``inspect_ai`` is installed (the ``vendors/eval_awareness/`` clone is
the source). When it isn't, those cases skip; the golden-string cases
still run and cover the full branch matrix.

To regenerate the upstream comparison locally::

    uv pip install inspect_ai
    uv run pytest tests/test_phase_4_needham_xml.py -q
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from warp_taskgen.phase_4.needham_chat_types import ChatMessage, ToolCall
from warp_taskgen.phase_4.needham_xml import _format_xml, format_xml


def test_system_message_uses_role_attribute_no_separate_tag() -> None:
    msg = ChatMessage(role="system", text="You are a browser agent.")
    assert _format_xml(msg) == ('<message role="system">\nYou are a browser agent.\n</message>')


def test_user_message_shape() -> None:
    msg = ChatMessage(role="user", text="Find the cheapest flight.")
    assert _format_xml(msg) == ('<message role="user">\nFind the cheapest flight.\n</message>')


def test_assistant_message_without_tool_calls_omits_tool_calls_block() -> None:
    msg = ChatMessage(role="assistant", text="I will open the form.")
    out = _format_xml(msg)
    assert "<tool_calls>" not in out
    assert out == '<message role="assistant">\nI will open the form.\n</message>'


def test_tool_role_opening_tag_has_comma_separator() -> None:
    """Upstream quirk: `<message role="tool", function="click">`.

    The comma between attributes is ungrammatical XML but Needham's code
    emits it verbatim; preserving it is the point of byte-equivalence.
    """
    msg = ChatMessage(role="tool", text="click ok.", function="click")
    out = _format_xml(msg)
    assert '<message role="tool", function="click">\n' in out


def test_tool_role_without_function_falls_back_to_role_only() -> None:
    msg = ChatMessage(role="tool", text="raw tool output", function=None)
    assert _format_xml(msg).startswith('<message role="tool">\n')


def test_assistant_tool_calls_render_with_tool_call_and_argument_tags() -> None:
    msg = ChatMessage(
        role="assistant",
        text="I will click the button.",
        tool_calls=(
            ToolCall(
                id="0",
                function="click",
                arguments={"selector": "#ok", "timeout": 500},
            ),
        ),
    )
    out = _format_xml(msg)
    assert (
        '<tool_calls><tool_call id="0" function="click">\n'
        '<argument name="selector">#ok</argument>\n'
        '<argument name="timeout">500</argument>\n'
        "</tool_call>\n"
        "</tool_calls>" in out
    )
    assert "<parameter" not in out


def test_multiple_tool_calls_concatenate_without_internal_newline() -> None:
    """Upstream quirk: the second `<tool_call` has no leading newline."""
    msg = ChatMessage(
        role="assistant",
        text="act.",
        tool_calls=(
            ToolCall(id="0", function="click", arguments={"a": 1}),
            ToolCall(id="1", function="fill", arguments={"b": 2}),
        ),
    )
    out = _format_xml(msg)
    # Close of first tool_call is followed directly by the second's opening
    # with no intervening newline — upstream accumulates via string concat.
    assert (
        "</tool_call>"
        '<tool_call id="1" function="fill">\n'
        '<argument name="b">2</argument>\n'
        "</tool_call>" in out
    )


def test_argument_values_stringified_via_fstring_python_defaults() -> None:
    """Int/float/bool argument values stringify per Python's f-string defaults."""
    msg = ChatMessage(
        role="assistant",
        text="x",
        tool_calls=(
            ToolCall(
                id="0",
                function="f",
                arguments={"i": 1, "f": 1.5, "b": True, "n": None},
            ),
        ),
    )
    out = _format_xml(msg)
    assert '<argument name="i">1</argument>' in out
    assert '<argument name="f">1.5</argument>' in out
    assert '<argument name="b">True</argument>' in out
    assert '<argument name="n">None</argument>' in out


def test_no_reasoning_tag_emitted_ever() -> None:
    """Thinking tokens are folded into assistant text upstream; no <reasoning>."""
    msg = ChatMessage(role="assistant", text="I'm thinking about the form.")
    assert "<reasoning>" not in _format_xml(msg)


def test_format_xml_joins_with_single_newline_and_trails_double() -> None:
    messages = [
        ChatMessage(role="system", text="sys."),
        ChatMessage(role="user", text="u."),
    ]
    out = format_xml(messages)
    assert out.endswith("\n\n")
    # Between-message separator is exactly one newline (after a closing
    # </message>, i.e. `</message>\n<message ...`).
    assert "</message>\n<message " in out


def test_format_xml_deterministic_on_repeat() -> None:
    msg = ChatMessage(role="user", text="stable")
    a = format_xml([msg])
    b = format_xml([msg])
    assert a == b


def test_xml_special_chars_in_content_are_escaped() -> None:
    msg = ChatMessage(role="user", text="<b>hi</b> & go")
    out = _format_xml(msg)
    assert "&lt;b&gt;hi&lt;/b&gt; &amp; go" in out
    assert "<b>hi</b>" not in out


def test_xml_special_chars_in_tool_calls_are_escaped() -> None:
    msg = ChatMessage(
        role="assistant",
        text='Use </message> and "quotes"',
        tool_calls=(
            ToolCall(
                id="0",
                function='fill"value',
                arguments={"selector": "</tool_call>", "value": "<unsafe>&"},
            ),
        ),
    )
    out = _format_xml(msg)
    assert "&lt;/message&gt;" in out
    assert 'function="fill&quot;value"' in out
    assert "&lt;/tool_call&gt;" in out
    assert "&lt;unsafe&gt;&amp;" in out


# ---------- Upstream comparison (skipped when inspect_ai isn't installed) ----

_VENDOR_PATH = Path(__file__).resolve().parents[1] / "vendors" / "eval_awareness"


def _load_upstream_format_xml():
    """Import ``pipeline.apply_pq._format_xml`` from the vendor clone.

    Skips when ``inspect_ai`` is unavailable (its import is load-bearing
    at the top of upstream's module, even though ``_format_xml`` itself
    only reads attributes).
    """
    if not (_VENDOR_PATH / "pipeline" / "apply_pq.py").exists():
        pytest.skip("vendor clone missing: vendors/eval_awareness/")
    try:
        importlib.import_module("inspect_ai")
    except ImportError:
        pytest.skip("inspect_ai not installed; upstream comparison skipped")
    sys.path.insert(0, str(_VENDOR_PATH))
    try:
        mod = importlib.import_module("pipeline.apply_pq")
    finally:
        if sys.path and sys.path[0] == str(_VENDOR_PATH):
            sys.path.pop(0)
    return mod._format_xml, mod.format_xml


def test_matches_upstream_format_xml_across_all_message_shapes() -> None:
    _upstream_format_xml_msg, upstream_format_xml = _load_upstream_format_xml()
    from inspect_ai.model import (  # type: ignore[import-not-found]
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )
    from inspect_ai.tool import ToolCall as UpstreamToolCall  # type: ignore[import-not-found]

    our_messages = [
        ChatMessage(role="system", text="sys body"),
        ChatMessage(role="user", text="user body"),
        ChatMessage(
            role="assistant",
            text="I will click.",
            tool_calls=(
                ToolCall(
                    id="0",
                    function="click",
                    arguments={"selector": "#ok", "n": 2},
                ),
            ),
        ),
        ChatMessage(role="tool", text="clicked.", function="click"),
    ]
    upstream_messages = [
        ChatMessageSystem(content="sys body"),
        ChatMessageUser(content="user body"),
        ChatMessageAssistant(
            content="I will click.",
            tool_calls=[
                UpstreamToolCall(
                    id="0",
                    function="click",
                    arguments={"selector": "#ok", "n": 2},
                )
            ],
        ),
        ChatMessageTool(content="clicked.", function="click"),
    ]
    ours = format_xml(our_messages)
    theirs, _ = upstream_format_xml(upstream_messages, [])
    assert ours == theirs
