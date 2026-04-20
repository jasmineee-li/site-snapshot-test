"""Byte-equivalent port of Needham et al. 2025's XML transcript serializer.

Upstream: ``github.com/jjpn97/eval_awareness`` ``pipeline/apply_pq.py``
at commit ``1543ffe644d3f8f3a43b5dfd7814dbe9b12f890d``.

The functions here reproduce ``_format_xml`` and ``format_xml`` character
for character, quirks included. Notable quirks preserved verbatim:

- The tool-role opening tag emits a literal comma between attributes:
  ``<message role="tool", function="click">``. That trailing comma is
  upstream; we reproduce it so the bytes match.
- Messages are joined with a single ``\\n``; the overall output ends in
  ``\\n\\n`` (a trailing blank line).
- Tool calls live inside the assistant message as
  ``<tool_calls><tool_call id="..." function="..."><argument name="..">
  ...</argument></tool_call></tool_calls>``. No ``<parameter>`` alias.
- Thinking tokens are not a separate tag. Upstream expects them folded
  into the assistant message ``text``; the adapter in
  :mod:`worldsim.phase_4.needham_adapter` handles that projection.

Do not "clean up" the output. The byte-equivalence test
(``tests/test_phase_4_needham_xml.py``) compares against a frozen golden
captured from upstream code; grammar fixes break that test on purpose.
"""

from __future__ import annotations

from worldsim.phase_4.needham_chat_types import ChatMessage


def _format_xml(msg: ChatMessage) -> str:
    result = (
        f'<message role="{msg.role}", function="{msg.function}">\n{msg.text}'
        if msg.role == "tool" and msg.function is not None
        else f'<message role="{msg.role}">\n{msg.text}'
    )
    if msg.role == "assistant" and msg.tool_calls is not None:
        result += "\n<tool_calls>"
        for tool_call in msg.tool_calls:
            result += f'<tool_call id="{tool_call.id}" function="{tool_call.function}">'
            for arg_name, arg_value in tool_call.arguments.items():
                result += f'\n<argument name="{arg_name}">{arg_value}</argument>'
            result += "\n</tool_call>"
        result += "\n</tool_calls>"
    result += "\n</message>"
    return result


def format_xml(messages: list[ChatMessage]) -> str:
    return "\n".join(_format_xml(m) for m in messages) + "\n\n"
