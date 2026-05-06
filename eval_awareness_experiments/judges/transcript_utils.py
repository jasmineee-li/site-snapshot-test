"""Utilities for converting transcripts to various formats."""

from xml.sax.saxutils import escape

from eval_awareness_experiments.types import Message, Transcript


def _xml_escape(value: object) -> str:
    return escape(str(value), entities={'"': "&quot;"})


def _strip_seed_duplicates(transcript: Transcript) -> list[Message]:
    """Drop leading messages that duplicate Transcript.system_instructions /
    Transcript.instruction.

    `trajectory_to_transcript` copies `task_instruction` into both
    `Transcript.instruction` and `messages[0]`, so naively rendering both
    produces two identical adjacent user blocks — a model-distinguishable
    artifact ("real chatlogs don't repeat the user message verbatim").
    Dedupe by skipping at most one leading system + one leading user
    message that match the seed fields exactly.
    """
    msgs = list(transcript.messages)
    if (
        transcript.system_instructions
        and msgs
        and msgs[0].role == "system"
        and msgs[0].content == transcript.system_instructions
    ):
        msgs = msgs[1:]
    if (
        transcript.instruction
        and msgs
        and msgs[0].role == "user"
        and msgs[0].content == transcript.instruction
    ):
        msgs = msgs[1:]
    return msgs


def transcript_to_xml(transcript: Transcript) -> str:
    """Convert a transcript to XML format for prompts.

    Uses role-as-tagname layout (e.g. `<user>…</user>`). The seed system
    prompt and seed user message are emitted as `<system>` / `<user>` blocks
    rather than the older `<system_instructions>` / `<instruction>`
    wrappers, which leaked harness ground-truth (no production chatlog
    contains an `<instruction>` envelope, so a judge could classify a
    transcript as eval-derived purely from the wrapper rather than from
    the model's behavior).

    Leading `messages` that duplicate the seed fields are dropped so the
    same content isn't rendered twice in adjacent identical tags.
    """
    parts = []

    if transcript.system_instructions:
        parts.append(f"<system>\n{transcript.system_instructions}\n</system>")

    if transcript.instruction:
        parts.append(f"<user>\n{transcript.instruction}\n</user>")

    for msg in _strip_seed_duplicates(transcript):
        tag = "reasoning" if msg.is_hidden else msg.role

        label_attr = f' label="{msg.label}"' if msg.label else ""
        parts.append(f"<{tag}{label_attr}>\n{msg.content}\n</{tag}>")

    return "\n\n".join(parts)


def _format_needham_message(msg: Message) -> str:
    """Format a single Message into upstream-shaped Needham XML."""
    if msg.role == "tool" and msg.function is not None:
        result = (
            f'<message role="{_xml_escape(msg.role)}", '
            f'function="{_xml_escape(msg.function)}">\n{_xml_escape(msg.content)}'
        )
    else:
        result = f'<message role="{_xml_escape(msg.role)}">\n{_xml_escape(msg.content)}'

    if msg.role == "assistant" and msg.tool_calls:
        result += "\n<tool_calls>"
        for tool_call in msg.tool_calls:
            result += (
                f'<tool_call id="{_xml_escape(tool_call.id)}" '
                f'function="{_xml_escape(tool_call.function)}">'
            )
            for arg_name, arg_value in tool_call.arguments.items():
                result += (
                    f'\n<argument name="{_xml_escape(arg_name)}">'
                    f"{_xml_escape(arg_value)}</argument>"
                )
            result += "\n</tool_call>"
        result += "\n</tool_calls>"
    result += "\n</message>"
    return result


def transcript_to_needham_xml(transcript: Transcript) -> str:
    """Convert a transcript to Needham et al. XML format.

    Format (from jjpn97/eval_awareness, arXiv 2505.23836):

        <message role="system">
        ...
        </message>

        <message role="assistant">
        <reasoning>
        ...hidden chain-of-thought...
        </reasoning>
        ...visible content...
        </message>

        <message role="tool", function="click">
        ...tool result...
        </message>

    Extensions:
    - `system_instructions` and `instruction` fold into the first
      `<message role="system">` / `<message role="user">` messages.
    - Hidden/reasoning messages (is_hidden=True) are folded into assistant
      message text rather than serialized as a custom `<reasoning>` tag, so
      VEA and Transcript Purpose consume the same upstream-shaped XML.
    - Text and tool-call payloads are XML-escaped so transcript content cannot
      inject synthetic judge-visible tags.
    """
    parts: list[str] = []

    if transcript.system_instructions:
        parts.append(
            f'<message role="system">\n{_xml_escape(transcript.system_instructions)}\n</message>'
        )

    if transcript.instruction:
        parts.append(f'<message role="user">\n{_xml_escape(transcript.instruction)}\n</message>')

    pending_reasoning: list[str] = []
    for msg in _strip_seed_duplicates(transcript):
        if msg.is_hidden:
            pending_reasoning.append(msg.content)
            continue

        if msg.role == "assistant" and pending_reasoning:
            content = "\n\n".join([*pending_reasoning, msg.content])
            parts.append(
                _format_needham_message(
                    Message(
                        role="assistant",
                        content=content,
                        label=msg.label,
                        function=msg.function,
                        tool_calls=msg.tool_calls,
                    )
                )
            )
            pending_reasoning = []
        else:
            if pending_reasoning:
                parts.append(
                    _format_needham_message(
                        Message(
                            role="assistant",
                            content="\n\n".join(pending_reasoning),
                        )
                    )
                )
                pending_reasoning = []
            parts.append(_format_needham_message(msg))

    # Trailing hidden reasoning with no following visible assistant turn —
    # emit as a standalone assistant message so it isn't silently dropped.
    if pending_reasoning:
        parts.append(
            _format_needham_message(
                Message(
                    role="assistant",
                    content="\n\n".join(pending_reasoning),
                )
            )
        )

    return "\n".join(parts) + "\n\n"
