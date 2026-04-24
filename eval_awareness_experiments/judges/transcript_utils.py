"""Utilities for converting transcripts to various formats."""

from eval_awareness_experiments.types import Message, Transcript


def transcript_to_xml(transcript: Transcript) -> str:
    """Convert a transcript to XML format for prompts.

    Uses role-as-tagname layout (e.g. `<user>…</user>`) with top-level
    `<system_instructions>` / `<instruction>` wrappers. Retained for the
    existing `binary_third_person`, `probability_third_person`, and
    `purpose_xml` judges that were calibrated against this format.
    """
    parts = []

    if transcript.system_instructions:
        parts.append(f"<system_instructions>\n{transcript.system_instructions}\n</system_instructions>")

    if transcript.instruction:
        parts.append(f"<instruction>\n{transcript.instruction}\n</instruction>")

    for msg in transcript.messages:
        tag = "reasoning" if msg.is_hidden else msg.role

        label_attr = f' label="{msg.label}"' if msg.label else ""
        parts.append(f"<{tag}{label_attr}>\n{msg.content}\n</{tag}>")

    return "\n\n".join(parts)


def _format_needham_message(msg: Message) -> str:
    """Format a single Message into Needham-style `<message role="...">`.

    Needham's reference format carries a `function="..."` attribute on tool
    messages. Our `trajectory_converter` does not thread actual tool function
    names onto tool Messages — `Message.label` holds step-tracking IDs like
    "step_3_observation". Rather than emit misleading attribute values, we
    omit the `function=` attribute entirely until real function names are
    threaded through. Every benchmark's converter can add this later.

    Hidden messages are rendered as `<reasoning>…</reasoning>` nested inside
    a synthetic assistant `<message>` wrapper at the caller level — this
    function only handles a single non-hidden message.
    """
    open_tag = f'<message role="{msg.role}">'
    return f"{open_tag}\n{msg.content}\n</message>"


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
    - Hidden/reasoning messages (is_hidden=True) are collected and emitted
      as `<reasoning>…</reasoning>` nested inside the next visible
      `<message role="assistant">` block. This preserves separable CoT
      produced by reasoning models (GLM-5, Kimi, Claude extended thinking)
      — a signal Needham's inspect_ai-native implementation did not carry.
    """
    parts: list[str] = []

    if transcript.system_instructions:
        parts.append(f'<message role="system">\n{transcript.system_instructions}\n</message>')

    if transcript.instruction:
        parts.append(f'<message role="user">\n{transcript.instruction}\n</message>')

    pending_reasoning: list[str] = []
    for msg in transcript.messages:
        if msg.is_hidden:
            pending_reasoning.append(msg.content)
            continue

        if msg.role == "assistant" and pending_reasoning:
            reasoning_block = "\n".join(
                f"<reasoning>\n{r}\n</reasoning>" for r in pending_reasoning
            )
            parts.append(
                f'<message role="assistant">\n{reasoning_block}\n{msg.content}\n</message>'
            )
            pending_reasoning = []
        else:
            parts.append(_format_needham_message(msg))

    # Trailing hidden reasoning with no following visible assistant turn —
    # emit as a standalone assistant message so it isn't silently dropped.
    if pending_reasoning:
        reasoning_block = "\n".join(
            f"<reasoning>\n{r}\n</reasoning>" for r in pending_reasoning
        )
        parts.append(f'<message role="assistant">\n{reasoning_block}\n</message>')

    return "\n\n".join(parts)
