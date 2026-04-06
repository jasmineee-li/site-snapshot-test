"""Utilities for converting transcripts to various formats."""

from eval_awareness_experiments.types import Transcript


def transcript_to_xml(transcript: Transcript) -> str:
    """Convert a transcript to XML format for prompts."""
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
