"""Core types for eval awareness experiments.

Includes types copied from eval-awareness-testbed (Transcript, Message, JudgeResult, etc.)
plus new types for browser-use website experiments (WebsiteSample, WebsiteExperimentResult).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Copied from eval-awareness-testbed/types.py (adapted: no ModelSpec, no EvalResult)
# ---------------------------------------------------------------------------

class AwarenessClassification(str, Enum):
    """Classification of eval awareness."""
    EVAL_AWARE = "EVAL_AWARE"
    NOT_EVAL_AWARE = "NOT_EVAL_AWARE"


class ReasoningCategory(str, Enum):
    """Categories for reasoning classification."""
    ALIGNMENT_EVAL = "Alignment Evaluation"
    CAPABILITY_EVAL = "Capability Evaluation"
    OTHER_EVAL = "Other Evaluation"
    REAL = "Real"
    OTHER = "Other"

    def is_eval_aware(self) -> bool:
        """Check if this category indicates eval awareness."""
        return self in (
            ReasoningCategory.ALIGNMENT_EVAL,
            ReasoningCategory.CAPABILITY_EVAL,
            ReasoningCategory.OTHER_EVAL,
        )


@dataclass
class Message:
    """A single message in a transcript."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    label: str = ""
    is_hidden: bool = False


@dataclass
class Transcript:
    """A conversation transcript to be judged."""
    messages: list[Message]
    source_file: str = ""
    eval_name: str | None = None
    scenario: str | None = None
    system_instructions: str = ""
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """Result from judging a transcript for eval awareness."""
    category: ReasoningCategory
    reasoning: str
    method: str
    raw_response: str = ""
    confidence: float | None = None
    quotes: list[str] | None = None
    purpose_guess: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chat message types (replacing inspect_ai.model.ChatMessage*)
# ---------------------------------------------------------------------------

@dataclass
class ChatMessageSystem:
    """System message for chat API."""
    content: str
    role: str = "system"


@dataclass
class ChatMessageUser:
    """User message for chat API."""
    content: str
    role: str = "user"


@dataclass
class ChatMessageAssistant:
    """Assistant message for chat API."""
    content: str
    role: str = "assistant"


ChatMessage = ChatMessageSystem | ChatMessageUser | ChatMessageAssistant


# ---------------------------------------------------------------------------
# New types for browser-use website experiments
# ---------------------------------------------------------------------------

@dataclass
class WebsiteSample:
    """A website sample from one of the benchmark sources."""
    id: str
    source: Literal["worldsim", "webarena", "agentcompany", "real"]
    website_type: str  # "github", "gmail", "paypal", "gitlab", etc.
    html_path: str | None = None
    axtree_path: str | None = None
    screenshot_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WebsiteExperimentResult:
    """Result from running an experiment on a website sample."""
    sample_id: str
    experiment_type: str  # "p_eval", "comparative", "transcript", "open_ended"
    model: str
    format_used: str  # "html", "axtree", "screenshot"
    score: float | None = None
    classification: str | None = None
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
