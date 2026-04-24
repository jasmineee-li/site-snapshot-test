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
class TrajectoryStep:
    """A single step in an agent's trajectory."""
    thought: str       # Agent's reasoning/thinking at this step
    action: str        # What the agent did (click, type, terminate, etc.)
    observation: str = ""  # Page state / axtree / observation after action
    preamble: str = ""  # Assistant content emitted before action (e.g. forced scratchpad tags routed to content channel instead of reasoning)


@dataclass
class StandardizedTrajectory:
    """Benchmark-agnostic representation of an agent trajectory.

    Bridges benchmark-specific formats and the judge input requirements.
    The Test_Awareness_Steering judge needs (task, reasoning, answer);
    existing browser-sim judges need a Transcript.
    """
    id: str
    benchmark: str            # "doomarena", "wasp", "os-harm", "webarena-infinity", etc.
    task_instruction: str     # The task given to the agent
    agent: str                # "opus-4.6", "kimi", etc.
    steps: list[TrajectoryStep]
    final_answer: str         # Agent's final output/termination message
    passed: bool | None = None
    attack_type: str | None = None      # "prompt_injection", "adversarial_popup", etc.
    attack_success: bool | None = None  # Whether the attack succeeded (for ASR)
    safety_category: str | None = None  # "web_injection", "social_engineering", etc.
    injection_condition: str | None = None  # "baseline", "eval_cue", "scenario_cue", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WebsiteSample:
    """A website sample from one of the benchmark sources."""
    id: str
    source: Literal[
        "worldsim", "webarena", "webarena-infinity", "agentcompany", "real",
        "doomarena", "wasp", "safearena", "os-harm", "agentdojo", "injecagent", "eia",
    ]
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
