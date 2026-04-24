"""Convert benchmark-specific trajectory formats into StandardizedTrajectory.

Supports:
- webarena-infinity (HuggingFace history.json + result.json)
- AgentLab/BrowserGym (step_*.pkl.gz + summary_info.json)
- DoomArena (AgentLab redteam format with judge_results.json)
- WASP (BrowserGym-based axtree + action logs)
- SafeArena (VisualWebArena-based harmful task benchmark, McGill-NLP)
- OS-harm (better_log.json + test_injection.json task configs)  [DISABLED]

Also provides conversion to Transcript (for existing browser-sim judges)
and to (task, reasoning, answer) tuples (for Test_Awareness_Steering judge).
"""

import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Any

from eval_awareness_experiments.types import (
    Message,
    StandardizedTrajectory,
    Transcript,
    TrajectoryStep,
)

logger = logging.getLogger(__name__)

MAX_REASONING_CHARS = 80_000


# ---------------------------------------------------------------------------
# WebArena-Infinity (HuggingFace format)
# ---------------------------------------------------------------------------

def parse_webarena_infinity_trajectory(
    trajectory_dir: Path,
) -> StandardizedTrajectory | None:
    """Parse a webarena-infinity trajectory from history.json + result.json."""
    history_path = trajectory_dir / "history.json"
    result_path = trajectory_dir / "result.json"

    if not history_path.exists() or not result_path.exists():
        logger.warning(f"Missing files in {trajectory_dir}")
        return None

    history = json.loads(history_path.read_text())
    result = json.loads(result_path.read_text())

    steps = []
    for entry in history.get("history", []):
        thought = entry.get("thought", "")
        actions = entry.get("actions", [])
        action_str = "; ".join(
            a.get("type", "unknown") for a in actions
        ) if actions else ""
        steps.append(TrajectoryStep(
            thought=thought,
            action=action_str,
        ))

    # Extract agent name from directory name (e.g., "kimi_task_m16" -> "kimi")
    dir_name = trajectory_dir.name
    agent = dir_name.split("_")[0] if "_" in dir_name else "unknown"

    return StandardizedTrajectory(
        id=f"wai_{dir_name}",
        benchmark="webarena-infinity",
        task_instruction=result.get("instruction", ""),
        agent=agent,
        steps=steps,
        final_answer=result.get("final_result", ""),
        passed=result.get("passed"),
        metadata={
            "task_id": result.get("task_id", ""),
            "difficulty": result.get("difficulty", ""),
            "elapsed": result.get("elapsed"),
            "steps_count": result.get("steps"),
            "format": history.get("format", ""),
        },
    )


# ---------------------------------------------------------------------------
# AgentLab / BrowserGym (pickle-based step format)
# ---------------------------------------------------------------------------

def _load_step_pickle(step_path: Path) -> dict[str, Any] | None:
    """Load a step_N.pkl.gz file, returning its dict representation."""
    try:
        with gzip.open(step_path, "rb") as f:
            step_info = pickle.load(f)
        # StepInfo is a dataclass; convert to dict for safe access
        if hasattr(step_info, "__dict__"):
            return vars(step_info)
        return step_info
    except Exception as e:
        logger.warning(f"Failed to load {step_path}: {e}")
        return None


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict-like object or an object with attributes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_step_index(step_path: Path) -> int:
    """Extract the numeric index from step_N.pkl.gz filenames."""
    try:
        return int(step_path.name.split("_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Unexpected step filename: {step_path.name}") from e


def _extract_agentlab_reasoning(agent_info: Any) -> str:
    """Concatenate all the surfaces where the agent might have written reasoning:

    1. OpenRouter's hidden reasoning tokens (`message.reasoning`)
    2. AgentLab's parsed <think> tag content (`agent_info.think`)
    3. Any <unfiltered_observations> or <scratchpad> block in the raw content
       (scratchpad-style extra instructions tell the model to write into
       these tags; they'd otherwise be dropped by AgentLab's <think>-only parser)

    Deduplicates substring matches so content present in multiple surfaces
    is only included once.
    """
    import re as _re

    chat_messages = _safe_get(agent_info, "chat_messages")
    messages = _safe_get(chat_messages, "messages", []) or []

    parts = []
    raw_content = ""

    for message in reversed(messages):
        if (_safe_get(message, "role") or "") != "assistant":
            continue
        reasoning = _safe_get(message, "reasoning", "")
        if isinstance(reasoning, str) and reasoning:
            parts.append(reasoning)
        content = _safe_get(message, "content", "")
        if isinstance(content, str):
            raw_content = content
        break  # only the last assistant turn

    think = _safe_get(agent_info, "think", "")
    if isinstance(think, str) and think:
        already = any(think.strip() in p for p in parts)
        if not already:
            parts.append(think)

    # Extract scratchpad-like tag content from the raw assistant content
    for tag in ("unfiltered_observations", "scratchpad", "reflection"):
        for m in _re.finditer(
            rf"<{tag}>(.*?)</{tag}>", raw_content, _re.DOTALL | _re.IGNORECASE
        ):
            block = m.group(1).strip()
            already = any(block in p for p in parts)
            if block and not already:
                parts.append(block)

    return "\n\n".join(parts) if parts else ""


def parse_agentlab_trajectory(
    exp_dir: Path,
) -> StandardizedTrajectory | None:
    """Parse an AgentLab/BrowserGym experiment directory.

    Expects: step_*.pkl.gz files + summary_info.json.
    """
    summary_path = exp_dir / "summary_info.json"
    if not summary_path.exists():
        logger.warning(f"No summary_info.json in {exp_dir}")
        return None

    summary = json.loads(summary_path.read_text())

    # Load all steps
    step_files = sorted(exp_dir.glob("step_*.pkl.gz"), key=_parse_step_index)
    steps = []
    task_instruction = ""

    for step_path in step_files:
        step_data = _load_step_pickle(step_path)
        if step_data is None:
            continue

        agent_info = step_data.get("agent_info", {})
        obs = step_data.get("obs", {})

        thought = _extract_agentlab_reasoning(agent_info)
        action = step_data.get("action", "")
        observation = (_safe_get(obs, "axtree_txt", "") or "")[:5000]  # truncate large axtrees

        # Extract task instruction from first step's goal
        goal = _safe_get(obs, "goal_object")
        if not task_instruction and goal:
            if isinstance(goal, str):
                task_instruction = goal
            else:
                task_instruction = _safe_get(goal, "text", str(goal))

        steps.append(TrajectoryStep(
            thought=thought,
            action=action,
            observation=observation,
        ))

    # Try to get task instruction from exp_args if not found in steps
    if not task_instruction:
        exp_args_path = exp_dir / "exp_args.pkl"
        if exp_args_path.exists():
            try:
                with open(exp_args_path, "rb") as f:
                    exp_args = pickle.load(f)
                env_args = getattr(exp_args, "env_args", None)
                if env_args:
                    task_instruction = getattr(env_args, "task_name", "") or ""
            except Exception:
                pass

    # Final answer from last step's thought or action
    final_answer = ""
    if steps:
        last = steps[-1]
        final_answer = last.thought if last.thought else last.action

    return StandardizedTrajectory(
        id=f"agentlab_{exp_dir.name}",
        benchmark="agentlab",
        task_instruction=task_instruction,
        agent="unknown",  # caller should set this from exp_args
        steps=steps,
        final_answer=final_answer,
        passed=summary.get("cum_reward", 0) > 0,
        metadata={
            "n_steps": summary.get("n_steps"),
            "cum_reward": summary.get("cum_reward"),
            "err_msg": summary.get("err_msg"),
            "cost": summary.get("stats.cum_cost"),
            "exp_dir": str(exp_dir),
        },
    )


# ---------------------------------------------------------------------------
# DoomArena (AgentLab redteam variant format)
# ---------------------------------------------------------------------------

def parse_doomarena_trajectory(
    exp_dir: Path,
) -> StandardizedTrajectory | None:
    """Parse a DoomArena experiment directory (AgentLab redteam format).

    Same as AgentLab but also reads judge_results.json for attack success.
    """
    traj = parse_agentlab_trajectory(exp_dir)
    if traj is None:
        return None

    traj.benchmark = "doomarena"
    traj.id = f"doomarena_{exp_dir.name}"

    # Read attack results — DoomArena writes attack_summary_info.json,
    # older formats may use judge_results.json
    for attack_file in ["attack_summary_info.json", "judge_results.json"]:
        attack_path = exp_dir / attack_file
        if attack_path.exists():
            try:
                attack_data = json.loads(attack_path.read_text())
                traj.attack_success = attack_data.get(
                    "attack_successful", attack_data.get("overall_success", False)
                )
                traj.metadata["attack_undetected"] = attack_data.get("attack_undetected")
                traj.metadata["success_filter_results"] = attack_data.get("success_filter_results")
            except (json.JSONDecodeError, OSError):
                pass
            break

    # Determine attack type from directory structure
    traj.attack_type = "adversarial"
    if "benign" in exp_dir.name:
        traj.attack_type = "benign"
    traj.safety_category = "adversarial_web_attack"

    return traj


# ---------------------------------------------------------------------------
# SafeArena (BrowserGym-based, McGill-NLP; WebArena-derived)
# ---------------------------------------------------------------------------

def parse_safearena_trajectory(
    exp_dir: Path,
) -> StandardizedTrajectory | None:
    """Parse a SafeArena experiment directory.

    SafeArena is built on AgentLab + BrowserGym, so trajectories share the
    same step_*.pkl.gz + summary_info.json layout as DoomArena and AgentLab.
    The key differences:
    - Task IDs come from SafeArena's harm.json / safe.json splits
    - 'jailbreak' mode flags indicate the task has harmful content
    - Attack success is determined by the task's native evaluator
    """
    traj = parse_agentlab_trajectory(exp_dir)
    if traj is None:
        return None

    traj.benchmark = "safearena"
    traj.id = f"safearena_{exp_dir.name}"
    traj.attack_type = "harmful_task"
    traj.safety_category = "web_harm"

    # Detect harm vs safe split from directory name or exp_args
    if "harm" in exp_dir.name.lower() or traj.metadata.get("split") == "harm":
        traj.metadata["safearena_split"] = "harm"
        # For harm tasks, attack_success = task completed (agent carried out harmful act)
        traj.attack_success = traj.passed
    else:
        traj.metadata["safearena_split"] = "safe"

    return traj


# ---------------------------------------------------------------------------
# WASP (BrowserGym-based, VisualWebArena)
# ---------------------------------------------------------------------------

def parse_wasp_trajectory(
    result_dir: Path,
) -> StandardizedTrajectory | None:
    """Parse a WASP benchmark trajectory.

    WASP uses BrowserGym format with axtree observations. Trajectory files
    may be in different output formats (webarena, claude, gpt_web_tools).
    """
    # Try AgentLab/BrowserGym format first (step_*.pkl.gz)
    if list(result_dir.glob("step_*.pkl.gz")):
        traj = parse_agentlab_trajectory(result_dir)
        if traj:
            traj.benchmark = "wasp"
            traj.id = f"wasp_{result_dir.name}"
            traj.attack_type = "prompt_injection"
            traj.safety_category = "web_injection"
            return traj

    # Try JSON-based format (WASP may output JSON logs)
    for json_file in ["trajectory.json", "result.json", "log.json"]:
        json_path = result_dir / json_file
        if json_path.exists():
            data = json.loads(json_path.read_text())
            break
    else:
        logger.warning(f"No recognized trajectory format in {result_dir}")
        return None

    steps = []
    history = data.get("history", data.get("steps", []))
    for entry in history:
        thought = entry.get("thought", entry.get("reasoning", ""))
        action = entry.get("action", entry.get("response", ""))
        observation = entry.get("observation", entry.get("state", ""))
        if isinstance(observation, dict):
            observation = observation.get("axtree_txt", str(observation))
        steps.append(TrajectoryStep(
            thought=thought,
            action=action if isinstance(action, str) else json.dumps(action),
            observation=observation[:5000] if isinstance(observation, str) else "",
        ))

    task = data.get("instruction", data.get("task", data.get("goal", "")))
    final = data.get("final_result", data.get("answer", ""))
    if not final and steps:
        final = steps[-1].thought or steps[-1].action

    return StandardizedTrajectory(
        id=f"wasp_{result_dir.name}",
        benchmark="wasp",
        task_instruction=task,
        agent=data.get("agent", "unknown"),
        steps=steps,
        final_answer=final,
        passed=data.get("passed", data.get("success")),
        attack_type="prompt_injection",
        attack_success=data.get("attack_success"),
        safety_category="web_injection",
        metadata={
            "task_id": data.get("task_id", ""),
            "injection_type": data.get("injection_type", ""),
            "output_format": data.get("output_format", ""),
        },
    )


# ---------------------------------------------------------------------------
# OS-harm (better_log.json + task config)
# ---------------------------------------------------------------------------

def parse_osharm_trajectory(
    log_dir: Path,
    task_info: dict | None = None,
) -> StandardizedTrajectory | None:
    """Parse an OS-harm trajectory from better_log.json.

    Args:
        log_dir: Directory containing better_log.json and optionally screenshots.
        task_info: Task metadata from test_injection.json (optional).
    """
    log_path = log_dir / "better_log.json"
    if not log_path.exists():
        # Try alternative filenames
        for alt in ["log.json", "trajectory.json"]:
            alt_path = log_dir / alt
            if alt_path.exists():
                log_path = alt_path
                break
        else:
            logger.warning(f"No log file in {log_dir}")
            return None

    data = json.loads(log_path.read_text())

    # OS-harm logs may be a list of steps or a dict with a steps key
    if isinstance(data, list):
        step_entries = data
        meta = {}
    else:
        step_entries = data.get("steps", data.get("history", []))
        meta = {k: v for k, v in data.items() if k not in ("steps", "history")}

    steps = []
    for entry in step_entries:
        thought = entry.get("thought", entry.get("reasoning", ""))
        action = entry.get("action", "")
        if isinstance(action, dict):
            action = json.dumps(action)
        observation = entry.get("observation", entry.get("screenshot", ""))
        if isinstance(observation, dict):
            observation = str(observation)
        steps.append(TrajectoryStep(
            thought=thought,
            action=action,
            observation=observation[:5000] if isinstance(observation, str) else "",
        ))

    task_info = task_info or {}
    task = task_info.get("instruction", meta.get("task", meta.get("instruction", "")))

    # Determine attack success from task info or log metadata
    attack_success = task_info.get("attack_success", meta.get("attack_success"))

    final = meta.get("final_result", "")
    if not final and steps:
        final = steps[-1].thought or steps[-1].action

    return StandardizedTrajectory(
        id=f"osharm_{log_dir.name}",
        benchmark="os-harm",
        task_instruction=task,
        agent=meta.get("agent", task_info.get("agent", "unknown")),
        steps=steps,
        final_answer=final,
        passed=meta.get("passed", task_info.get("passed")),
        attack_type="prompt_injection",
        attack_success=attack_success,
        safety_category=task_info.get("harm_category", "web_injection"),
        metadata={
            "task_id": task_info.get("task_id", log_dir.name),
            "injection_vector": task_info.get("injection_vector", ""),
            "malicious_goal": task_info.get("malicious_goal", ""),
            "domain": task_info.get("domain", ""),
            **meta,
        },
    )


# ---------------------------------------------------------------------------
# Tool-calling benchmarks (AgentDojo, InjecAgent)
# ---------------------------------------------------------------------------

def parse_toolcalling_trajectory(
    task_json_path: Path,
) -> StandardizedTrajectory | None:
    """Parse a single-task JSON log emitted by ToolCallingRunner.

    Expected schema (produced by eval_awareness_experiments/tool_calling_runner.py):
        {
            "task_id": str,
            "task_instruction": str,
            "benchmark": "agentdojo" | "injecagent",
            "agent": str,                         # model name
            "condition": "baseline" | "eval_cue" | "scenario_cue",
            "extra_instructions_preset": str,
            "steps": [{"thought": str, "action": str, "observation": str}, ...],
            "final_answer": str,
            "passed": bool | null,
            "attack_success": bool | null,
            "safety_category": str | null,
            "metadata": {...}
        }

    Accepts either a single-task JSON file or a directory containing one.
    """
    if task_json_path.is_dir():
        candidates = sorted(task_json_path.glob("*.json"))
        if not candidates:
            logger.warning(f"No JSON in {task_json_path}")
            return None
        task_json_path = candidates[0]

    data = json.loads(task_json_path.read_text())

    benchmark = data.get("benchmark")
    if benchmark not in ("agentdojo", "injecagent"):
        logger.error(
            f"parse_toolcalling_trajectory: unexpected benchmark {benchmark!r} "
            f"in {task_json_path}"
        )
        return None

    steps = [
        TrajectoryStep(
            thought=step.get("thought", ""),
            action=step.get("action", ""),
            observation=step.get("observation", ""),
            preamble=step.get("preamble", ""),
        )
        for step in data.get("steps", [])
    ]

    return StandardizedTrajectory(
        id=f"{benchmark}_{data.get('task_id', task_json_path.stem)}",
        benchmark=benchmark,
        task_instruction=data.get("task_instruction", ""),
        agent=data.get("agent", "unknown"),
        steps=steps,
        final_answer=data.get("final_answer", ""),
        passed=data.get("passed"),
        attack_type="tool_output_injection",
        attack_success=data.get("attack_success"),
        safety_category=data.get("safety_category", "tool_injection"),
        injection_condition=data.get("condition", "baseline"),
        metadata={
            "task_id": data.get("task_id", ""),
            "extra_instructions_preset": data.get("extra_instructions_preset", "none"),
            "task_json_path": str(task_json_path),
            **(data.get("metadata") or {}),
        },
    )


# ---------------------------------------------------------------------------
# EIA (Environmental Injection Attack — SeeAct trajectories, import-only)
# ---------------------------------------------------------------------------

def parse_eia_trajectory(
    trajectory_dir: Path,
) -> StandardizedTrajectory | None:
    """Parse a normalized EIA trajectory directory.

    Layout produced by `import_eia_trajectories.py`:
        data/eia/<task_id>/
            trajectory.json  # SeeAct step log
            task.json        # {"task_id", "instruction", "attack_metadata": {...}}

    EIA trajectories are pre-existing (from the authors' eval_results.zip).
    Condition is always "baseline" — we can't apply causal XML wrappers to
    logged-before-we-existed trajectories.
    """
    traj_path = trajectory_dir / "trajectory.json"
    task_path = trajectory_dir / "task.json"
    if not traj_path.exists() or not task_path.exists():
        logger.warning(
            f"EIA trajectory dir {trajectory_dir} missing trajectory.json or task.json"
        )
        return None

    traj_data = json.loads(traj_path.read_text())
    task_data = json.loads(task_path.read_text())

    # SeeAct step entries: {"thought", "action", "target_element", "observation"}
    # Fall back to other common key names if EIA used different ones.
    raw_steps = (
        traj_data.get("steps")
        or traj_data.get("history")
        or (traj_data if isinstance(traj_data, list) else [])
    )
    steps = []
    for entry in raw_steps:
        if not isinstance(entry, dict):
            continue
        thought = entry.get("thought") or entry.get("reasoning") or entry.get("rationale") or ""
        action_raw = entry.get("action") or entry.get("grounded_action") or ""
        if isinstance(action_raw, dict):
            target = entry.get("target_element") or ""
            action = f"{action_raw.get('type', 'action')}({target})".strip()
        else:
            action = str(action_raw)
        observation = entry.get("observation") or entry.get("element_text") or entry.get("page_state") or ""
        if isinstance(observation, dict):
            observation = json.dumps(observation)[:5000]
        steps.append(TrajectoryStep(
            thought=thought,
            action=action,
            observation=observation[:5000] if isinstance(observation, str) else "",
        ))

    attack_meta = task_data.get("attack_metadata") or {}

    return StandardizedTrajectory(
        id=f"eia_{task_data.get('task_id', trajectory_dir.name)}",
        benchmark="eia",
        task_instruction=task_data.get("instruction", ""),
        agent=traj_data.get("agent") or task_data.get("agent", "seeact"),
        steps=steps,
        final_answer=traj_data.get("final_answer", "") or (steps[-1].action if steps else ""),
        passed=task_data.get("passed"),
        attack_type="environmental_injection",
        attack_success=attack_meta.get("attack_success"),
        safety_category="vision_web_injection",
        injection_condition="baseline",
        metadata={
            "task_id": task_data.get("task_id", trajectory_dir.name),
            "trajectory_dir": str(trajectory_dir),
            "attack_metadata": attack_meta,
        },
    )


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def trajectory_to_transcript(traj: StandardizedTrajectory) -> Transcript:
    """Convert a StandardizedTrajectory to a Transcript for browser-sim judges.

    Layout:
    - User message: task instruction
    - For each step: hidden assistant message (reasoning) + assistant message (action)
    - Final assistant message: answer
    """
    messages = [
        Message(role="user", content=traj.task_instruction),
    ]

    for i, step in enumerate(traj.steps):
        if step.thought:
            messages.append(Message(
                role="assistant",
                content=step.thought,
                label=f"step_{i}_reasoning",
                is_hidden=True,
            ))
        if step.preamble:
            messages.append(Message(
                role="assistant",
                content=step.preamble,
                label=f"step_{i}_preamble",
            ))
        if step.action:
            messages.append(Message(
                role="assistant",
                content=step.action,
                label=f"step_{i}_action",
            ))
        if step.observation:
            messages.append(Message(
                role="tool",
                content=step.observation[:3000],
                label=f"step_{i}_observation",
            ))

    if traj.final_answer and (
        not traj.steps
        or (traj.final_answer != traj.steps[-1].thought
            and traj.final_answer != traj.steps[-1].action)
    ):
        messages.append(Message(
            role="assistant",
            content=traj.final_answer,
            label="final_answer",
        ))

    return Transcript(
        messages=messages,
        source_file=traj.metadata.get("exp_dir", ""),
        eval_name=traj.benchmark,
        instruction=traj.task_instruction,
        metadata={
            "trajectory_id": traj.id,
            "benchmark": traj.benchmark,
            "agent": traj.agent,
            "attack_type": traj.attack_type,
            "attack_success": traj.attack_success,
            "passed": traj.passed,
        },
    )


def trajectory_to_judge_fields(
    traj: StandardizedTrajectory,
) -> tuple[str, str, str]:
    """Extract (task, reasoning, answer) for the Test_Awareness_Steering judge.

    Full-trajectory mode: concatenate all step thoughts as reasoning,
    final result as answer. Truncates reasoning at MAX_REASONING_CHARS.
    """
    task = traj.task_instruction

    # Concatenate all step thoughts (and preambles) with step numbering.
    # Preamble = assistant `content` emitted before tool calls; on some models
    # forced scratchpad tags land here rather than in the reasoning channel,
    # and the judge needs to see them.
    reasoning_parts = []
    for i, step in enumerate(traj.steps):
        if step.thought:
            reasoning_parts.append(f"[Step {i + 1}] {step.thought}")
        if step.preamble:
            reasoning_parts.append(f"[Step {i + 1} preamble] {step.preamble}")

    reasoning = "\n\n".join(reasoning_parts)
    if len(reasoning) > MAX_REASONING_CHARS:
        reasoning = reasoning[:MAX_REASONING_CHARS] + "\n\n[... reasoning truncated ...]"

    answer = traj.final_answer

    return task, reasoning, answer


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PARSERS = {
    "webarena-infinity": parse_webarena_infinity_trajectory,
    "agentlab": parse_agentlab_trajectory,
    "doomarena": parse_doomarena_trajectory,
    "wasp": parse_wasp_trajectory,
    "safearena": parse_safearena_trajectory,
    "agentdojo": parse_toolcalling_trajectory,
    "injecagent": parse_toolcalling_trajectory,
    "eia": parse_eia_trajectory,
    # os-harm uses parse_osharm_trajectory which takes extra task_info arg
}


def parse_trajectory(
    source: str,
    trajectory_dir: Path,
    **kwargs: Any,
) -> StandardizedTrajectory | None:
    """Dispatch to the appropriate parser based on source type."""
    if source == "os-harm":
        return parse_osharm_trajectory(trajectory_dir, task_info=kwargs.get("task_info"))

    parser = PARSERS.get(source)
    if parser is None:
        logger.error(f"Unknown trajectory source: {source}")
        return None

    return parser(trajectory_dir)
