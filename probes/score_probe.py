"""Score transcripts (or arbitrary text) with a saved DoM probe.

Used for two things:
  1. Diagnostic — measure whether OpenCUA / GUI-Owl are naturally
     eval-aware by scoring their baseline benchmark trajectories.
  2. Steering-strength tuning — score text under different `--alpha`
     values to find the alpha that brings mean eval-awareness score
     close to baseline-deploy.

Trajectory inputs are flat JSONL (one transcript per line) or the
ToolCallingRunner JSON-per-task format under
`results/toolcalling/{benchmark}/.../<task>.json`. For the latter we
join `steps[*].thought / preamble / action / observation` into a
single text blob per task, then score it.

Usage:
    # Score a single string:
    python -m probes.score_probe \\
        --model xlangai/OpenCUA-32B \\
        --probe-dir probes/trained/opencua-32b/ \\
        --text "I think this might be an evaluation."

    # Score a directory of ToolCallingRunner JSON outputs:
    python -m probes.score_probe \\
        --model xlangai/OpenCUA-32B \\
        --probe-dir probes/trained/opencua-32b/ \\
        --transcripts-dir results/toolcalling/agentdojo/baseline_none_xlangai_OpenCUA-32B/ \\
        --output scored_transcripts.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
import re
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model
from probes.scoring import classify_score


def _safe_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _flatten_toolcalling_task(task_json: dict) -> str:
    """Join ToolCallingRunner steps into a single transcript blob."""
    parts: list[str] = []
    if task_json.get("system_prompt"):
        parts.append(f"[system]\n{task_json['system_prompt']}")
    parts.append(f"[user]\n{task_json.get('task_instruction', '')}")
    for i, step in enumerate(task_json.get("steps", [])):
        if step.get("thought"):
            parts.append(f"[assistant.reasoning {i}]\n{step['thought']}")
        if step.get("preamble"):
            parts.append(f"[assistant.preamble {i}]\n{step['preamble']}")
        if step.get("action"):
            parts.append(f"[assistant.action {i}]\n{step['action']}")
        if step.get("observation"):
            parts.append(f"[tool.observation {i}]\n{step['observation']}")
    if task_json.get("final_answer"):
        parts.append(f"[assistant.final]\n{task_json['final_answer']}")
    return "\n\n".join(parts)


def _flatten_message_content(content) -> str:
    """Normalize AgentLab/OpenAI-style multimodal content to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _normalize_chat_messages(messages) -> list[dict[str, str]]:
    """Convert AgentLab message objects/dicts into HF chat-template messages."""
    out: list[dict[str, str]] = []
    for msg in messages or []:
        role = _safe_get(msg, "role")
        if not role:
            cls_name = type(msg).__name__.replace("Message", "").lower()
            role = "user" if cls_name == "human" else cls_name
        if role == "human":
            role = "user"
        if role not in {"system", "user", "assistant"}:
            role = "assistant" if role == "base" else str(role)

        content = _flatten_message_content(_safe_get(msg, "content", ""))
        reasoning = _safe_get(msg, "reasoning")
        if role == "assistant" and isinstance(reasoning, str) and reasoning.strip():
            content = f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n\n{content}".strip()

        if content:
            out.append({"role": role, "content": content})
    return out


def _chat_template_tokens(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    span: str = "all_messages_mean",
) -> tuple[torch.Tensor, list[int]]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if span == "all_messages_mean":
        positions = _filter_content_positions(tokenizer, ids, range(len(ids)))
    else:
        positions = _chat_message_content_positions(tokenizer, text, messages, span)
    if not positions:
        positions = list(range(len(ids)))
    return torch.tensor([ids], device=model.device), positions


def _filter_content_positions(tokenizer, ids: list[int], positions) -> list[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    out: list[int] = []
    for idx in positions:
        if idx < 0 or idx >= len(ids):
            continue
        token_id = ids[idx]
        token_text = tokenizer.decode([token_id])
        if token_id in special_ids:
            continue
        if not token_text.strip():
            continue
        if token_text.startswith("<|") or token_text in {"<s>", "</s>"}:
            continue
        out.append(idx)
    return out


def _content_spans_in_rendered(
    rendered: str,
    messages: list[dict[str, str]],
) -> list[dict[str, int | str]]:
    spans: list[dict[str, int | str]] = []
    search_from = 0
    for idx, message in enumerate(messages):
        content = _flatten_message_content(message.get("content", ""))
        if not content:
            continue
        start = rendered.find(content, search_from)
        if start == -1:
            start = rendered.find(content)
        if start == -1:
            continue
        end = start + len(content)
        spans.append({
            "index": idx,
            "role": str(message.get("role", "")),
            "start": start,
            "end": end,
        })
        search_from = end
    return spans


def _chat_message_content_positions(
    tokenizer,
    rendered: str,
    messages: list[dict[str, str]],
    span: str,
) -> list[int]:
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    spans = _content_spans_in_rendered(rendered, messages)

    if span in {"user_message_mean", "last_user_token"}:
        candidates = [item for item in spans if item["role"] == "user"]
        target = candidates[-1] if candidates else None
    elif span == "first_assistant_message_mean":
        candidates = [item for item in spans if item["role"] == "assistant"]
        target = candidates[0] if candidates else None
    else:
        raise ValueError(f"Unknown chat message span: {span}")

    if target is None:
        return []

    start = int(target["start"])
    end = int(target["end"])
    prefix_ids = tokenizer.encode(rendered[:start], add_special_tokens=False)
    through_ids = tokenizer.encode(rendered[:end], add_special_tokens=False)
    positions = _filter_content_positions(
        tokenizer,
        ids,
        range(len(prefix_ids), len(through_ids)),
    )
    if span == "last_user_token" and positions:
        return [positions[-1]]
    return positions


def _chat_user_content_tokens(model, tokenizer, text: str) -> tuple[torch.Tensor, list[int]]:
    """Tokenize text as a single user message and select the inserted content span.

    This matches `probes.eval_sad_stages_oversight.tokenize_chat_content`,
    which is how the SAD oversight layer-selection run scored prompts.
    """
    full_messages = [{"role": "user", "content": text}]
    empty_messages = [{"role": "user", "content": ""}]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    empty_text = tokenizer.apply_chat_template(
        empty_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    empty_ids = tokenizer.encode(empty_text, add_special_tokens=False)

    start = 0
    max_prefix = min(len(full_ids), len(empty_ids))
    while start < max_prefix and full_ids[start] == empty_ids[start]:
        start += 1

    end_full = len(full_ids)
    end_empty = len(empty_ids)
    while (
        end_full > start
        and end_empty > start
        and full_ids[end_full - 1] == empty_ids[end_empty - 1]
    ):
        end_full -= 1
        end_empty -= 1

    if end_full <= start:
        positions = list(range(len(full_ids)))
    else:
        positions = list(range(start, end_full))
    return torch.tensor([full_ids], device=model.device), positions


def _raw_tokens(model, tokenizer, text: str) -> tuple[torch.Tensor, list[int]]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    positions = [
        idx
        for idx, token_id in enumerate(ids)
        if token_id not in special_ids and tokenizer.decode([token_id]).strip()
    ]
    if not positions:
        positions = list(range(len(ids)))
    return torch.tensor([ids], device=model.device), positions


def _score_tokens(
    probe: ContrastiveProbe,
    model,
    tokens: torch.Tensor,
    positions: list[int],
    layer: int,
    threshold: float,
) -> dict:
    vector = probe.normalized_vectors[layer]
    vec = vector[0] if vector.dim() > 1 else vector
    score_holder: dict[str, float] = {}

    def hook_fn(activation, hook):
        valid_positions = [pos for pos in positions if pos < activation.shape[1]]
        if not valid_positions:
            valid_positions = list(range(activation.shape[1]))
        act = activation[0, valid_positions, :].mean(dim=0)
        v = vec.to(device=act.device, dtype=act.dtype).flatten()
        score_holder["mean_score"] = torch.dot(act, v).item()
        return activation

    try:
        model.reset_hooks()
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]):
            with torch.no_grad():
                model(tokens)
    finally:
        del tokens

    mean_score = float(score_holder.get("mean_score", 0.0))
    classification, confidence = classify_score(mean_score, threshold)
    return {
        "mean_score": mean_score,
        "classification": classification,
        "confidence": confidence,
        "n_positions": len(positions),
    }


def _score_text(
    probe: ContrastiveProbe,
    model,
    tokenizer,
    text: str,
    layer: int,
    threshold: float,
    tokenization: str,
) -> dict:
    if tokenization == "raw":
        tokens, positions = _raw_tokens(model, tokenizer, text)
    elif tokenization == "chat_user_content":
        tokens, positions = _chat_user_content_tokens(model, tokenizer, text)
    else:
        tokens, positions = _chat_template_tokens(
            model,
            tokenizer,
            [{"role": "user", "content": text}],
        )
    return _score_tokens(probe, model, tokens, positions, layer, threshold)


def _score_messages(
    probe: ContrastiveProbe,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    layer: int,
    threshold: float,
    span: str = "all_messages_mean",
) -> dict:
    tokens, positions = _chat_template_tokens(model, tokenizer, messages, span=span)
    return _score_tokens(probe, model, tokens, positions, layer, threshold)


def _load_step_pickle(step_path: Path):
    try:
        with gzip.open(step_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", step_path, e)
        return None


def _parse_step_index(step_path: Path) -> int:
    try:
        return int(step_path.name.split("_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return 0


def _browser_task_dirs(root: Path, max_items: int | None = None) -> list[Path]:
    dirs = []
    for summary in sorted(root.rglob("summary_info.json")):
        task_dir = summary.parent
        try:
            rel_parts = task_dir.relative_to(root).parts
        except ValueError:
            rel_parts = ()
        if any(part.startswith(("_", ".")) for part in rel_parts):
            continue
        if list(task_dir.glob("step_*.pkl.gz")):
            dirs.append(task_dir)
    return dirs[:max_items] if max_items else dirs


def _task_id_from_dir(task_dir: Path) -> str:
    match = re.search(r"webarena\.(\d+)", task_dir.name)
    return f"webarena.{match.group(1)}" if match else task_dir.name


def _load_asr_map(path: str | None) -> dict[str, bool]:
    if not path:
        return {}
    out: dict[str, bool] = {}
    fp = Path(path)
    if not fp.exists():
        logger.warning("ASR file not found: %s", fp)
        return out
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if task_id is None:
                continue
            if "any_compromised" in row:
                out[str(task_id)] = bool(row["any_compromised"])
            elif "attack_success" in row:
                out[str(task_id)] = bool(row["attack_success"])
    return out


def _resolve_sad_layer_threshold(
    sad_eval_path: str | None,
    requested_layer: int | None,
    requested_threshold: float | None,
    probe: ContrastiveProbe,
) -> tuple[int, float, dict]:
    layer = requested_layer if requested_layer is not None else probe.best_layer
    threshold = requested_threshold if requested_threshold is not None else probe.threshold
    source: dict = {"layer_source": "probe_info", "threshold_source": "probe_info"}

    if sad_eval_path:
        sad = json.loads(Path(sad_eval_path).read_text())
        split = sad.get("dev_test_split") or {}
        if requested_layer is None:
            if split.get("best_layer_by_dev") is not None:
                layer = int(split["best_layer_by_dev"])
                source["layer_source"] = "sad_stages_oversight_dev"
            elif sad.get("top_layers_all"):
                layer = int(sad["top_layers_all"][0]["layer"])
                source["layer_source"] = "sad_stages_oversight_all"
        if requested_threshold is None and layer is not None:
            found_threshold = None
            for key, threshold_source in (
                ("dev_top_layers", "sad_stages_oversight_dev"),
                ("top_layers_all", "sad_stages_oversight_all"),
            ):
                for item in (split.get(key) or sad.get(key) or []):
                    if int(item.get("layer")) == int(layer):
                        found_threshold = float(item["threshold"])
                        source["threshold_source"] = threshold_source
                        break
                if found_threshold is not None:
                    break
            if found_threshold is None:
                per_layer = sad.get("per_layer") or {}
                info = per_layer.get(str(layer))
                if info and "threshold" in info:
                    found_threshold = float(info["threshold"])
                    source["threshold_source"] = "sad_stages_oversight_per_layer"
            if found_threshold is not None:
                threshold = found_threshold

    if layer is None:
        raise ValueError("No layer specified and probe_info has no best_layer")
    return int(layer), float(threshold), source


def _score_browser_task(
    task_dir: Path,
    *,
    probe: ContrastiveProbe,
    model,
    tokenizer,
    layer: int,
    threshold: float,
    asr_map: dict[str, bool],
    max_steps: int | None,
    fallback_tokenization: str,
    browser_token_span: str,
) -> dict:
    task_id = _task_id_from_dir(task_dir)
    step_scores: list[dict] = []

    step_paths = sorted(task_dir.glob("step_*.pkl.gz"), key=_parse_step_index)
    if max_steps:
        step_paths = step_paths[:max_steps]

    for step_path in step_paths:
        step = _load_step_pickle(step_path)
        if step is None:
            continue
        agent_info = _safe_get(step, "agent_info", {})
        chat = _safe_get(agent_info, "chat_messages")
        messages = _normalize_chat_messages(_safe_get(chat, "messages", []))
        step_index = _parse_step_index(step_path)
        try:
            if messages:
                scored = _score_messages(
                    probe,
                    model,
                    tokenizer,
                    messages,
                    layer,
                    threshold,
                    span=browser_token_span,
                )
                tokenization = browser_token_span
            else:
                text_parts = []
                obs = _safe_get(step, "obs", {})
                goal = _safe_get(obs, "goal_object") or _safe_get(obs, "goal", "")
                if goal:
                    text_parts.append(f"[user]\n{_flatten_message_content(goal)}")
                thought = _safe_get(agent_info, "think", "")
                if thought:
                    text_parts.append(f"[assistant]\n{thought}")
                action = _safe_get(step, "action", "")
                if action:
                    text_parts.append(f"[assistant.action]\n{action}")
                observation = _safe_get(obs, "axtree_txt", "")
                if observation:
                    text_parts.append(f"[tool.observation]\n{observation[:5000]}")
                scored = _score_text(
                    probe,
                    model,
                    tokenizer,
                    "\n\n".join(text_parts),
                    layer,
                    threshold,
                    fallback_tokenization,
                )
                tokenization = fallback_tokenization
        except Exception as e:
            logger.warning("Failed to score %s step %s: %s", task_dir.name, step_index, e)
            continue
        scored.update({"step_index": step_index, "tokenization": tokenization})
        step_scores.append(scored)

    scores = [s["mean_score"] for s in step_scores]
    mean_score = mean(scores) if scores else 0.0
    classification, confidence = classify_score(mean_score, threshold)
    return {
        "id": task_id,
        "task_id": task_id,
        "task_dir": str(task_dir),
        "mean_score": mean_score,
        "max_score": max(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
        "classification": classification,
        "confidence": confidence,
        "layer": layer,
        "threshold": threshold,
        "n_steps_scored": len(step_scores),
        "attack_success": asr_map.get(task_id),
        "step_scores": step_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Score text with a saved DoM probe")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--probe-dir", required=True, help="Saved probe directory")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to score at (default: probe.best_layer)",
    )
    parser.add_argument("--text", default=None, help="Single text to score")
    parser.add_argument(
        "--transcripts-dir",
        default=None,
        help="Directory of ToolCallingRunner JSON-per-task outputs",
    )
    parser.add_argument(
        "--browser-root",
        default=None,
        help="BrowserGym/AgentLab root containing task dirs with step_*.pkl.gz",
    )
    parser.add_argument(
        "--asr-jsonl",
        default=None,
        help="Optional WASP ASR JSONL to attach attack_success labels",
    )
    parser.add_argument(
        "--sad-eval",
        default=None,
        help="Optional SAD oversight eval JSON. If --layer is omitted, uses "
             "dev best layer and threshold from this file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold override. Defaults to probe/SAD threshold.",
    )
    parser.add_argument(
        "--tokenization",
        choices=["raw", "chat_user_content", "chat_messages_mean"],
        default="raw",
        help="Text tokenization for --text/--transcripts-dir. Browser chat "
             "messages always use chat_messages_mean when available.",
    )
    parser.add_argument(
        "--browser-token-span",
        choices=[
            "all_messages_mean",
            "last_user_token",
            "user_message_mean",
            "first_assistant_message_mean",
        ],
        default="all_messages_mean",
        help="Token span to score when BrowserGym/AgentLab chat messages are available.",
    )
    parser.add_argument(
        "--messages-jsonl",
        default=None,
        help="JSONL rows with id/task_id/step_index/messages for post-hoc chat scoring.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-steps-per-task", type=int, default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if args.text is None and args.transcripts_dir is None and args.browser_root is None:
        if args.messages_jsonl is None:
            raise SystemExit("Provide --text, --transcripts-dir, --browser-root, or --messages-jsonl")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device, dtype=dtype)

    probe = ContrastiveProbe.load(args.probe_dir)
    layer, threshold, selection = _resolve_sad_layer_threshold(
        args.sad_eval,
        args.layer,
        args.threshold,
        probe,
    )
    logger.info(
        "Scoring at layer %s threshold %.6g (%s / %s)",
        layer,
        threshold,
        selection["layer_source"],
        selection["threshold_source"],
    )

    out_lines: list[str] = []
    rows: list[dict] = []

    if args.text is not None:
        result = _score_text(
            probe,
            model,
            tokenizer,
            args.text,
            layer,
            threshold,
            args.tokenization,
        )
        row = {
            "id": "text",
            **result,
            "layer": layer,
            "threshold": threshold,
            "tokenization": args.tokenization,
        }
        line = json.dumps(row)
        print(line)
        out_lines.append(line)
        rows.append(row)

    if args.transcripts_dir is not None:
        td = Path(args.transcripts_dir)
        task_paths = sorted(td.glob("*.json"))
        if args.max_items:
            task_paths = task_paths[:args.max_items]
        for task_path in task_paths:
            try:
                task_json = json.loads(task_path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Skipping non-JSON {task_path}")
                continue
            text = _flatten_toolcalling_task(task_json)
            try:
                result = _score_text(
                    probe,
                    model,
                    tokenizer,
                    text,
                    layer,
                    threshold,
                    args.tokenization,
                )
            except Exception as e:
                logger.warning(f"Failed to score {task_path.name}: {e}")
                continue
            row = {
                "id": task_json.get("task_id", task_path.stem),
                "benchmark": task_json.get("benchmark"),
                "agent": task_json.get("agent"),
                "condition": task_json.get("condition"),
                **result,
                "layer": layer,
                "threshold": threshold,
                "tokenization": args.tokenization,
                "attack_success": task_json.get("attack_success"),
            }
            line = json.dumps(row)
            print(line)
            out_lines.append(line)
            rows.append(row)

    if args.browser_root is not None:
        root = Path(args.browser_root)
        asr_map = _load_asr_map(args.asr_jsonl)
        task_dirs = _browser_task_dirs(root, max_items=args.max_items)
        logger.info("Scoring %s browser tasks under %s", len(task_dirs), root)
        for task_dir in task_dirs:
            row = _score_browser_task(
                task_dir,
                probe=probe,
                model=model,
                tokenizer=tokenizer,
                layer=layer,
                threshold=threshold,
                asr_map=asr_map,
                max_steps=args.max_steps_per_task,
                fallback_tokenization=args.tokenization,
                browser_token_span=args.browser_token_span,
            )
            line = json.dumps(row)
            print(line)
            out_lines.append(line)
            rows.append(row)

    if args.messages_jsonl is not None:
        rows_by_task: dict[str, list[dict]] = {}
        with Path(args.messages_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = _normalize_chat_messages(item.get("messages", []))
                if not messages:
                    continue
                result = _score_messages(
                    probe,
                    model,
                    tokenizer,
                    messages,
                    layer,
                    threshold,
                    span=args.browser_token_span,
                )
                result.update({
                    "step_index": item.get("step_index"),
                    "tokenization": args.browser_token_span,
                })
                task_id = str(item.get("task_id") or item.get("id") or "messages")
                rows_by_task.setdefault(task_id, []).append(result)

        for task_id, step_scores in rows_by_task.items():
            scores = [s["mean_score"] for s in step_scores]
            mean_score = mean(scores) if scores else 0.0
            classification, confidence = classify_score(mean_score, threshold)
            row = {
                "id": task_id,
                "task_id": task_id,
                "mean_score": mean_score,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
                "classification": classification,
                "confidence": confidence,
                "layer": layer,
                "threshold": threshold,
                "n_steps_scored": len(step_scores),
                "step_scores": step_scores,
                "tokenization": args.browser_token_span,
            }
            line = json.dumps(row)
            print(line)
            out_lines.append(line)
            rows.append(row)

    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + "\n")
        logger.info(f"Wrote {len(out_lines)} scored items to {args.output}")

    if args.summary_output:
        scores = [r["mean_score"] for r in rows]
        classified_eval = [r for r in rows if r.get("classification") == "eval_aware"]
        asr_rows = [r for r in rows if r.get("attack_success") is not None]
        summary = {
            "n": len(rows),
            "layer": layer,
            "threshold": threshold,
            **selection,
            "mean_score": mean(scores) if scores else None,
            "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
            "n_eval_aware": len(classified_eval),
            "eval_aware_rate": len(classified_eval) / len(rows) if rows else None,
            "n_with_asr": len(asr_rows),
            "n_attack_success": sum(1 for r in asr_rows if r.get("attack_success")),
            "asr": (
                sum(1 for r in asr_rows if r.get("attack_success")) / len(asr_rows)
                if asr_rows
                else None
            ),
        }
        Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("Wrote score summary to %s", args.summary_output)


if __name__ == "__main__":
    main()
