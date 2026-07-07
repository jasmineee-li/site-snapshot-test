"""Score the jjpn2/eval_awareness transcripts with a saved chat-template probe.

This is intentionally close to the post-WASP probe scoring block in
``launch_guiowl32_wasp_full_attacked_with_probes.sh``: it uses the same
message normalization, SAD layer/threshold selection, and span names, but
adds dataset metadata summaries for the Hugging Face benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from statistics import mean, pstdev

import torch

from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model
from probes.score_probe import (
    _chat_message_content_positions,
    _filter_content_positions,
    _normalize_chat_messages,
    _resolve_sad_layer_threshold,
    _score_tokens,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_SPANS = (
    "last_user_token",
    "user_message_mean",
    "first_assistant_message_mean",
)


def _load_metadata(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def _token_window(
    *,
    ids: list[int],
    positions: list[int],
    span: str,
    max_tokens: int | None,
) -> tuple[list[int], list[int], dict]:
    if not max_tokens or len(ids) <= max_tokens:
        return ids, positions, {
            "truncated": False,
            "token_start": 0,
            "token_end": len(ids),
            "full_n_tokens": len(ids),
        }

    if positions:
        if span == "last_user_token":
            end = min(len(ids), positions[-1] + 1)
            start = max(0, end - max_tokens)
        elif span == "first_assistant_message_mean":
            start = max(0, positions[0] - 256)
            end = min(len(ids), start + max_tokens)
        else:
            if positions[-1] - positions[0] + 1 <= max_tokens:
                start = max(0, positions[0] - 256)
            else:
                start = 0
            end = min(len(ids), start + max_tokens)
    else:
        start = 0
        end = min(len(ids), max_tokens)

    if end - start < max_tokens and start > 0:
        start = max(0, end - max_tokens)

    window_positions = [pos - start for pos in positions if start <= pos < end]
    if not window_positions:
        window_positions = list(range(end - start))

    return ids[start:end], window_positions, {
        "truncated": True,
        "token_start": start,
        "token_end": end,
        "full_n_tokens": len(ids),
    }


def _positions_for_span(tokenizer, rendered: str, messages: list[dict], span: str) -> list[int]:
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if span == "all_messages_mean":
        return _filter_content_positions(tokenizer, ids, range(len(ids)))
    return _chat_message_content_positions(tokenizer, rendered, messages, span)


def _score_item_spans(
    *,
    item: dict,
    model,
    tokenizer,
    probe: ContrastiveProbe,
    layer: int,
    threshold: float,
    spans: tuple[str, ...],
    max_tokens: int | None,
) -> list[dict]:
    messages = _normalize_chat_messages(item.get("messages", []))
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(rendered, add_special_tokens=False)
    rows = []
    for span in spans:
        if span in {"first_assistant_message_mean"} and not any(
            message.get("role") == "assistant" and str(message.get("content", "")).strip()
            for message in messages
        ):
            LOGGER.debug("skipping %s for %s: no assistant message", span, item.get("id"))
            continue
        if span in {"last_user_token", "user_message_mean"} and not any(
            message.get("role") == "user" and str(message.get("content", "")).strip()
            for message in messages
        ):
            LOGGER.debug("skipping %s for %s: no user message", span, item.get("id"))
            continue
        positions = _positions_for_span(tokenizer, rendered, messages, span)
        if not positions and span == "all_messages_mean":
            positions = list(range(len(full_ids)))
        elif not positions:
            LOGGER.debug("skipping %s for %s: no content positions", span, item.get("id"))
            continue
        window_ids, window_positions, truncation = _token_window(
            ids=full_ids,
            positions=positions,
            span=span,
            max_tokens=max_tokens,
        )
        tokens = torch.tensor([window_ids], device=model.device)
        scored = _score_tokens(
            probe,
            model,
            tokens,
            window_positions,
            layer,
            threshold,
        )
        rows.append({
            "id": item.get("id"),
            "task_id": item.get("task_id") or item.get("id"),
            "step_index": item.get("step_index"),
            "span": span,
            "layer": layer,
            "threshold": threshold,
            **scored,
            **truncation,
        })
    return rows


def _group_key(row: dict, key: str) -> str:
    if key == "all":
        return "all"
    return str(row.get(key, ""))


def _summarize(rows: list[dict], meta_by_id: dict[str, dict], selection: dict) -> dict:
    enriched = []
    for row in rows:
        meta = meta_by_id.get(str(row.get("task_id")), {})
        enriched.append({**row, **meta})

    summary = {
        "n_rows": len(rows),
        "n_items": len({row.get("task_id") for row in rows}),
        "layer": rows[0]["layer"] if rows else None,
        "threshold": rows[0]["threshold"] if rows else None,
        **selection,
        "spans": {},
    }

    group_fields = ("all", "eval_category", "is_real", "style", "category", "eval_type")
    for span in sorted({row["span"] for row in enriched}):
        span_rows = [row for row in enriched if row["span"] == span]
        summary["spans"][span] = {}
        for field in group_fields:
            groups = sorted({_group_key(row, field) for row in span_rows})
            summary["spans"][span][field] = {}
            for group in groups:
                group_rows = [row for row in span_rows if _group_key(row, field) == group]
                scores = [float(row["mean_score"]) for row in group_rows]
                summary["spans"][span][field][group] = {
                    "n": len(group_rows),
                    "mean_score": mean(scores) if scores else None,
                    "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None,
                    "n_eval_aware": sum(1 for row in group_rows if row["classification"] == "eval_aware"),
                    "eval_aware_rate": (
                        sum(1 for row in group_rows if row["classification"] == "eval_aware")
                        / len(group_rows)
                        if group_rows else None
                    ),
                    "mean_n_positions": (
                        mean([int(row["n_positions"]) for row in group_rows])
                        if group_rows else None
                    ),
                    "n_truncated": sum(1 for row in group_rows if row.get("truncated")),
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages-jsonl", required=True)
    parser.add_argument("--metadata-csv")
    parser.add_argument("--model", default="mPLUG/GUI-Owl-1.5-32B-Think")
    parser.add_argument(
        "--probe-dir",
        default="probes/trained/gui-owl-32b-chat-template-span/framing",
    )
    parser.add_argument(
        "--sad-eval",
        default="probes/trained/gui-owl-32b-chat-template-span/sad_stages_oversight_eval_framing.json",
    )
    parser.add_argument("--layer", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--spans", default=",".join(DEFAULT_SPANS))
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    spans = tuple(span.strip() for span in args.spans.split(",") if span.strip())

    LOGGER.info("loading model %s", args.model)
    model, tokenizer = load_model(args.model, device=args.device, dtype=dtype)
    probe = ContrastiveProbe.load(args.probe_dir)
    layer, threshold, selection = _resolve_sad_layer_threshold(
        args.sad_eval,
        args.layer,
        args.threshold,
        probe,
    )
    if args.sad_eval and args.layer is None and args.threshold is None:
        sad = json.loads(Path(args.sad_eval).read_text())
        top_layers_all = sad.get("top_layers_all") or []
        if top_layers_all:
            layer = int(top_layers_all[0]["layer"])
            threshold = float(top_layers_all[0]["threshold"])
        selection["layer_source"] = "sad_stages_oversight_all"
        selection["threshold_source"] = "sad_stages_oversight_all"
    LOGGER.info("probe layer=%s threshold=%s selection=%s", layer, threshold, selection)

    items = []
    with Path(args.messages_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
            if args.max_items and len(items) >= args.max_items:
                break
    LOGGER.info("scoring items=%s spans=%s max_tokens=%s", len(items), spans, args.max_tokens)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items, 1):
            if idx == 1 or idx % 25 == 0:
                LOGGER.info("item %s/%s %s", idx, len(items), item.get("id"))
            try:
                item_rows = _score_item_spans(
                    item=item,
                    model=model,
                    tokenizer=tokenizer,
                    probe=probe,
                    layer=layer,
                    threshold=threshold,
                    spans=spans,
                    max_tokens=args.max_tokens,
                )
            except torch.cuda.OutOfMemoryError:
                LOGGER.exception("CUDA OOM on item %s", item.get("id"))
                torch.cuda.empty_cache()
                continue
            for row in item_rows:
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()

    meta_by_id = _load_metadata(Path(args.metadata_csv) if args.metadata_csv else None)
    summary = _summarize(rows, meta_by_id, selection)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s rows=%s", out_path, len(rows))
    LOGGER.info("wrote %s", args.summary_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
