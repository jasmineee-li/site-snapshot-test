"""Load and render YAML prompt templates."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PromptTemplate:
    id: str
    type: str  # "classify" | "compare" | "open_ended"
    question: str
    modalities: list[str]
    preamble: str | None = None
    choices: list[str] | None = None  # e.g. ["YES", "NO"] or ["A", "B"]
    label_mapping: dict | None = None  # maps choices to canonical labels


def load_prompt(path: Path) -> PromptTemplate:
    """Load a single prompt template from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return PromptTemplate(
        id=data["id"],
        type=data["type"],
        question=data["question"],
        modalities=data["modalities"],
        preamble=data.get("preamble"),
        choices=data.get("choices"),
        label_mapping=data.get("label_mapping"),
    )


def load_all_prompts(prompts_dir: Path) -> dict[str, PromptTemplate]:
    """Load all YAML prompt templates from a directory."""
    templates = {}
    for path in sorted(prompts_dir.glob("*.yaml")):
        t = load_prompt(path)
        templates[t.id] = t
    return templates


def _build_item_content(item: dict, modality: str) -> tuple[str, list[Path]]:
    """Build text content and image list for an item given a modality.

    Returns (text_content, image_paths).
    """
    item_path = Path(item["path"])
    text = ""
    images: list[Path] = []

    if modality == "html":
        html_path = item_path / "page.html"
        if html_path.exists():
            text = html_path.read_text()
        else:
            # Multi-file html/ directory — concatenate all HTML files
            html_dir = item_path / "html"
            if html_dir.is_dir():
                parts = []
                for hp in sorted(html_dir.glob("*.html")):
                    parts.append(f"<!-- {hp.name} -->\n{hp.read_text()}")
                text = "\n\n".join(parts)
    elif modality == "axtree":
        ax_path = item_path / "axtree.txt"
        if ax_path.exists():
            text = ax_path.read_text()
    elif modality == "transcript":
        t_path = item_path / "transcript.json"
        if t_path.exists():
            text = t_path.read_text()
    elif modality == "screenshot":
        img_path = item_path / "screenshot.png"
        if img_path.exists():
            images.append(img_path)

    return text, images


def render_classify(
    template: PromptTemplate, item: dict, modality: str
) -> tuple[list[dict], list[Path]]:
    """Render a classify prompt for a single item.

    Returns (messages, image_paths).
    """
    text_content, images = _build_item_content(item, modality)

    parts = []
    if text_content:
        parts.append(f"--- BEGIN CONTENT ---\n{text_content}\n--- END CONTENT ---\n")
    if template.preamble:
        parts.append(template.preamble.strip())
    parts.append(template.question.strip())
    if template.choices:
        parts.append(f"Answer with one of: {', '.join(template.choices)}")

    user_text = "\n\n".join(parts)
    messages = [{"role": "user", "content": user_text}]
    return messages, images


def render_compare(
    template: PromptTemplate,
    item_a: dict,
    item_b: dict,
    modality: str,
    rng: random.Random | None = None,
) -> tuple[list[dict], list[Path], dict]:
    """Render a pairwise comparison prompt.

    Randomizes A/B position to control for position bias.

    Returns (messages, image_paths, position_map) where
    position_map = {"A": item_id, "B": item_id}.
    """
    if rng is None:
        rng = random.Random()

    # Randomize order
    if rng.random() < 0.5:
        first, second = item_a, item_b
    else:
        first, second = item_b, item_a

    position_map = {"A": first["id"], "B": second["id"]}

    text_a, images_a = _build_item_content(first, modality)
    text_b, images_b = _build_item_content(second, modality)

    parts = []
    if text_a:
        parts.append(f"--- WEBSITE A ---\n{text_a}\n--- END WEBSITE A ---")
    if text_b:
        parts.append(f"--- WEBSITE B ---\n{text_b}\n--- END WEBSITE B ---")

    if template.preamble:
        parts.append(template.preamble.strip())
    parts.append(template.question.strip())
    if template.choices:
        parts.append(f"Answer with one of: {', '.join(template.choices)}")

    user_text = "\n\n".join(parts)
    messages = [{"role": "user", "content": user_text}]
    images = images_a + images_b
    return messages, images, position_map


def render_open_ended(
    template: PromptTemplate, item: dict, modality: str
) -> tuple[list[dict], list[Path]]:
    """Render an open-ended prompt for a single item.

    Returns (messages, image_paths).
    """
    text_content, images = _build_item_content(item, modality)

    parts = []
    if text_content:
        parts.append(f"--- BEGIN CONTENT ---\n{text_content}\n--- END CONTENT ---\n")
    if template.preamble:
        parts.append(template.preamble.strip())
    parts.append(template.question.strip())

    user_text = "\n\n".join(parts)
    messages = [{"role": "user", "content": user_text}]
    return messages, images
