"""Load manifest, sample items, and build pairwise pairs."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataItem:
    id: str
    source: str  # "ours", "webarena", "real", etc.
    label: str  # "synthetic" or "real"
    modalities: list[str]
    path: Path
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "label": self.label,
            "modalities": self.modalities,
            "path": str(self.path),
            "metadata": self.metadata,
        }


def load_manifest(manifest_path: Path) -> list[DataItem]:
    """Load data items from a JSONL manifest file."""
    items = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                DataItem(
                    id=data["id"],
                    source=data["source"],
                    label=data["label"],
                    modalities=data["modalities"],
                    path=Path(data["path"]),
                    metadata=data.get("metadata", {}),
                )
            )
    return items


def sample_items(
    items: list[DataItem],
    per_source: dict[str, int],
    modality_filter: list[str] | None = None,
    seed: int = 42,
) -> list[DataItem]:
    """Sample items from the manifest, grouped by source.

    Args:
        items: All available data items.
        per_source: How many items to sample per source, e.g. {"ours": 50, "real": 50}.
        modality_filter: If set, only include items that have all of these modalities.
        seed: Random seed for reproducibility.

    Returns:
        Sampled list of DataItem.
    """
    rng = random.Random(seed)

    # Filter by modality
    if modality_filter:
        items = [it for it in items if all(m in it.modalities for m in modality_filter)]

    # Group by source
    by_source: dict[str, list[DataItem]] = {}
    for it in items:
        by_source.setdefault(it.source, []).append(it)

    sampled = []
    for source, count in per_source.items():
        available = by_source.get(source, [])
        if len(available) < count:
            raise ValueError(
                f"Requested {count} items from source '{source}' but only "
                f"{len(available)} available (after modality filter)."
            )
        sampled.extend(rng.sample(available, count))

    return sampled


def make_pairs(
    items: list[DataItem],
    pair_sources: tuple[str, str],
    n_pairs: int,
    seed: int = 42,
) -> list[tuple[DataItem, DataItem]]:
    """Create pairwise pairs from two different sources.

    Args:
        items: All available data items.
        pair_sources: The two sources to pair, e.g. ("ours", "real").
        n_pairs: Number of pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (item_from_source_a, item_from_source_b) tuples.
    """
    rng = random.Random(seed)

    source_a, source_b = pair_sources
    pool_a = [it for it in items if it.source == source_a]
    pool_b = [it for it in items if it.source == source_b]

    if len(pool_a) < n_pairs:
        raise ValueError(
            f"Need {n_pairs} items from '{source_a}' but only {len(pool_a)} available."
        )
    if len(pool_b) < n_pairs:
        raise ValueError(
            f"Need {n_pairs} items from '{source_b}' but only {len(pool_b)} available."
        )

    sampled_a = rng.sample(pool_a, n_pairs)
    sampled_b = rng.sample(pool_b, n_pairs)

    return list(zip(sampled_a, sampled_b, strict=True))
