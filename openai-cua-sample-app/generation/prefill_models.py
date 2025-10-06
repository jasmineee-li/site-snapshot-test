"""Data models for prefill data generation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class PrefillEntity:
    """Specification for a prefill data entity."""

    page: str  # Which page this appears on
    description: str  # Detailed description of what data is needed
    count: int  # How many items
    data_type: str  # General type hint: "list", "table", "cards", etc.
    attributes: List[str]  # What fields each item should have
    special_requirements: List[str] = field(default_factory=list)
    example_item: Dict[str, Any] = field(default_factory=dict)  # Example of one item


@dataclass
class PrefillSpec:
    """Specification of required prefill data for a benchmark."""

    entities: List[PrefillEntity]
    required_tools: List[str]
    required_ui_elements: List[Dict[str, str]]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrefillSpec":
        """Create from dictionary (e.g., from JSON)."""
        entities = []
        for e_data in d.get("entities", []):
            entity = PrefillEntity(
                page=e_data["page"],
                description=e_data["description"],
                count=e_data["count"],
                data_type=e_data.get("data_type", "list"),
                attributes=e_data["attributes"],
                special_requirements=e_data.get("special_requirements", []),
                example_item=e_data.get("example_item", {}),
            )
            entities.append(entity)

        return cls(
            entities=entities,
            required_tools=d.get("required_tools", []),
            required_ui_elements=d.get("required_ui_elements", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entities": [
                {
                    "page": e.page,
                    "description": e.description,
                    "count": e.count,
                    "data_type": e.data_type,
                    "attributes": e.attributes,
                    "special_requirements": e.special_requirements,
                    "example_item": e.example_item,
                }
                for e in self.entities
            ],
            "required_tools": self.required_tools,
            "required_ui_elements": self.required_ui_elements,
        }


@dataclass
class PrefillData:
    """Generated prefill data organized by page."""

    data_by_page: Dict[str, List[Dict[str, Any]]]

    def get_for_page(self, page: str) -> List[Dict[str, Any]]:
        """Get prefill data for a specific page."""
        return self.data_by_page.get(page, [])

    def save(self, path: Path):
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data_by_page, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PrefillData":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)
