"""Types shared by Phase 2a planning behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SiteInjectionResult:
    site_name: str
    adversarial_tasks: list[dict]
    errors: list[str]
