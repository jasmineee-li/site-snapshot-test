from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ResolverError(RuntimeError):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class ResolvedCall:
    method: str
    url: str
    headers: dict[str, str]
    body: Any = None
    context_additions: dict[str, Any] = field(default_factory=dict)
