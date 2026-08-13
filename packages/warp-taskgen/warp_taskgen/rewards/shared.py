from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _records_contain_witness(records: Any, witness: str, *, fields: tuple[str, ...]) -> bool:
    if isinstance(records, Mapping):
        iterable = [records]
    elif isinstance(records, list):
        iterable = records
    else:
        return False
    for record in iterable:
        if not isinstance(record, Mapping):
            continue
        for field in fields:
            value = record.get(field)
            if isinstance(value, str) and witness in value:
                return True
    return False


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in out:
            out.append(value)
    return out


def _state_probe_kind(state_probe: Mapping[str, Any] | None) -> str:
    if not isinstance(state_probe, Mapping):
        return ""
    return str(state_probe.get("kind") or "").strip()
