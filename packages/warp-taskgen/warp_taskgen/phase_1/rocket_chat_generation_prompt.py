"""Model-facing prompt contract for generated Rocket.Chat conversations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.rocket_chat_generated_content import (
    ROCKET_CHAT_GENERATED_CONTENT_KEY,
)
from warp_taskgen.phase_1.rocket_chat_generation import (
    ROCKET_CHAT_DECISION_GENERATION_FAMILY,
    ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION,
    ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY,
    ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION,
)


def rocket_chat_generation_prompt_addendum(
    task_card_plan: Mapping[str, Any] | None,
) -> str:
    """Describe the only model-owned Rocket.Chat output slots."""

    if not isinstance(task_card_plan, Mapping):
        return ""
    contracts: list[dict[str, str]] = []
    for card in task_card_plan.get("task_cards", []):
        if not isinstance(card, Mapping) or str(card.get("status", "active")) != "active":
            continue
        raw = card.get("generation_contract")
        if not isinstance(raw, Mapping):
            continue
        family = raw.get("family")
        if family not in {
            ROCKET_CHAT_DECISION_GENERATION_FAMILY,
            ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY,
        }:
            continue
        expected_composition = (
            ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION
            if family == ROCKET_CHAT_DECISION_GENERATION_FAMILY
            else ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION
        )
        summary = {
            "task_card_id": str(card.get("id") or ""),
            "family": str(family),
            "runtime_composition": str(raw.get("runtime_composition") or expected_composition),
        }
        if family == ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY:
            host = raw.get("host")
            host = host if isinstance(host, Mapping) else raw
            summary["notification_recipient"] = str(host.get("notification_recipient") or "")
        contracts.append(summary)
    if not contracts:
        return ""
    return f"""

<rocket_chat_generation>
The active task-card plan includes an exact Rocket.Chat runtime composition.
The host owns the benchmark (TheAgentCompany), Site, room, thread graph,
writer and reader identities, evaluator, seed, notification action and
provenance. For a task whose `task_card_id` matches one of the cards below,
add exactly one top-level `{ROCKET_CHAT_GENERATED_CONTENT_KEY}` object to the
normal task object. It is the only model-owned field and contains substantive
semantic facts, not workflow structure:

```json
{{
  "{ROCKET_CHAT_GENERATED_CONTENT_KEY}": {{
    "initial_decision": {{"owner": "...", "due_date": "..."}},
    "corrected_decision": {{"owner": "...", "due_date": "..."}},
    "messages": [
      {{"slot": "plan", "text": "..."}},
      {{"slot": "update", "text": "..."}},
      {{"slot": "correction", "text": "..."}}
    ]
  }}
}}
```

Emit one each of `plan`, `update`, and `correction`; make all prose
substantive and distinct, and change both decision fields in the confirmed
correction. For a notification contract, the corrected owner must equal that
contract's exact `notification_recipient`. Do not emit `room_id`, `thread_key`,
`writer_user`, `reader_user`, `benchmark`, `site`, `start_urls`, IDs, editor
methods, `data_seed`, `reward_function`, evaluator settings, an action
contract, or a notification recipient. Do not add another generated-content
field. The host derives notification recipient/body from the corrected owner.

Active exact contracts:
{json.dumps(contracts, sort_keys=True)}
</rocket_chat_generation>
""".strip()


__all__ = ["rocket_chat_generation_prompt_addendum"]
