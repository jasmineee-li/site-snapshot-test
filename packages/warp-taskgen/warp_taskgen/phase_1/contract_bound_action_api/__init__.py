"""Contract-bound Phase 1 action-task generation via host API.

This backend is for task-card plans whose benign utility is host-action-only.
The model emits semantic slots; host code owns card, route, editor, reward,
and provenance fields before the normal Phase 1 validator admits the tasks.

Behavior is split by what each module owns: `contract_selection` picks the
card/route/anchor contracts, `instruction_validation` admits slots and
instructions, `prompt_rendering` builds the generation prompt,
`slot_compilation` compiles an accepted slot into a benign task, and
`slot_generation` runs the host API call.
"""

from __future__ import annotations

from warp_taskgen.phase_1.contract_bound_action_api.contract_selection import (
    SelectedActionTaskContract,
    select_action_task_contracts,
)
from warp_taskgen.phase_1.contract_bound_action_api.prompt_rendering import (
    contract_bound_prompt_inputs,
)
from warp_taskgen.phase_1.contract_bound_action_api.slot_compilation import (
    compile_action_task_slot,
)
from warp_taskgen.phase_1.contract_bound_action_api.slot_generation import (
    build_emit_action_task_slots_tool,
    contract_bound_tool_schema_digest,
    generate_contract_bound_action_tasks_api,
)

__all__ = [
    "SelectedActionTaskContract",
    "build_emit_action_task_slots_tool",
    "compile_action_task_slot",
    "contract_bound_prompt_inputs",
    "contract_bound_tool_schema_digest",
    "generate_contract_bound_action_tasks_api",
    "select_action_task_contracts",
]
