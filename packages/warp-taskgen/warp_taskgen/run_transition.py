"""Pure transition from CLI inputs and persisted state to one Run identity."""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from warp_taskgen.run_definition import (
    _project_requested_definition,
    define_run,
)
from warp_taskgen.run_definition_contracts import (
    RunDefinition,
    RunTransition,
)


def resolve_run_request(
    effective_inputs: Mapping[str, object],
    *,
    existing_state: Mapping[str, object] | None,
    new_run_id: str | None = None,
) -> RunTransition:
    """Resolve a CLI dispatch without writing state or materializing a child Run."""

    if not isinstance(effective_inputs, Mapping):
        raise ValueError("effective_inputs must be a mapping")
    if existing_state is None:
        identifier = new_run_id or f"run-{uuid.uuid4().hex}"
        projected = define_run(effective_inputs)
        definition = RunDefinition(
            schema_version=projected.schema_version,
            run_id=identifier,
            source_run_id=None,
            definition_digest=projected.definition_digest,
            contributions=projected.contributions,
            legacy=False,
        )
        return RunTransition(kind="new", definition=definition, source_definition=None)
    if not isinstance(existing_state, Mapping):
        raise ValueError("existing_state must be a mapping or null")

    source = define_run(existing_state)
    requested, drift = _project_requested_definition(
        source,
        effective_inputs,
        persisted_state=existing_state,
    )
    if source.legacy:
        # Envelope-less runs retain the historical mutable resume contract.
        # They have no identity to misattribute and are never upgraded in place.
        return RunTransition(kind="legacy", definition=source, source_definition=source)
    if drift:
        return RunTransition(
            kind="derived_required",
            definition=requested,
            source_definition=source,
            drift_fields=drift,
            reason_code="definition_drift_requires_isolated_child_run",
        )
    return RunTransition(kind="exact", definition=source, source_definition=source)


__all__ = ["RunTransition", "resolve_run_request"]
