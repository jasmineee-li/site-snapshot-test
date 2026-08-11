---
status: accepted
---

# Define resume from an immutable Run Definition

Every Run will have one immutable, versioned Run Definition containing its normalized effective inputs. Start, resume, status, checkpoint fingerprints, and drift reporting will consume that definition rather than maintain separate option lists. A persisted opaque Run ID identifies one execution, while a deterministic Definition Digest compares semantic inputs; separately started Runs may therefore have equal definitions and different identities. An exact resume retains the Run ID, while a result-affecting override creates a Derived Run with a new Run ID and a reference to its source Run, and it may reuse only checkpoints that a read-only Resume Plan proves compatible. The public resumability interface will be limited to `define_run` and `plan_resume`; feature-owned definition contributors normalize and classify inputs, and separate feature-owned checkpoint policies decide compatibility. The first resumability slice will add Run Definition and Resume Plan behavior without changing lifecycle state. A later slice will add cooperative checkpoint-aligned pause semantics: `pausing` stops new scheduling, current atomic work may finish, `paused` is then persisted, and abrupt termination is represented as `interrupted` rather than deliberate pause. This accepts explicit Run derivation and lifecycle states to preserve provenance, explain reuse, and keep existing atomic artifact guarantees.

The first slice is deliberately a read-only projection over the non-secret, result-affecting inputs already present in pipeline state. It exposes a deterministic Definition Digest and an advisory Resume Plan through `status`; legacy runs remain explicitly `legacy` and receive no invented Run ID. Feature-owned Phase 2 and Phase 4 validators remain the only authorities that may accept or reject their checkpoints, so the plan reports those families as `not_inspected`, including when definition drift requires their validators to reassess compatibility. Persisting opaque Run IDs, creating Derived Run lineage, and making start/resume consume the definition are the next slice after this projection has passed one compatibility cycle. No `pausing`, `paused`, `interrupted`, lease, or worker-quiescence state is introduced here.

The second slice persists an opaque Run ID and the immutable Run Definition for
new CLI-orchestrated state roots. A state-root-scoped context injects the same
envelope into the discovery pointer and authoritative atomic checkpoint; exact
continuations retain the envelope, while legacy roots remain identity-less.
Result-affecting drift on an identified Run fails before dispatch and leaves the
source root unchanged. Creating the isolated child root, allocating its new ID,
setting `source_run_id`, and transferring only feature-validator-approved
checkpoints is deliberately deferred to the next slice because doing so inside
the parent root would corrupt provenance. This slice does not add lifecycle,
pause, lease, worker, or checkpoint-acceptance behavior.

At this boundary, "effective inputs" means the allowlisted CLI inputs after
shared static-default normalization. Absent optional paths are omitted rather
than frozen as `null`. Values and signatures resolved only inside a feature
runner remain owned by that feature's existing checkpoint fingerprint and do
not retroactively mutate the persisted definition. Bringing those contributors
to a pre-dispatch resolver is a future deepening step, not permission for the
top-level digest to accept a feature checkpoint.

The third slice materializes result-affecting drift as an isolated child Run.
For one identified source root and requested Definition Digest, WARP atomically
reserves one sibling child root and persists a random opaque child Run ID in
that reservation. Retries recover the same reservation and ID. The child
definition records the source Run ID, starts from the existing `phase_0a`
failed/rerun lifecycle state, and contains no copied phase or task artifacts;
the owning Phase 2 and Phase 4 validators therefore remain the only checkpoint
reuse authorities. Materialization re-reads the authoritative source state,
fails closed on source or reservation drift, and never writes the source state
or the process-wide resume pointer. The CLI prints an explicit child-root
resume command with a child-local resume pointer instead of automatically
dispatching it. Automatic child
execution, checkpoint transfer, and pause/lease semantics remain later slices.
