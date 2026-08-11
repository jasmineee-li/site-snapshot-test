# WARP Taskgen resumability architecture research

**Date:** 2026-08-10
**Scope:** current Taskgen code, tests, technical specification, and package
guides. These are the primary sources for this note; no external precedent is
needed to describe the current behavior. This is research and sequencing input,
not an architectural decision.

## Executive readout

- The effective run is currently implicit: the canonical state directory
  (`logs` or `WARP_TASKGEN_STATE_DIR`, with the historical
  `WORLDSIM_STATE_DIR` alias), the `pipeline_state.json` checkpoint, and—once
  Phase 4 starts—the persisted `task_dir_root`. The state payload is an open
  dictionary with `step`, `iteration`, `timestamp`, `logs_dir`, and arbitrary
  metadata; it does not define a stable run id, schema version, or lease.
  [`worldsim/state.py:23-75`](../../worldsim/state.py#L23-L75)

- The durable boundary is atomic replacement plus a discovery mirror. State
  writes the default `last_run_state.json` mirror first and then atomically
  replaces the authoritative state file. Loading rejects malformed candidates,
  enforces an explicit state-directory override, and chooses the newest valid
  candidate. This is good process-crash recovery, but the write helper does not
  `fsync`; “durable” currently means atomic replacement and recovery from the
  surviving files, not a documented power-loss guarantee.
  [`worldsim/state.py:48-89`](../../worldsim/state.py#L48-L89)
  [`worldsim/state.py:97-218`](../../worldsim/state.py#L97-L218)
  [`worldsim/atomic_io.py:23-76`](../../worldsim/atomic_io.py#L23-L76)

- Pipeline resume has one routing state machine: `complete` and
  `partial_complete` advance to the next fixed phase; `running` and `failed`
  rerun the same phase; any other status is an error. A running checkpoint is
  therefore interpreted as “likely crashed,” not “operator intentionally
  paused.” [`worldsim/cli/_impl.py:1503-1559`](../../worldsim/cli/_impl.py#L1503-L1559)

- Phase 4 has the strongest resumability design. It reuses a persisted task
  root, treats an atomically written valid `result.json` as the initial-task
  completion sentinel, requires a matching execution fingerprint and required
  sidecars, and layers postprocess, placement, strategy, variant, and iterator
  checkpoints on top. [`docs/warp-taskgen-technical-spec.md:403-417`](../warp-taskgen-technical-spec.md#per-task-resume-phase-4)
  [`worldsim/eval_worker_pool.py:184-275`](../../worldsim/eval_worker_pool.py#L184-L275)

- Provenance is feature-local rather than one pipeline-wide contract. Phase 4
  fingerprints task data, reachable instances, URL placeholders, auth context,
  model/runner/provider/timeouts, benchmark root, profile, and variant versions;
  Phase 2 uses settings/signatures and feasibility fingerprint/TTL checks. This
  prevents many false reuses while preserving locality (unrelated sites do not
  invalidate a task). [`worldsim/resume_metadata.py:9-36`](../../worldsim/resume_metadata.py#L9-L36)
  [`worldsim/phase_4/resume.py:130-233`](../../worldsim/phase_4/resume.py#L130-L233)
  [`worldsim/phase_2/reuse.py:11-189`](../../worldsim/phase_2/reuse.py#L11-L189)

- `status` is already a rich read-only operator card: it combines pipeline
  state, Phase 4 heartbeat and summaries, cost, artifact provenance,
  reachability, Phase 2c admission/reasons, and task-bank data. The missing
  explanation layer is which checkpoint candidate was accepted, why a saved
  artifact was reused or rejected, what the effective run identity is, and
  whether an operator stop was intentional. [`worldsim/cli_status.py:167-238`](../../worldsim/cli_status.py#L167-L238)
  [`worldsim/cli_status.py:330-365`](../../worldsim/cli_status.py#L330-L365)

The smallest safe slice is consequently an additive, read-only run/checkpoint
explanation view built from the existing paths and statuses. It should precede
any new pause state or lease protocol. A pause feature would change worker
quiescence, benchmark reset, evidence, and remote-job semantics; the current
code has no operator pause request or pause-aware state transition to extend.

## 1. Current effective run definition

### What identifies a run today

1. `get_state_dir()` resolves `logs` or the canonical WARP environment override,
   falling back to the legacy environment variable. `pipeline_state.json` is
   always relative to that directory; a pointer at the default `logs` path
   discovers custom directories. [`worldsim/state.py:23-45`](../../worldsim/state.py#L23-L45)
   [`worldsim/state.py:97-142`](../../worldsim/state.py#L97-L142)
2. `save_state()` records the current phase, status, timestamp, state directory,
   and arbitrary phase metadata. Phase 4 adds `task_dir_root`, which is a
   timestamped directory on a fresh run and is explicitly reused on resume.
   [`worldsim/state.py:48-75`](../../worldsim/state.py#L48-L75)
   [`worldsim/phase_4/runner.py:19-35`](../../worldsim/phase_4/runner.py#L19-L35)
3. Remote operation supplies a job id, explicit state directory, and expected
   output in a separate registry. The launch guide calls those an operational
   contract, not a pipeline-level identity field. [`agent_docs/remote-runs.md:7-23`](../../agent_docs/remote-runs.md#L7-L23)
   [`agent_docs/remote-runs.md:150-153`](../../agent_docs/remote-runs.md#L150-L153)

This makes “same run” mean “the same state root and (for Phase 4) task root,”
provided the operator did not copy or reuse those paths for a different
invocation. That is workable for the current CLI but ambiguous for archives,
host migration, copied artifacts, or two operators sharing a state directory.
There is also no common schema/version field at the pipeline state boundary;
`progress.json` and several feature artifacts have their own schema/version
fields, but `pipeline_state.json` does not.

### Checkpoint lifecycle by layer

| Owner | Durable artifact and boundary | Reuse/routing rule | Source |
| --- | --- | --- | --- |
| Pipeline | `pipeline_state.json` plus `last_run_state.json`; atomic JSON replacement and pointer fallback | `complete`/`partial_complete` advance; `running`/`failed` rerun the same phase | [`state.py`](../../worldsim/state.py), [`_impl.py`](../../worldsim/cli/_impl.py#L1526-L1559) |
| Phase 2 | Plans, text-fill diagnostics, tasks, and feasibility artifacts; state records stage and settings | Reuse only for matching phase/status, settings/signatures, valid files, and feasibility fingerprint/TTL; otherwise regenerate | [`phase_2/runner.py:103-128`](../../worldsim/phase_2/runner.py#L103-L128), [`phase_2/phase_2c/stage.py:41-83`](../../worldsim/phase_2/phase_2c/stage.py#L41-L83), [`technical spec:363-369`](../warp-taskgen-technical-spec.md#cli-flags) |
| Phase 4 initial evaluation | Per-task `result.json`, written atomically; history and runtime sidecars are evidence, not a completion sentinel by themselves | Valid canonical result + expected fingerprint + required sidecars is reusable; history-only, corrupt, stale, suffixed, or incomplete artifacts rerun | [`trajectory.py:29-88`](../../worldsim/trajectory.py#L29-L88), [`eval_worker_pool.py:184-275`](../../worldsim/eval_worker_pool.py#L184-L275) |
| Phase 4 postprocess | `processed_result.json` with `_source_fingerprint` | Matching source fingerprint reuses postprocess; stale/malformed file is ignored | [`postprocess.py:35-116`](../../worldsim/phase_4/postprocess.py#L35-L116) |
| Phase 4 strategy/placement/variants | Feature-local checkpoints and variant `result.json`/`resume_metadata.json` sidecars | Fingerprint and required-artifact checks gate reuse; each strategy generation/evaluation is persisted incrementally | [`resume.py:402-581`](../../worldsim/phase_4/resume.py#L402-L581), [`strategy_variation.py:130-205`](../../worldsim/phase_4/strategy_variation.py#L130-L205), [`strategy_variation.py:635-857`](../../worldsim/phase_4/strategy_variation.py#L635-L857) |
| Phase 4 eval-awareness iterator | Per-task checkpoint with baseline, iteration records, current task/result, and stop reason; writes a `started` record before external work | Matching source fingerprint restores records; in-flight records without a result remain diagnostic and are not scored | [`eval_awareness_iterator.py:990-1025`](../../worldsim/phase_4/eval_awareness_iterator.py#L990-L1025), [`eval_awareness_iterator.py:1040-1110`](../../worldsim/phase_4/eval_awareness_iterator.py#L1040-L1110) |
| Operator telemetry | `phase_4/progress.json`, schema version 1, atomically written | Observational only; status and remote status consume it, but routing never branches on it | [`postprocess_progress.py:1-20`](../../worldsim/phase_4/postprocess_progress.py#L1-L20), [`postprocess_progress.py:164-203`](../../worldsim/phase_4/postprocess_progress.py#L164-L203) |

The separation is valuable: a cheap pipeline checkpoint does not pretend to be
a complete browser trajectory, and a feature checkpoint can be validated by the
feature that owns its semantics. The cost is that an operator must reconstruct
one effective run from several files and vocabularies.

### Atomicity and failure boundary

`write_json_atomic()` creates a temporary file in the destination directory,
preserves mode bits, optionally trips failpoints before/after `os.replace`, and
cleans the temporary file on any exception. The state layer deliberately writes
the default discovery pointer before the authoritative custom state file, so a
crash between those writes can still recover a full snapshot when the target
directory is present. Tests cover authoritative-vs-pointer precedence, missing
custom state, corrupt candidates, and explicit-directory rejection.
[`worldsim/atomic_io.py:23-76`](../../worldsim/atomic_io.py#L23-L76)
[`worldsim/state.py:67-89`](../../worldsim/state.py#L67-L89)
[`tests/test_state.py:42-378`](../../tests/test_state.py#L42-L378)

The helper has no explicit flush/sync protocol. If the product needs durability
across sudden power loss rather than recoverability after a process crash, that
must be decided and tested separately; do not infer it from `os.replace` alone.

## 2. Intentional pause is not a current state

The implementation has no pipeline `pause`/`paused` status or operator pause
request. `_dispatch_resume()` recognizes only `complete`, `partial_complete`,
`running`, and `failed`; `running` prints “likely crashed” and reruns the phase,
while an unknown status is rejected. [`worldsim/cli/_impl.py:1542-1559`](../../worldsim/cli/_impl.py#L1542-L1559)

Remote stopping is process termination, not a checkpoint transition:
`remote_job_stop.sh` records `stop.json`, sends `SIGTERM`, optionally sends
`SIGKILL`, and records an exit status in the remote registry. It does not write a
pipeline `paused` state or establish a safe boundary for browser/API work.
[`scripts/remote_job_stop.sh:112-146`](../../scripts/remote_job_stop.sh#L112-L146)

The host lifecycle “sweep in progress” tag is a different guard. It prevents
auto-stop/park layers from interrupting an active sweep and is cleared after the
run/archive; it is not a resumable pipeline lease or an operator intent record.
[`docs/infra/r8a-control-plane.md:128-148`](../infra/r8a-control-plane.md#L128-L148)

There are useful partial ingredients: the eval-awareness iterator writes a
`started` record before its next external operation, the Phase 4 run lock
rejects concurrent Phase 4 runs, and bounded async shutdown tests make process
exit finite. None records whether a `running`
checkpoint was a crash, a deliberate stop, a host failure, or an operator who
needs to resume on the same topology. [`tests/test_state.py:1078-1085`](../../tests/test_state.py#L1078-L1085)
[`tests/test_phase_4_shutdown.py:14-51`](../../tests/test_phase_4_shutdown.py#L14-L51)

This distinction matters for evidence: a killed browser session may have a
history file, a partial network trace, and no valid result sentinel. Reusing it
must continue to follow the existing fingerprint/sidecar rules; calling that
state “paused” would imply an operator-safe resume contract that does not yet
exist.

## 3. Provenance, drift, and locality

### Strong existing checks

- Stable fingerprints are SHA-256 over canonical JSON. Instance identity includes
  site/replica, URLs, reset endpoint, PVPO URL, DB connection, placeholders, and
  auth fields; persisted result data carries the fingerprint key rather than raw
  identity in the sentinel. [`worldsim/resume_metadata.py:9-36`](../../worldsim/resume_metadata.py#L9-L36)
- Phase 4 evaluation context includes a resume version, reachable instances and
  placeholders, model/runner/provider/timeouts, sandbox model, and benchmark
  root. Task-specific contexts intentionally project to sites and placeholders
  reachable from that task, so an unrelated site change does not invalidate
  every task. [`worldsim/phase_4/resume.py:130-224`](../../worldsim/phase_4/resume.py#L130-L224)
- Initial results, processed results, placement iterations, strategy checkpoints,
  and variant results all reject mismatched fingerprints; richer AgentLab
  outcomes require their declared sidecars. [`worldsim/phase_4/resume.py:423-581`](../../worldsim/phase_4/resume.py#L423-L581)
- Phase 2 stores model, action-policy, resolution, exposure-contract, text-fill,
  and feasibility settings, then validates identifiers/contracts before reuse.
  Feasibility reuse is additionally bounded by fingerprint and TTL, with an
  explicit force-reverify escape hatch. [`worldsim/phase_2/runner.py:117-128`](../../worldsim/phase_2/runner.py#L117-L128)
  [`docs/warp-taskgen-technical-spec.md:363-369`](../warp-taskgen-technical-spec.md#phase-2c-feasibility-verification)
- Host-locality is documented as an execution contract: Phase 0c Modal probes,
  host inventory, Phase 2c render checks, and Phase 4 agent/PVPO use different
  instance views. Generated topology is host-local and a topology mismatch is an
  infrastructure symptom, not a task/strategy verdict. [`agent_docs/remote-runs.md:62-83`](../../agent_docs/remote-runs.md#L62-L83)

### Gaps to preserve as questions, not assumptions

1. There is no one pipeline-level configuration/provenance digest covering all
   phases. Phase 4 can explain a task-level fingerprint mismatch, but a status
   reader cannot currently answer “which state candidate won?” or “which
   benchmark/config/topology change made Phase 2 or Phase 3 unsafe to reuse?”
2. `instances_path` is saved as metadata, while the strongest Phase 4 checks use
   selected content from the loaded instances. A copied path, regenerated host
   topology, or path with the same name can therefore be an operator-level
   identity question even when task-level drift is correctly detected.
3. Fingerprints intentionally include auth-affecting context. Any new status or
   run envelope must expose hashes, labels, and validation outcomes—not raw
   credentials, headers, cookies, or private connection strings. State metadata is
   an open dictionary, so a blanket “print all metadata” status feature would be
   unsafe without a redaction contract.
4. The old `.aer_inflight` sentinel is swept but never consumed for routing; the
   current source of truth is the processed-result fingerprint. Keeping dead
   markers in the same namespace makes checkpoint discovery harder to explain.
   [`worldsim/phase_4/resume.py:11-38`](../../worldsim/phase_4/resume.py#L11-L38)

## 4. Status explainability

`build_status_payload()` resolves a run root, loads pipeline state and Phase 4
progress/results, and optionally adds cost, artifact manifest, reachability,
intermediate ASR, Phase 2c grouped admission/rejection reasons, and task-bank
summary. The text formatter adds pipeline step/status/timestamp, task root,
Phase 4 counts, variant counters, and artifact paths. The existing status tests
pin those operator-facing facts. [`worldsim/cli_status.py:167-238`](../../worldsim/cli_status.py#L167-L238)
[`worldsim/cli_status.py:330-365`](../../worldsim/cli_status.py#L330-L365)
[`tests/test_cli_status.py:196-260`](../../tests/test_cli_status.py#L196-L260)

What it cannot currently explain without reading logs/source:

- whether the authoritative state file or pointer mirror was selected;
- why a `result.json`, processed result, variant result, or Phase 2 artifact was
  accepted or rejected for reuse (fingerprint, sidecar, schema, age, or missing
  file);
- whether `running` reflects a crash, remote stop, host loss, or intentional
  pause;
- an effective run identifier independent of a path, the owner/lease, host
  locality, or checkpoint schema version;
- freshness/age of each checkpoint and whether a progress heartbeat is stale.

`progress.json` should remain observational. It reports active/completed task and
variant counters under an asyncio lock and explicitly states that routing must
not branch on heartbeat state. [`worldsim/phase_4/postprocess_progress.py:1-20`](../../worldsim/phase_4/postprocess_progress.py#L1-L20)

## 5. Locality, modularity, and semantic naming

The package guide says to organize by behavior owner, keep compatibility facades
thin for one migration cycle, and avoid generic `utils.py`/global types. Phase 4
already follows that direction: runner orchestration, resume/fingerprint logic,
postprocess, strategy variation, iterator, result aggregation, and progress are
separate feature modules. The progress module documents that it was extracted
from closures specifically to restore importable, explicit helpers. [`agent_docs/code-organization.md:5-33`](../../agent_docs/code-organization.md#L5-L33)
[`agent_docs/code-organization.md:116-138`](../../agent_docs/code-organization.md#L116-L138)
[`worldsim/phase_4/postprocess_progress.py:15-20`](../../worldsim/phase_4/postprocess_progress.py#L15-L20)

That locality should constrain a resumability refactor:

- `state.py` should own only pipeline checkpoint discovery/normalization and
  compatibility aliases;
- Phase 2 should own plan/task/feasibility reuse decisions;
- Phase 4 should own task and feature checkpoint semantics;
- progress/status should explain those decisions without becoming a second
  routing engine;
- each feature checkpoint should retain its own validator/version, while a
  small common envelope may carry run context and non-secret provenance.

Current names expose several semantic layers, but they are easy to conflate:

| Name | Actual meaning | Naming question |
| --- | --- | --- |
| `pipeline_state.json` / `step` / `status` | phase routing checkpoint | Is this a *phase checkpoint* rather than a general run state? |
| `progress.json` | observational heartbeat | Keep “progress” distinct from resumable checkpoint? |
| `result.json` | valid per-task completion sentinel | Call it a completion record in operator output? |
| `processed_result.json` | postprocess output/checkpoint | Distinguish “processed” from final scored result? |
| `strategy_variation_checkpoint.json`, placement checkpoint, iterator checkpoint | feature-local resumable state | Should each expose a common checkpoint kind/version? |
| `_resume_fingerprint` vs `_source_fingerprint` | result execution context vs feature source context | Are both “provenance fingerprints,” with a typed scope? |
| `.aer_inflight` | legacy marker that is swept, not consumed | Remove/quarantine dead sentinel names in a later cleanup? |
| `WARP_TASKGEN_STATE_DIR` / `WORLDSIM_STATE_DIR` and `worldsim.*` paths | canonical WARP names plus compatibility surface | Which names are public and which are historical aliases? |

These are semantic questions for review, not a request to rename compatibility
surfaces in this slice.

## 6. Test surface and uncovered contracts

| Contract | Existing evidence | What is still missing for resumability ergonomics |
| --- | --- | --- |
| State pointer, custom directory, corruption, missing keys, and status dispatch | [`tests/test_state.py:15-378`](../../tests/test_state.py#L15-L378), [`tests/test_state.py:381-875`](../../tests/test_state.py#L381-L875) | A fixture asserting an operator-readable candidate/reason/age explanation |
| Atomic crash/restart for Phase 0c, 2, 3, 4, and variant metadata | [`tests/test_crash_resume_phase_0c.py:9-35`](../../tests/test_crash_resume_phase_0c.py#L9-L35), [`tests/test_crash_resume_phase_2.py:9-40`](../../tests/test_crash_resume_phase_2.py#L9-L40), [`tests/test_crash_resume_phase_3.py:9-40`](../../tests/test_crash_resume_phase_3.py#L9-L40), [`tests/test_crash_resume_phase_4.py:9-46`](../../tests/test_crash_resume_phase_4.py#L9-L46), [`tests/test_crash_resume_phase_4_variant.py:9-40`](../../tests/test_crash_resume_phase_4_variant.py#L9-L40) | Power-loss/fsync semantics, if required, are not represented by these failpoints |
| Corrupt/stale feature artifacts and resume drift | [`tests/test_crash_resume_corrupted_artifacts.py`](../../tests/test_crash_resume_corrupted_artifacts.py), [`tests/test_crash_resume_resume_drift.py`](../../tests/test_crash_resume_resume_drift.py), [`tests/phase_4/test_resume_1.py`](../../tests/phase_4/test_resume_1.py), [`tests/phase_4/test_resume_2.py`](../../tests/phase_4/test_resume_2.py), [`tests/phase_4/test_resume_3.py`](../../tests/phase_4/test_resume_3.py) | No common status explanation for why a particular artifact was rejected |
| Process-pool partial output, checkpoint salvage, and fail-closed canonical results | [`tests/phase_4/test_process_pool.py:160-465`](../../tests/phase_4/test_process_pool.py#L160-L465) | No operator-pause or resume-after-TERM contract; salvage must not silently become paper-eligible |
| Concurrent Phase 4 run protection | [`tests/test_state.py:1078-1085`](../../tests/test_state.py#L1078-L1085) | No cross-phase lease/ownership, stale-lock recovery, or concurrent `status`/resume race test |
| Status text and provenance card | [`tests/test_cli_status.py:196-260`](../../tests/test_cli_status.py#L196-L260) | No run identity, pause intent, checkpoint source, freshness, or reuse-reason fields |
| Bounded async shutdown | [`tests/test_phase_4_shutdown.py:14-51`](../../tests/test_phase_4_shutdown.py#L14-L51) | Shutdown is not quiescence: no test proves all in-flight work has reached a safe checkpoint before stop |

The absence of pause tests is a meaningful finding, not evidence that process
termination is safe to call pause.

## 7. Smallest safe vertical slice (candidate for review)

### Candidate: additive, read-only checkpoint explanation

First add a status-facing explanation derived from existing artifacts; do not
change routing, result acceptance, or worker behavior. The view can report:

- the canonical effective run root (`logs_dir`) and existing Phase 4
  `task_dir_root`, without inventing a random identifier;
- the selected checkpoint source (authoritative file versus pointer mirror),
  latest timestamp/age, phase/status, and the exact existing resume action
  (`advance_phase`, `rerun_phase`, or `none`);
- whether the state is legacy (fields absent) and which feature checkpoint
  families are present; and
- non-secret validation labels such as “fingerprint matched,” “stale,”
  “missing sidecar,” “malformed,” or “not inspected.”

This is deliberately an explanation, not a new source of truth. It can be
implemented as a read-only projection first; any additive persisted envelope
should be optional, backward-compatible, and deterministic for the same state
root. A generated UUID in the first slice would make clean-vs-crash harness
comparisons and copied-run semantics harder, so whether an explicit id is needed
must be settled before writing one.

### Preserved invariants

The slice must leave all of these unchanged:

1. Atomic temp-file replacement, pointer fallback, explicit state-directory
   filtering, corruption rejection, and legacy environment aliases.
2. Existing pipeline routing: complete/partial-complete advances, running/failed
   reruns, unknown status fails closed.
3. Phase 4 completion semantics: only valid canonical result sentinels with
   matching fingerprints and required sidecars are reused; history-only or stale
   artifacts rerun; process-pool partial outputs remain inspection-only.
4. Phase 2 admission, exposure, feasibility fingerprint/TTL, task identity, and
   contract validation behavior.
5. Deterministic worker assignment and the Phase 4 run lock; no second run may
   reset a shared benchmark stack.
6. Progress remains observational and status remains read-only; no new status
   field may expose auth, cookies, proxy tokens, DB connection strings, or raw
   private metadata.
7. WARP naming and historical `worldsim` import/environment compatibility remain
   intact.

### Migration order

1. **Contract inventory (docs/tests only).** Name the effective run, checkpoint,
   completion sentinel, progress heartbeat, provenance fingerprint, and lease
   separately. Decide whether path-derived identity is sufficient for archives
   and copied runs.
2. **Read-only projection.** Add status JSON/text fixtures for authoritative vs
   mirror selection, custom state directories, legacy fields, stale/corrupt
   artifacts, and the existing complete/running/failed branches. Do not alter
   `save_state()` or resume dispatch yet.
3. **Optional additive envelope.** If the review establishes a need for persisted
   run context, add fields with defaults when absent and preserve unknown fields;
   make the envelope non-secret and deterministic. Keep old paths and aliases as
   compatibility facades for one migration cycle.
4. **One feature at a time.** The first behavioral checkpoint migration should
   be a single Phase 4 feature (iterator or strategy variation), because it has
   the richest existing fingerprints and crash tests. Report reuse/rejection
   reasons through the projection while retaining the feature-local validator.
5. **Only after pause semantics are agreed:** design an operator request,
   quiescence boundary, lease/owner, and explicit paused outcome. Start with a
   phase-boundary stop if that is the accepted contract; do not claim an
   in-flight browser/API task is paused until its checkpoint and benchmark reset
   semantics are proven. Migrate other phases in separate focused changes.

### Acceptance evidence

For the read-only slice, the minimum evidence should be:

- `uv run pytest tests/test_state.py tests/test_cli_status.py -q` with legacy,
  pointer, custom-directory, and malformed-artifact fixtures;
- `uv run pytest -m crash_resume tests/test_crash_resume_*.py -q` with clean and
  failpoint runs retaining equivalent outputs/state after normalization;
- `uv run pytest tests/phase_4/test_resume_1.py tests/phase_4/test_resume_2.py tests/phase_4/test_resume_3.py tests/phase_4/test_process_pool.py -q`;
- `bash scripts/verify_fast.sh` (and the package acceptance command
  `bash scripts/accept_taskgen.sh` when the change is promoted); and
- a status JSON/text comparison demonstrating that explanation is advisory and
  does not alter admission, exposure, visibility, scoring, readback, or safety
  outcomes. This is a docs/code-local slice, so a live stack is not required;
  live gates become necessary only when execution or host behavior changes.

### Risks to surface before implementation

- A canonical path is not an identity if an operator reuses a state directory;
  an explicit id can solve that but may break deterministic crash comparisons or
  copied-run interpretation.
- Pointer and authoritative writes have equal logical timestamps in some paths;
  adding a new field must not accidentally change candidate ordering.
- Status currently includes the full pipeline state object in JSON. Reuse-reason
  reporting must redact open-ended metadata and avoid turning diagnostics into a
  secret export.
- A “matched” label is meaningful only if it names the feature, fingerprint
  scope/version, and required sidecars checked; a generic green check would be
  misleading.
- A `paused` status without worker quiescence, benchmark reset policy, and
  topology binding would make a stopped run look safer than it is.
- New schema fields must be tolerant of old mirrors, archives, and feature
  checkpoints; strict migration would make the existing recovery path less safe.
- Atomic replacement protects against truncated JSON but not necessarily sudden
  power loss; a stronger durability claim would require an explicit storage and
  test contract.

## 8. Facts and questions to grill before choosing a pause design

Facts established by the current sources:

- `running` means rerun the phase, not “wait for an operator.”
- Remote stop sends process signals and records job-registry state, not pipeline
  pause intent.
- Phase 4 checkpoints can safely reuse only artifacts that pass feature-local
  fingerprint and sidecar checks.
- Host topology and instance files are execution-locality contracts; a fresh
  host/setup failure gets a fresh state directory in the operator guide.
- Progress is observational and cannot become a routing oracle.

Questions requiring an explicit answer:

1. Does “pause” mean stop admitting new work, stop at the next phase boundary,
   stop after each task, or suspend an in-flight browser/API operation?
2. May a paused run resume on another host or regenerated topology? If not, what
   exact locality/provenance binds it, and where is that evidence stored?
3. Does an operator stop preserve the same evidentiary run, or must a stop create
   a new state directory after preconditions are repaired?
4. Who owns the lease (local CLI, remote-job registry, or both), how is stale
   ownership recovered, and how is a second operator prevented from resuming?
5. What result/sidecar set is sufficient when a task is interrupted during seed,
   browser evaluation, PVPO, judge, or cleanup? Which partial traces are
   diagnostic only?
6. Should resume after `SIGTERM` require an explicit operator acknowledgement,
   or should it be indistinguishable from a crash after the current checks pass?
7. What non-secret provenance must status expose, and which fields must remain
   hash-only or entirely private?
8. Is process-crash recovery the durability target, or must the storage contract
   cover power loss and remote filesystem failure as well?
9. Which term is canonical for each scope: run identity, pipeline checkpoint,
   feature checkpoint, completion sentinel, progress heartbeat, source
   fingerprint, and lease?
10. What evidence would prove that introducing the new state does not change
    admission, exposure, visibility, scoring, readback, safety, or the paper
    eligibility of a repaired process-pool run?
