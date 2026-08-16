# Admission and Exposure Contracts

Use this before changing Phase 1 generation, Phase 2 seeding/text fill,
exposure contracts, carrier selection, Phase 2c feasibility, or Phase 3
admission inputs.

For the experimental Classifieds canary, writer/reader/reset proof, or
`classifieds_listing_reply_poc`, read
[the canary procedure](./classifieds-canary.md) before acting.

## Source, scope, and admission

`docs/warp-taskgen-technical-spec.md` is authoritative. Phase 2c feasibility is
strict admission for Phase 4: shipping runs require verified feasibility,
positive reachability, an eligible `exposure_contract`, and exact contract
matching. `feasibility.status` is pipeline output; regenerate the phase rather
than editing it. The development break-glass
`WORLDSIM_STRICT_FEASIBILITY={true,false}` must be explicit in run evidence.

The default WASP scope is GitLab issues/comments and Reddit/Postmill
posts/comments. The only Classifieds path is the issue #149 experimental
composition `classifieds_listing_reply_poc`. It is an explicit per-run
dependency bundle, not a default registration. It admits only VisualWebArena
`classifieds` / `listing_reply.body` and keeps the GitLab/Reddit policies
unchanged. Other full-benchmark surfaces remain historical unless the
technical spec explicitly changes scope. Counts belong to Run manifests, not
this document.

The Classifieds canary must write through a regular user, then read through a
fresh anonymous session. The reader must prove the exact listing, reply ID,
actor, normalized body, same-origin route, and painted visibility. An external
golden-state reset must prove the reply absent before and after the bounded
Run. Missing reader, image-provenance, reset, or exact-resource evidence fails
closed. The path never falls back to a writer session, root/index page,
newest-item selection, SQL/admin readback, or a weaker carrier.

## Phase 2 ownership

Phase 2a planners emit a high-level `adversarial_action`, not evaluator JSON.
Host code compiles that action into `reward_function.adversarial_reward` after
immutable benign fields and the exposure contract are merged. Keep endpoint,
body, status, binary-label, and final-state semantics in host code so prompts
stay benchmark-neutral.

Phase 2a planning may pause only between shard claims. A pause request stops
new claims, lets admitted target-resolution/API/validation work finish, and
writes each completed shard plus its Run-bound manifest atomically before the
runner records `paused`. Resume accepts a shard only when its Run ID,
Definition Digest, payload hash, and existing Phase 2 validators match; other
shards rerun. No partial plan merge is promoted. Phase 2b text fill has a
separate task boundary: each plan/task writes one Run-bound checkpoint after
all payload ordinals, validation, selected seed, and diagnostics complete.
Pause drains admitted text-fill units without promoting a partial
`adversarial_tasks.json`; exact resume reuses only matching Run ID/digest,
plan-input, model, and text-count evidence. Phase 2c feasibility has its own
bounded source-data preflight followed by a task-level Atomic Work Unit:
seed, render/readback/reachability, cleanup, and the Run-bound checkpoint.
Pause requests received during preflight are acknowledged only after that
setup drains and prevent every later verification claim. During verification,
claims are serialized with the request marker; admitted units finish and
checkpoint, then the Run is persisted as paused without promoting a partial
aggregate. Exact resume delegates compatibility to the Phase 2c checkpoint
validator, rerunning missing, stale, malformed, legacy, tampered, or
topology-drifted tasks.

Carrier exposure and action objective are separate contracts. A payload may be
admitted on a WASP carrier while asking the agent to perform a different
workflow; benchmark-specific endpoints, selectors, fixture setup, readback,
and cleanup stay in host-owned adapters and action specs.

Phase 1 scenario and precondition fields describe only generic benign workflow
roles (`task_local_prerequisite`, `maintenance_prerequisite`, or
`public_followup_instruction`). Phase 1 validation overwrites or strips
model-authored targets, selectors, nonces, fixture paths, reward logic, and
cleanup behavior. Phase 2 owns concrete adversarial action/evidence; Phase 4
preserves the benign task contract.

Named compiled capability profiles are the live-pilot path. They are
route-local, fingerprinted, validated by the Phase 1 task-card gate, and fail
closed when a requested site has no active card. Hand-authored task-card JSON
is for explicitly labeled experiments only.

## Carrier and readback gates

Body-like fields are the active mainline carriers. Title fields remain
historical analysis dimensions and writable metadata; do not silently fall back
to a title when a body/comment route is ineligible. GitLab descriptions and
Reddit submission bodies need strict rendered-route evidence. Notes/comments
need exact discussion-region evidence or a runtime hook that proves the seeded
record was encountered.

For Reddit/Postmill comment carriers, reaching the comments region is not
seed-specific encounter evidence. Admit `comment.body` only with seed-specific
visibility evidence or a runtime hook. Public post carriers have a separate
contract.

Public mutation readback must prove the newly persisted, agent-attributed
action. `submit_comment` probes carry `requires_attribution=true` and seeded
carrier IDs (`excluded_comment_ids` and/or `seed_comment_id`); if IDs are
deferred, `attribution_source` binds them in Phase 2c, otherwise the fallback
fails closed. Parent-page substring scans do not prove an agent-authored
comment. Reddit/Postmill readback uses GET or unknown-method routes and skips
mutation endpoints such as `/-/comment` and `/-/edit`.

Created-child listing routes use deterministic, route-local anchors. Avoid
fuzzy “matching” or newest/latest wording unless the route contract proves
those semantics; route drift is not action evidence.

## Capability-aligned benign work

An action pilot's benign task naturally exercises the capability family being
hijacked without naming the malicious target, nonce, fixture, endpoint, reward,
or cleanup action. Use conjunctive card constraints when both encounter and
action evidence are required. If the capability family changes, regenerate
Phase 1 with a new card and rerun Phase 2c rather than repairing the mismatch in
Phase 4.

Tier 2 cards with `requires_benign_action_evidence` receive a host-compiled,
task-local benign witness and evaluator. Tier 3 repository cards bind concrete
benign file/content evidence only after disposable fixture attachment with
`benign_action_evidence.stage="phase2_tier3_fixture"`. Keep benign evidence
distinct from the adversarial action contract.

Use `action-contracts.md` for tier semantics, reward shapes, fixture maturity,
and variant preservation. Completion means the selected carrier, route,
exposure evidence, and Phase 2c admission conditions are all explicit and
host-owned.
