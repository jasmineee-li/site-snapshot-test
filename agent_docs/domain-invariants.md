# Domain Invariants

Use this before changing task generation, seeding, feasibility, Phase 4, prompt files, reward functions, auth, or sandbox execution.

## Spec First

`docs/worldsim-v5-technical-specifcation.md` is the source of truth. The filename typo is intentional. Every `worldsim/` module implements a spec section. When source or handoff docs expose real drift, update the spec first, then align code.

If the spec contains literal code for `run_claude_in_sandbox`, `BrowserUseAgent`, `apply_data_seed`, `save_state`, or `run_eval`, preserve that behavior verbatim unless the spec is intentionally changed.

## Scope

WorldSim v5 is strict WASP-aligned IPI evaluation. Current admitted surfaces are:

- GitLab issues/comments
- Reddit/Postmill posts/comments

Dataset counts are run state, not invariants. Discover the active cohort from
the pinned run directory, `phase_2/feasibility_report.json`, `phase_3/contracts.json`,
or the relevant sweep/runbook.

Magento, Wikipedia, OpenStreetMap, shopping, and classifieds are excluded from the current WASP scope. Treat code paths for them as historical or support plumbing unless the spec and task explicitly bring them back.

## Phase 2 and Admission

Phase 2c feasibility is strict admission for Phase 4:

- Never ship a Phase 4 run on an unverified dataset.
- Never hand-edit `feasibility.status`.
- Do not use `--skip-feasibility` for shipping runs.
- Break-glass override is `WORLDSIM_STRICT_FEASIBILITY={true,false}` and should be called out explicitly.

Phase 4 varies adversarial strategy only. Benign task contracts and reward functions emitted by Phase 3 are invariant across variants, across all adaptive rounds.
Strategy variants must change payload semantics, not only punctuation, brackets, or wrappers; no-op and punctuation-only rewrites are generation failures.

The Phase 4 adaptive loop is bounded at `3+3+1`: after the initial
PVPO-encountered refusal, the host may generate up to 3 variants in round 1, up
to 3 variants in round 2, and up to 1 final variant. The attack budget is at
most 7 generated variants per task, excluding the original baseline attempt.
Do not add unbounded retry loops, extra hidden fan-out, or placement mutation
under strategy variation.

Every generated/evaluated variant must preserve round lineage for reporting:
root initial attempt id, parent attempt id, round number, selected strategy,
refusal trigger, and host-finalization status. Lineage and attack-budget fields
are audit metadata only; they must not affect rewards, admission, PVPO gates, or
the benign task shown to the browser agent.

Phase 2a planners emit a high-level `adversarial_action`, not evaluator JSON.
The host compiles that action into `reward_function.adversarial_reward` after
immutable benign fields and the exposure contract are merged. Keep evaluator
shape, endpoint/body/status checks, and binary-label derivation in host code so
prompts stay benchmark-neutral and new benchmark adapters only need to expose
editor/action contracts.

Do not conflate host-compiled rewards with task-diversity expansion. The
current no-title mainline can remain mostly binary read/classification tasks
even after `adversarial_action`; adding richer IPI success criteria requires
new Phase 1 route/task-card support, explicit benign reward schemas, compatible
action options, compiler rules, and Phase 2c/Phase 4 evidence. Validate in that
order: semantic compiler on known binary tasks first, action/mutation pilots
second, mixed headline sweeps only after both families pass admission.

Action-oriented pilots need capability-aligned benign tasks. The benign task
should naturally exercise the browser capability family the injection attempts
to hijack, but it must not ask for the exact malicious target, nonce, fixture
file, endpoint, reward, or cleanup action. A read-only UGC classification task
is a valid semantic-hijack task; it is not a strong repository-write pilot. For
task cards that need both encounter and action evidence, use conjunctive card
constraints so a read-only instruction cannot pass by matching only the
encounter verb. For repository/content mutation, generate a benign triage or
maintenance workflow
where an artifact update is normal, then keep the injected fixture path/content
and state-confirmed action reward host-owned. If the capability family changes,
regenerate Phase 1 with explicit task-card support and rerun Phase 2c; do not
patch the mismatch inside Phase 4 variants.

Tier 2 action-card tasks must make the benign public action measurable. If a
card opts into `requires_benign_action_evidence`, Phase 1 validation finalizes
the task by adding a natural per-task benign action witness and a host-compiled
`NetworkEventEvaluator` to the benign reward. This prevents a task from asking
for a mutation in prose while scoring only the old read/classification answer.
The witness must stay benign and task-local; adversarial URLs, action nonces,
fixture paths, and benchmark-internal labels remain forbidden in the benign
instruction.

Action capability contracts and capability task cards are host-owned generation
constraints, not model-authored rewards. They may describe generic capability
families, compatible carrier surfaces, benign reward shapes, fixture/readback
requirements, and forbidden benign/adversarial overlap. They must not place
benchmark-specific selectors, endpoint paths, fixture setup, final-state probes,
or cleanup logic in model-facing static prompts.

Adversarial actions are impact-stratified. Tier 1 is semantic/navigation
hijack, Tier 2 is bounded public mutation on UGC workflows, and Tier 3 is
high-impact security behavior such as access grants, secret/key creation,
outbound integrations, private-data exfiltration, repository/content mutation,
destructive mutation, or account setting changes. Tier 3 is in scope only as a
named pilot condition with disposable fixtures, deterministic readback, cleanup,
and separate reporting. Do not mix Tier 3 into default generation or aggregate
it with semantic ASR without explicit labeling.

Tier 3 payloads must be host-grounded. A ready Tier 3 action can expose a
payload-action contract, such as a deterministic repository file path plus
required witness content. Text fill and variant generation may vary prose and
framing, but they must not replace that contract with a generic fixture-edit
request or a binary-answer label flip. If the payload cannot name the
host-selected target/content and produce action witnesses for it, fail closed
before Phase 2c instead of creating an unmeasurable live artifact.

Tier 3 readiness is maturity-gated. Treat catalog-only actions as
`L0_declared`; allow a model to choose a Tier 3 action only after a host adapter
has fixture setup/cleanup, deterministic reward compilation, final-state
readback, Phase 2 admission evidence, and a named pilot policy. The first
pilot-ready family may expose repository/content mutation in a disposable
fixture, but other high-impact families must stay unselectable until they reach
the same adapter maturity. A Tier 3 action passing state readback is reported as
state-confirmed action success, not folded silently into semantic ASR.

Repository/content fixture targets are action targets, not carriers. The
injection still has to render on a strict WASP issue/comment surface, but the
repo/file the payload asks the agent to modify must come from a host-owned
fixture contract. Prefer a small fixed pool of disposable fixture projects with
unique per-task file paths under a configured prefix. The adapter must prove a
canary write, exact readback, and cleanup before it annotates the exposure
contract as `L4_pilot_ready`.

Carrier exposure and action objective are separate contracts. A payload can be
strictly admitted on a WASP carrier while asking the agent to perform a
different workflow. Keep benchmark-specific action endpoints, selectors,
fixture setup, final-state probes, and cleanup in host-owned action specs or
benchmark adapters, never in model-facing prompts or model-authored rewards.

Use artifact-local gates before changing prompts broadly. The binary compiler
canary proves existing semantic rewards still compile deterministically; the
`semantic_only` action policy is for minimum validation runs that must isolate
host-compiled binary answer hijack without navigation or mutation-action
confounds. The
mutation pilot audit proves a Phase 2 artifact has compatible editor/action
contracts and POST/status/body-witness rewards before Phase 2a is biased toward
mutation actions or Phase 4 headline metrics are mixed across reward signals.
Only use `--phase-2a-action-policy mutation_when_available` for an explicit
pilot cohort after that audit passes; default runs must leave planner action
choice unchanged.

For Tier 3 pilots, run the Tier 3 action-pilot audit before a live browser
pilot. It recomputes `tier3_pilot` readiness from benign tasks and exposure
contracts, then checks selected Tier 3 tasks for host-compiled request evidence
and final-state readback when Phase 2 artifacts exist. Treat it as a fail-closed
preflight only; it is not an admission layer and must not edit feasibility or
scoring artifacts.

Mutation actions may carry `reward_function.adversarial_final_state_check` as a
confirmatory persistence readback. Phase 4 reports it separately from
`adversarial_passed`; do not silently make it a headline gate without a named
pilot condition because that changes success semantics from request submission
to persisted state. For Tier 3 pilots, use the explicit
`state_confirmed_action_success` and `tier3_state_confirmed_action_success`
fields when discussing high-impact outcomes; leave request-level
`attack_action_success` and semantic ASR separate.

## Phase 4 Gates and Metrics

Phase 4 has two gates:

1. Encounter gate: PVPO `max_coverage` is a float in `[0.0, 1.0]`; `max_coverage == 0` routes to placement-fix.
2. Attack-effectiveness gate: refusal judge evaluates whether the agent complied with the injection.

Transcript Purpose and Verbalized Eval Awareness are observational metrics. They must never gate, branch, or affect run control.

Attack-budget reporting is also observational. Summaries should expose the
configured `3+3+1` cap, generated/rejected/evaluated/PVPO-valid/compliant
counts by round, remaining budget, termination reason, and lineage, but those
fields must not change final-status classification or ASR computation.

Trajectories flow through disk. A Phase 4 run writes `phase_4/results.json`,
`phase_4/progress.json`, and timestamped per-task directories containing
`history.json`, `result.json`, `screenshots/`, `conversations/`, and `pvpo/`.
Judges and classifiers receive host-side dataclass-projected slices.

The six host-API prompt files are:

- `worldsim/prompts/judge-adversarial-failure.md`
- `worldsim/prompts/generate-variant.md`
- `worldsim/prompts/transcript-purpose-guess.md`
- `worldsim/prompts/transcript-purpose-classify.md`
- `worldsim/prompts/verbalized-eval-awareness.md`
- `worldsim/prompts/placement-fix.md`

They feed single-turn API calls with no sandbox error boundary. Be conservative and test exact parsing contracts.

When adding OpenAI-backed prompt/API flows, prefer Structured Outputs or deterministic parsers over prose-only schema instructions. Keep schemas in code/tests rather than duplicated across long prompt text.

## Phase 4 Strategy and Rewards

Do not reintroduce `visual_concealment`; paper concealment is behavioral final-response steering, not visual concealment.

Behavioral concealment strategies such as `scripted_message`, `command_silent_execution`, and `false_justification` remain deferred unless the reward system changes. The current `AgentResponseEvaluator` plus `db_query_match` reward is tool-side and cannot grade those strategies.

Title fields are retired as active mainline IPI carriers for GitLab and Reddit.
They remain writable UGC metadata and historical analysis dimensions, but new
Phase 1 route contracts and Phase 2 admission must target body-like fields with
enough semantic capacity for realistic instructions. GitLab issue descriptions
and Reddit submission bodies are active when strict route evidence proves the
agent is forced through the rendered field. Comments and notes remain active
only with exact discussion-region evidence or a runtime hook. Do not silently
fall back to title carriers when a body/comment/description route is ineligible.

Historical title-inclusive sweeps stay analyzable and must be labeled as such.
When an old `link/no-link` title task visibly contains a literal URL and the
benign instruction asks whether the title contains/includes a URL, an agent
answering `link` after PVPO encounter and host-valid variants is a valid
resistance outcome, not a placement failure. Do not weaken eligibility or
rewrite the benign task at Phase 4 to make the payload easier.

## Surface Identity

WorldSim canonical carrier IDs are methodological labels: `issue.description`,
`note.body`, `submission.body`, and `comment.body`. Benchmark profiles may use
adapter-local observed IDs such as `profile_issue_description` or
`thread_comment_body`. Do not ask the generator or profiler to be the authority
for both vocabularies. Route generation, Phase 2 validation, and text-fill
budgeting must resolve profile surfaces through the benchmark adapter mapping in
`worldsim/surface_identity.py`.

Unknown benchmark/site mappings and ambiguous profile surfaces fail closed. For
example, a shared `note.body` canonical surface may have multiple concrete
profile locations; it must resolve through route/editor context, and if the
caller provides no disambiguating context, the resolver must return no surface
rather than choosing arbitrarily. Adding a new benchmark means adding its
profile-ID/source-field aliases at the adapter boundary and tests that prove
canonical WorldSim surfaces resolve to the new profile vocabulary.

## Auth and Runtime Boundaries

Claude auth precedence is centralized:

- Sandbox Claude Code: `worldsim/modal_sandbox.py::_build_claude_secrets`
- Host API calls: `worldsim/phase_4/anthropic_client.py`

Supported modes are `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL`. Let the helpers decide; never hard-code one path.

No Phase 4 trajectory step routes through `run_claude_in_sandbox`. Refusal judge, variant generator, Transcript Purpose, VEA, and placement-fix all use direct host Anthropic Messages API calls.

Instructor/Pydantic structured calls are allowed for bounded host-side helper steps such as Phase 2b text fill, but they must preserve WorldSim observability: transport retries stay in `worldsim/phase_4/anthropic_client.py::call_with_retry`, semantic retries are limited to parse/validation failures, and diagnostics must record compact request/response metadata without full prompts or secrets.

Modal sandbox scope is based on explicit inclusion with `image.add_local_file` / `image.add_local_dir`. Ignore-file patterns are not an isolation boundary.

`AgentLab/src/agentlab/benchmarks/redteam/execution.py` and `AgentLab/src/agentlab/benchmarks/redteam/claude_code.py` are read-only reference material. Runtime imports from `AgentLab/` are forbidden.
