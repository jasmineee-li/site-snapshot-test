# Agent developer-experience frontier research

- Research baseline: 2026-08-30
- Planning date: 2026-09-04 (America/New_York)
- Source authority inspected: `origin/main` at `d7d9a33e`; planning continued
  in the existing worktree on `docs/ci-agent-devex`
- Decision status: recommendations accepted by the user on 2026-09-04
- Scope: read-only repository inspection and first-party online research; no
  provider, browser, infrastructure, dependency, credential, or benchmark
  execution

This note asks what can shorten the next WARP source-ticket cycle while paid
OpenRouter generation is unavailable. It is deliberately narrower than a new
agent framework. The 2026-09-03 note already covers the provider-independent
cache, lock, cost, remote-root, and Phase 2 status slices and the compact issue
execution contract. This update adds only the operational edges around those
slices: a fresh-context review handoff, actionable source-CI output, clean-state
resumption, and measurable DX counterfactuals. None of these changes may alter
Run Definition or Checkpoint compatibility, safety/exposure/admission gates,
reset isolation, or the authority of runtime evidence.

## Recommendation summary

The current independent decision frontier is three small investments, all
independent of a paid model or live Site:

1. Permit each source-ticket worktree to create its own locked Taskgen
   development environment, then use the existing focused and shipping
   commands from that environment. Do not share an editable package environment
   across concurrent worktrees.
2. Do not add a no-provider command or seven hand-authored provider-response
   fixtures. Representative family pipeline tests already replace only
   transport while using the production compiler and validator. When the exact
   frozen route is restored, use two one-row production micro-canaries to cover
   the actual SDK and sandbox response boundaries from a disposable run root.
3. Measure the two duplicate default-suite readiness repository walks and
   replace them with focused inputs only if they consume material time, always
   preserving both semantic assertion sets. Add concise source-CI job summaries.
   Route the disjoint root gate internally for clearly Taskgen-only changes while
   keeping its required job visible; defer diagnostic artifact uploads, new
   hooks, and worktree automation until measurements show they would repay their
   maintenance cost.

Keep the existing compact issue contract, two-pass writer/reviewer bound, and
five-line clean-state record as process guidance rather than a new schema. The
three recommended investments address observed environment ambiguity, paid
contract failures, and duplicated local work. The hook remains a measured
follow-up, not an immediate source slice.

Only the duplicate-scan removal and CI-routing work are immediate source-work
slices. The two micro-canaries are later runtime validation, not a new source
framework. No broad module split or agent framework is recommended.

## Source-supported facts

### Small, scoped repository maps beat instruction dumps

[Factory's Agent Readiness overview](https://docs.factory.ai/agent-readiness/overview)
places deterministic commands, fast tests, reproducible environments,
documentation, observability, and security controls in the foundation of agent
readiness; its higher maturity levels are diagnostic targets, not correctness
requirements. [Factory's AGENTS.md guidance](https://docs.factory.ai/harness/agents-md)
recommends a short root briefing with exact install/test/lint/build commands,
then nested instructions only where a subtree has genuinely different rules.
It also recommends linking to durable repository documentation rather than
putting a long runbook in the always-loaded file.

[OpenAI's harness-engineering report](https://openai.com/index/harness-engineering/)
similarly describes the root agent file as a map, with versioned repository
documents as the system of record. It warns that a monolithic instruction file
consumes context, rots, and is difficult to verify. This is guidance about
context and maintenance, not evidence that any vendor's readiness score predicts
WARP benchmark outcomes.

[OpenAI's Codex and AGENTS.md guidance](https://openai.com/index/introducing-codex/)
describes `AGENTS.md` as repository-local instructions for navigation, test
commands, and project practices; it also expects a clean worktree and asks the
agent to surface uncertainty and test results for human review. WARP adopts the
useful parts—exact commands, clean-state handoff, and visible uncertainty—while
retaining its own one-worktree/one-PR and evidence rules.

[GitHub's repository-instructions guidance](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions)
and [coding-agent best practices](https://docs.github.com/en/copilot/using-github-copilot/using-copilot-coding-agent-to-work-on-tasks/best-practices-for-using-copilot-to-work-on-tasks)
recommend concise repository and path-specific instructions containing project,
build, test, and validation commands. The useful inference for WARP is to keep
one source of truth and point consumers to it; duplicating a second GitHub-
specific policy tree would create drift.

### Feedback closes faster when the target and evidence are explicit

[Anthropic's Claude Code best-practices guide](https://code.claude.com/docs/en/best-practices)
recommends an executable test/build/lint target, a plan for uncertain
multi-file work, specific scope and acceptance criteria, and a fresh context for
independent review. It also recommends showing command output or other evidence
rather than merely asserting completion. The same guide says to stop and reset
context after repeated correction loops instead of accumulating unrelated
history.

[Anthropic's long-running-agent harness guidance](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
describes incremental work from a known clean state, explicit pass/fail feature
items, descriptive handoffs, and a basic startup check before changing a broken
baseline. The transferable practice is a small human-readable handoff, not a
new progress database: WARP already has git state, Run Definitions, Checkpoints,
and Run Artifacts with stronger semantics.

[Anthropic's context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
emphasizes a finite attention budget, high-signal instructions, self-contained
tools, and adding detail only when a concrete failure requires it. This supports
narrow source tickets and targeted commands rather than an all-purpose agent
prompt.

### Hooks are deterministic automation, not a safety boundary by themselves

[Factory's hook documentation](https://docs.factory.ai/harness/hooks) and
[Anthropic's hooks guide](https://code.claude.com/docs/en/hooks-guide) describe
hooks as lifecycle-triggered commands that can validate or block work. They
require review of local environment/credential exposure, absolute or resolved
paths, timeouts, and safe handling of exit status. Anthropic also documents that
matching hooks can run in parallel and that a denial does not undo side effects
already started by another hook. Therefore a hook that mutates files, starts a
network call, or writes shared state is a poor fit for WARP's safety boundary.

### CI can expose diagnosis without becoming runtime evidence

[GitHub Actions job summaries](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/workflow-commands-for-github-actions)
are visible on the workflow summary. [Artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
persist selected logs, test results, and build outputs after a job ends, while
caches are intended for reusable dependencies and intermediate outputs. The
distinction matters: a source-test artifact can reduce a rerun, but it is not a
Run Artifact and cannot prove a browser exposure, reset, admission, or attack
outcome.

[GitHub's concurrency syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
supports one cancellable source-CI run per workflow/ref. WARP's acceptance and
root-check workflows already use workflow/ref groups with `cancel-in-progress`;
that policy is appropriate for source CI and must not be copied to live or
reset-sensitive jobs.

[GitHub's Python workflow guidance](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
recommends an explicit Python setup and dependency caching. [uv's project
documentation](https://docs.astral.sh/uv/concepts/projects/sync/) specifies that
`uv run --locked`/`uv sync --locked` reject stale lock state while using the
checked-in exact environment. WARP's acceptance script already uses this
boundary, so the DX opportunity is to surface it consistently, not to add an
unlocked installer.

[uv's command reference](https://docs.astral.sh/uv/reference/cli/#uv-run)
also states that `uv run` creates or updates the discovered project environment
by default, while `--no-sync` refuses that update. A shared editable environment
is a poor fit for parallel WARP worktrees because the installed project path can
name a different checkout. The simplest reproducible policy is one locked
environment per source worktree, using uv's ordinary cache for downloaded
artifacts. A no-install policy needs an explicitly configured existing
environment plus a source-path assertion; silently borrowing a neighboring
worktree's editable environment is not equivalent.

## WARP friction observed

These are repository observations, not measured throughput claims:

- The execution plan says each issue is a compact execution contract and names
  the seam, failure, positive/counterfactual test, commands, artifact/status,
  evidence boundary, and escalation condition
  ([plan](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/web-benchmark-onboarding-2026-08-30/execution-plan.md:250)).
  It does not prescribe the exact writer/reviewer handoff or a clean restart
  record after a correction loop.
- `verification.md` distinguishes focused, default, live, and specialized
  checks, preserves full failure output with `run_silent.sh`, and says a
  handoff must name commands, evidence paths, and blockers
  ([verification guide](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/agent_docs/verification.md:6)).
  That guidance is strong, but the source acceptance workflow currently exposes
  only step output; it has no `$GITHUB_STEP_SUMMARY` or failure artifact step.
- `scripts/accept_taskgen.sh` already provides the canonical route-only check,
  locked sync, split lanes, and installed-package proof
  ([acceptance boundary](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/scripts/accept_taskgen.sh:15)).
  Agents can still lose time choosing root versus package cwd or rerunning a
  lane to rediscover the route decision when a CI failure is terse.
- The root `AGENTS.md`, Taskgen `CLAUDE.md`, feature-owned `agent_docs`, and
  code-organization rules already provide scoped routing and ownership. The
  code-organization guide explicitly rejects generic `utils.py`/`helpers.py`,
  broad technical-layer modules, and arbitrary splits
  ([ownership guide](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/agent_docs/code-organization.md:5)).
  A new cross-cutting readiness framework would conflict with this architecture.
- The provider-independent E3 source DAG #240--#246 is merged through
  `d7d9a33e`. Eighty non-TAC rows and all 40 Rocket.Chat rows remain
  ungenerated; the retained Run 15 has 20 reusable Postmill rows and no
  admitted bank. Source-CI and handoff improvements are useful without any
  model, credential, browser, reset, or live evidence, but they must not be
  described as remaining E3 task-generation work.
- Issue #212 records repeated paid Phase 1 failures before admission: Run 06
  exhausted three sandboxes at `$5.74`, Run 08 exhausted three at `$4.59`, Run
  09 retained at least `$3.3204` plus one partial call, and Run 14 reached the
  20-minute no-event watchdog followed by a five-minute drain. PRs #228--#236
  then repaired output-contract parity, selection semantics, batching, and
  diversity. The repeated counterfactual is a representative model-shaped
  output that fails the real host extraction/compiler/validator boundary before
  any provider dispatch.
- The current full readiness audit takes about 21.5 seconds over 5,898 tracked
  files and 1,048 code files in this checkout. `verify_fast.sh` runs that audit,
  while `tests/test_phase_compat_wrappers.py` and
  `tests/test_namespace_compatibility_evidence.py` each invoke `build_audit()`
  again in the default pytest lane. Their two semantic assertion sets remain
  useful, but the two extra repository walks add no distinct safety authority.
- For PRs #248--#254, the median GitHub check window was about 2.52 minutes and
  median PR creation-to-merge was about 3.05 minutes. Source CI is already
  parallel and reasonably short. The stronger immediate target is local
  reproducibility and failure diagnosis, not a new CI orchestration layer.
- The execution plan still presents #240--#246 as pending source slices and its
  operational-input section asks for model, budget, corpus allocation, and TAC
  information that later report sections and #212 already resolve. Parent #209
  also leaves E1/E2 unchecked. This is ordinary plan drift, but it can make a
  fresh agent rediscover inputs or stop on a blocker that no longer exists.

## Proposed practices

The fields below are intentionally repeated for each practice so a future issue
can be implemented without inferring scope. “Inference” is a recommendation
derived from the cited sources and the WARP observations; it is not a measured
result.

### DX-0 — Give each ticket worktree a locked, source-local environment

- **Observed WARP friction:** Per-ticket worktrees are the repository's required
  isolation unit, but `uv run` may create or update a local `.venv`, while a
  borrowed editable environment can import WARP from a different checkout.
  Review agents that cannot use the expected dependencies either skip commands
  or improvise `PYTHONPATH` and interpreter combinations.
- **Exact failure prevented:** A focused check passes against stale or
  neighboring source, or a reviewer cannot reproduce it at all. The PR then
  spends a review pass diagnosing its harness instead of its behavior.
- **Smallest sufficient change (inference):** Allow one
  `uv sync --directory packages/warp-taskgen --extra dev --locked` per ticket
  worktree, with no lockfile mutation and no provider/browser command. Reuse
  uv's cache, not another worktree's editable installation. Make the first
  focused check assert that `warp_taskgen.__file__` resolves under the current
  worktree. If dependency installation remains prohibited, require one named
  pre-existing interpreter and the same source assertion; never discover a
  neighboring environment heuristically.
- **Counterfactual and metric:** Intentionally point the interpreter at another
  worktree and confirm the source assertion fails before pytest. Record time
  from worktree creation to first focused test and environment-related review
  failures across the next five tickets.
- **Dependency/order:** First, because every later source slice and review uses
  it. This is local source-tooling setup, not benchmark infrastructure.
- **Classification:** Development-environment policy; optionally a narrow
  assertion in the existing verification script, not a new environment manager.
- **Evidence boundary:** A locked source environment proves dependency and
  checkout selection only. It does not prove a provider, Site, reset, exposure,
  admission, or outcome.

### DX-1 — Add a fresh-context writer/reviewer handoff

- **Observed WARP friction:** The plan bounds review to two passes, but a PR
  does not have a standard short handoff saying what the writer changed, what
  was actually run, which counterfactual regressed, and what remains unknown.
  A reviewer can repeat repository exploration or chase optional findings.
- **Exact failure prevented:** A source change is declared complete after a
  happy-path unit test while its negative/counterfactual behavior, feature seam,
  or evidence boundary is wrong; alternatively, a review “fix” expands into an
  unrelated refactor. This is a source-process failure, not a benchmark result.
- **Smallest sufficient change (inference):** Add a short PR/issue block, not a
  schema: `base SHA and worktree`, `changed paths/seam`, `invariant and
  prevented failure`, `focused positive + counterfactual command/result`,
  `shipping gate/result`, `existing status/artifact`, and `unknown or blocker`.
  The writer supplies it; a fresh, read-only reviewer checks only those fields,
  the diff, and the cited owner docs. Use the existing multi-agent/review
  workflow; do not create a reviewer service or an orchestration queue.
- **Counterfactual and metric:** Remove or invert the negative assertion in a
  local review fixture and confirm the focused test turns red. Across the next
  five tickets record writer-to-first-review elapsed time, correction passes,
  unscoped changed paths, and findings that would have escaped without the
  counterfactual. Success is fewer correction passes or faster first actionable
  feedback, not a readiness score.
- **Dependency/order:** Apply before the next source PR; it is independent of
  the completed #240--#246 order and needs no provider or live Site.
- **Classification:** Docs/process only (issue/PR text); optional local review
  prompt, with no source or runtime artifact changes.
- **Evidence boundary:** The handoff may cite source-test output and existing
  status. It must not label those outputs as Run Definition, Checkpoint,
  exposure, reset, admission, or benchmark evidence.

### DX-2 — Make source CI self-diagnosing in the workflow summary

- **Observed WARP friction:**
  `.github/workflows/taskgen-acceptance.yml` routes changes and runs two test
  lanes plus package proof, then the aggregate job reports only a lane result
  ([workflow](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/taskgen-acceptance.yml:16)).
  A reviewer must open raw logs to learn whether the route was `skip`, which
  command ran, and where to start.
- **Exact failure prevented:** A skipped route is mistaken for a green test, a
  package-proof failure is rerun as a core-test failure, or a failing command is
  re-executed without its first diagnostic. This wastes cycle time and can hide
  a missing focused check; it does not change the acceptance decision.
- **Smallest sufficient change (inference):** After route and after each lane,
  append a compact `$GITHUB_STEP_SUMMARY` section containing commit/ref,
  route decision, lane, exact command, exit status, and a safe next command.
  Keep the existing route script, matrix, aggregate `always()` job, lockfile,
  and workflow/ref concurrency unchanged. Never print environment values,
  credentials, prompts, provider responses, or live-run identifiers that are
  not already public source metadata.
- **Counterfactual and metric:** Run the same intentionally failing focused
  lane with and without the summary and compare time to identify the owner and
  next command. In normal use record time from CI failure to first corrective
  command, duplicate reruns, and wrong-lane reruns. The summary is useful only
  if it points to a command that reproduces the failure locally.
- **Dependency/order:** Can land as an independent CI-only change after the
  completed #240--#246 source DAG, after confirming the summary itself is
  source-only.
  Do not apply the pattern to Modal, browser, reset, or paid-provider jobs.
- **Classification:** CI configuration only.
- **Evidence boundary:** A GitHub check summary reports source validation. It
  never upgrades a Run Artifact, Checkpoint, safety/exposure/admission check,
  reset result, or benchmark evidence.

### DX-2a — Validate provider output at the real production boundaries

- **Observed WARP friction:** The paid failures recorded on #212 were followed
  by source repairs to sandbox/host output parity and family selection
  contracts. Representative GitLab and Rocket.Chat Phase 1 pipeline tests
  substitute only the sandbox transport and still execute the production
  parser, family compiler, generic validator, metadata, and cache behavior.
  Other lower-level tests patch narrower seams. The direct contract-bound tests
  use provider-shaped objects but do not establish compatibility with a genuine
  SDK response. The tracked Phase 4 judge fixture owns a different contract;
  no retained Phase 1 artifact contains a raw generation response suitable for
  honest replay.
- **Exact failure prevented:** An SDK/provider envelope or sandbox file response
  reaches production extraction but cannot cross the existing compiler and
  validator. Deterministic family behavior is already covered; the missing
  evidence is producer-to-consumer compatibility, not another test-owned model
  of the producer.
- **Accepted smallest sufficient change:** Do not
  add a no-network command, provider mock service, alternative parser, or
  valid/invalid fixture pair for every family. When the exact frozen provider
  route is available, call `generate_new_tasks_for_site` twice for one row with
  card slicing disabled. A host-action-only card and capability profile that
  require the direct backend exercise `generate_contract_bound_action_tasks_api`
  plus the top-level final validation/cache path; a model-owned,
  non-host-action-only card must reach `run_claude_in_sandbox`. Use a disposable
  output/state root and do not touch an external Site or browser; the calls
  still incur provider usage, write generation files, and may create sandbox
  work. Before dispatch, assert and record only the non-secret effective route:
  the expected OpenRouter base host and frozen model, required auth variable
  present, and higher-precedence OAuth variable absent. Never print the token.
  Retain the compiled output, current cost summary, and telemetry where the
  owning backend already produces it, under the existing run policy; current
  entry points do not preserve a raw provider envelope. Only if a real boundary
  regression recurs should a deliberately reviewed, sanitized recorder capture
  one genuine response for replay beside its owner through the untouched
  production extractor/compiler/validator.
- **Counterfactual and metric:** Each micro-canary succeeds only if the genuine
  producer output crosses the real consumer boundary and yields one validated
  row. A separate Anthropic credential can establish that provider's transport
  shape, but not the frozen OpenRouter route; label it accordingly.
- **Dependency/order:** The source tests remain independent of OpenRouter. The
  exact-route micro-canaries run immediately before a paid Phase 1 resume and
  remain outside ordinary credential-free CI.
- **Classification:** Later runtime validation through existing Phase 1 source;
  no immediate source or CLI slice.
- **Evidence boundary:** The canaries establish transport/output compatibility
  only. They do not establish task quality, admission, Site behavior, exposure,
  grading, or benchmark results; Phase 2c and the later sandbox checks retain
  those authorities.

### DX-2b — Remove duplicate readiness walks without deleting assertions

- **Observed WARP friction:** Two default-suite tests each call the full
  `readiness_audit.build_audit()` repository walk. The package-proof lane also
  runs the canonical production audit with every required failure category.
  Exact current per-call cost is unknown because `run_silent` hides successful
  node durations; the first visibility slice must measure it.
- **Exact failure prevented:** Every focused/default acceptance cycle spends
  another repository walk and AST/token scan, lengthening writer and reviewer
  feedback without increasing gate coverage.
- **Smallest sufficient change (inference):** First expose the real node
  durations. If the two walks consume material time, keep the standalone
  production readiness audit and its fail-on policy unchanged while replacing
  both default-suite repository walks with focused inputs to the same semantic
  assertions. Otherwise retain them. Do not delete either assertion set or its
  relevant negative case.
- **Counterfactual and metric:** Time the same acceptance lane before and after;
  both assertion sets and the canonical audit's failing counterexample must
  remain effective while the two pytest repository walks disappear.
- **Dependency/order:** Independent, no provider or live Site.
- **Classification:** Source-test optimization only.
- **Evidence boundary:** This preserves the existing readiness gate; it does
  not weaken safety, integrity, resume, exposure, or grading checks.

### DX-2c — Route the disjoint root gate inside the required job

- **Observed WARP friction:** The root workflow takes about 111 seconds median
  elapsed on recent Taskgen PRs; its `check-root` job itself takes 106 seconds
  median. Its Black/Ruff exclusions omit `packages/`,
  and its mypy `files` list names only the evaluation-awareness tree and root
  scripts. The workflow intentionally avoids GitHub `paths:` filters so the
  required check never disappears.
- **Exact failure prevented:** A Taskgen-only PR waits for an unrelated root
  environment sync and root lint/type pass before merge. The opposing failure
  is worse: an overbroad path filter can skip a root check that should run.
- **Smallest sufficient change (inference):** Keep the workflow and required
  check. Inside the job, resolve the base and skip setup/checks only when every
  changed path is inside the Taskgen-owned tree or its acceptance entrypoints.
  Any empty, unknown, or unresolvable diff and every root configuration,
  workflow-router, or evaluation-tree change runs the full gate. Do not use a
  workflow-level `paths:` filter.
- **Counterfactual and metric:** A Taskgen-only fixture returns `skip`; one
  root Python/config path returns `run`; an unknown base returns `run`. Measure
  the eliminated wait on five Taskgen PRs and any false route decision.
- **Dependency/order:** Independent after DX-2's summary makes the route
  visible. This is an explicit tradeoff because it adds a small path map.
- **Classification:** Root CI routing only.
- **Evidence boundary:** A skipped disjoint root lane says nothing about
  Taskgen correctness; the Taskgen acceptance aggregate remains mandatory.

### DX-3 — Retain focused failure output as a bounded CI artifact

- **Observed WARP friction:** `run_silent.sh` intentionally preserves full
  failure output locally, but the acceptance workflow has no selected artifact
  for a reviewer after the hosted job expires or the raw log is difficult to
  navigate.
- **Exact failure prevented:** A reviewer reruns a slow broad lane only to
  recover a truncated traceback, or copies an incomplete log into a ticket and
  loses the first failing assertion. Conversely, uploading every runtime file
  would expose secrets or contaminate evidence provenance.
- **Smallest sufficient change (inference):** Wrap each source-only lane with
  a stable, bounded test-output file (for example, the existing pytest/JUnit or
  captured stderr path) and upload it with `if: failure()` or `if: always()` plus
  an explicit short retention period. Upload only source tests, route output,
  and package-proof diagnostics; exclude `.env`, provider payloads, Modal state,
  browser traces, Run Artifacts, and credentials. Link the artifact from the
  job summary. If the current command cannot produce a bounded file without
  changing its semantics, keep the summary-only change (DX-2) and do not add a
  wrapper.
- **Counterfactual and metric:** Delete the artifact step in a test branch and
  compare time to recover the first failing assertion from the hosted job. Track
  artifact-open rate, duplicate reruns, and whether the saved file contains the
  same first failure as the raw command. Artifact availability is not a pass
  criterion and must not be used in a paper table.
- **Dependency/order:** Follow DX-2; verify retention and permissions in a
  source-only pull request before adding any artifact upload. Keep the existing
  `contents: read` permission and do not grant cloud credentials.
- **Classification:** CI configuration only.
- **Evidence boundary:** These are disposable CI diagnostics, separate from
  WARP Run Artifacts and runtime provenance. Never merge them into admission,
  checkpoint, reset, PVPO, VEA, or attack-result claims.

### DX-4 — Make the existing command card explicit, without a duplicate policy tree

- **Observed WARP friction:** The canonical acceptance command is root-facing,
  while most verification commands are documented as running from
  `packages/warp-taskgen/`. Agents can select a plausible command from the
  wrong cwd or use an unlocked environment even though the repository already
  has a locked gate.
- **Exact failure prevented:** A source ticket passes in an ambient or stale
  environment but fails in the fresh-checkout/package proof, or a developer
  runs an excluded live-marked suite for a docs/source-only change. This is
  configuration drift, not evidence of a working Site.
- **Smallest sufficient change (inference):** In the existing Taskgen `CLAUDE.md`
  or execution-plan ticket block, add a four-line command card: repository root,
  package cwd, route-only command, focused command, and shipping command
  (`bash scripts/accept_taskgen.sh` or the lane named by the ticket). State that
  dependency changes use `uv sync --locked`/`uv run --locked`. Link to
  `agent_docs/verification.md` rather than copying it. If GitHub Copilot or
  another consumer is enabled later, add a pointer-only instruction file; do
  not fork these rules into `.github/copilot-instructions.md` and `AGENTS.md`.
- **Counterfactual and metric:** Start a fresh checkout with a deliberately
  stale lock or wrong cwd and confirm the command fails before source results
  are trusted. For five tickets record first-command success, lock-drift
  failures, and time to reach the owning focused test. A successful local run
  is not evidence that a remote Site or provider is ready.
- **Dependency/order:** Docs-only and can land before source slices. Keep
  feature-owned seams from the code-organization guide; never move behavior to
  a generic command helper solely to make the card shorter.
- **Classification:** Docs-only (with an optional one-line shell alias in
  existing local tooling only if the command is already canonical).
- **Evidence boundary:** Locked source checks establish reproducibility of
  source dependencies, not Run Definition or Checkpoint compatibility beyond
  the tests that explicitly exercise those contracts.

### DX-5 — Record clean-state resumption, not a new progress manifest

- **Observed WARP friction:** WARP already requires short-lived worktrees and
  one PR per focused behavior, but a paused or handed-off source task can still
  make the next agent infer the base, changed paths, and unresolved blocker from
  chat or stale logs.
- **Exact failure prevented:** A resumed agent tests the wrong base, repeats a
  paid/live command that is currently blocked, or edits around an uncommitted
  unrelated file. It can also confuse a source-test artifact with a runtime
  Run Artifact.
- **Smallest sufficient change (inference):** Add a fixed five-line handoff to
  the issue or PR description: `base SHA`, `worktree clean?`, `changed paths`,
  `last focused/shipping command and result`, and `next safe command or named
  blocker`. Link the existing status/artifact path and explicitly write
  `unknown`, `blocked`, or `not_inspected` where applicable. Update it only at
  a meaningful handoff; do not add `progress.json`, a task registry, a second
  checkpoint, or a manifest/hash hierarchy.
- **Counterfactual and metric:** Give a fresh reviewer only the issue, commit,
  and handoff; it should reproduce the focused check without reading prior
  chat. Record clean-restart time, first-command success, dirty-worktree
  discoveries, and duplicate attempts after a known blocker. The handoff is
  successful when it prevents guessing, not when it makes an unverified run
  look complete.
- **Dependency/order:** Apply with DX-1 on every source ticket; no dependency on
  provider availability or the #240–#246 landing order.
- **Classification:** Docs/process only.
- **Evidence boundary:** Git state and local check output remain source-process
  evidence. They cannot certify reset isolation, ordinary-reader exposure,
  safety/admission, or any Phase 4 result.

### DX-6 — Measure first; do not expand the existing Claude-only hooks yet

- **Observed WARP friction:** Taskgen has committed package-local Claude hooks
  under `packages/warp-taskgen/.claude/`; the edit hook invokes a bare `ruff`
  command and ignores failures, while the stop hook uses `uv run` on changed
  tracked Python. Other agents use the canonical verification scripts instead.
  Expanding these hooks or copying them into each harness would create
  tool-specific drift and could run in contexts with credentials or network
  access.
- **Exact failure prevented:** A changed Python file reaches review without the
  narrow lint check, or a hook accidentally launches a provider/browser/reset,
  mutates shared state, or writes over another hook's input. Anthropic's hook
  semantics make sibling side effects possible even when one hook denies.
- **Smallest sufficient change (inference):** Do not add a hook in this slice.
  First collect DX metrics from five tickets. If the same source-only omission
  repeats, add one project-local, path-absolute hook that runs only
  `uv run --locked ruff check` on changed Python and exits nonzero with a
  readable remediation. It must have a timeout, no network/provider/browser/
  reset calls, no credential reads, no shared JSON writes, and an explicit
  opt-out for documentation-only work. Test it in a clean environment before
  committing. A hook is advisory convenience; the acceptance workflow remains
  the source gate.
- **Counterfactual and metric:** Compare pre-hook and post-hook focused-lint
  misses, false blocks, hook duration, and agent bypasses. If misses do not
  recur, reject the hook. Never count hook success as runtime evidence.
- **Dependency/order:** Last, after DX-1 through DX-5 metrics show a repeated
  local failure; independent of paid generation and live Sites.
- **Classification:** Deferred local tooling; no immediate source change.
- **Evidence boundary:** Hook output is local lint/process evidence only.

### DX-7 — Keep the provider-blocked path explicit and non-retrying

- **Observed WARP friction:** The provider-independent E3 source slices are
  complete while frozen-provider generation is blocked. A generic agent prompt
  can still treat the unavailable provider as a transient test failure and
  repeatedly retry.
- **Exact failure prevented:** An agent spends budget, invokes a model or live
  Site without approval, mutates reset-sensitive state, or records “zero rows”
  as if generation had run. That would weaken cost accounting and evidence
  validity.
- **Smallest sufficient change (inference):** Put one explicit escalation line
  in each source ticket: “No provider/browser/reset invocation; if the named
  dependency is absent, report `blocked` with the status command and stop.”
  The issue should link the existing E3 status and any next safe source-only
  ticket, not introduce a fallback provider, readiness probe, or orchestrator.
- **Counterfactual and metric:** In a dry, source-only review, remove the
  provider credential and confirm the prescribed command returns a named
  blocker without a network call. Track duplicate blocked retries and any
  attempted paid/live command. Zero unapproved invocations is a safety
  requirement, not an optimization target.
- **Dependency/order:** Apply to any new source-only follow-up before live
  generation; it is independent of CI changes.
- **Classification:** Docs-only issue wording.
- **Evidence boundary:** `blocked` is not `failed`, `empty`, `admitted`, or
  `evaluated`; no source status may be promoted into Run Artifact or paper
  evidence.

## Dependency-ordered adoption

1. Resolve DX-0's development-environment policy before the next source
   worktree. It determines whether writer and reviewer commands are genuinely
   reproducible.
2. Do not implement a no-provider output-contract rehearsal. Before another
   paid Phase 1 resume, run DX-2a's two production micro-canaries once the exact
   route is available. DX-2b may remove the duplicate readiness walks only after
   the visible duration trace shows they matter.
3. Add DX-2's source-only job summaries, preserving the current workflow/ref
   cancellation and aggregate gate. Add DX-2c's fail-safe internal root route
   as an independent source PR. Keep DX-1/DX-5 as concise PR/issue text.
4. Update the execution plan and parent issue at the next accepted planning
   join so #240--#246 are marked complete and the only current E3 runtime
   blocker is stated accurately. Do not add a freshness bot or manifest.
5. Add DX-3 only if a bounded failure file can be produced without changing
   command semantics or exposing runtime state. Measure the next five tickets,
   then decide whether any additional hook or root-CI routing earns its cost.

This sequence does not change the execution-plan joins: cache identity and
Phase 1 ownership remain prerequisites for safe Phase 1 resumes; cost status
still follows the cache work; Phase 2a, 2b, and 2c status remain ordered; and a
working frozen-provider route remains required before paid E3 generation. No
item here changes a Run Definition, feature Checkpoint,
safety/exposure/admission predicate, reset protocol, evaluator authority, or
publication gate.

## Minimal DX measurement packet

For the next five source tickets, record these process fields in the issue/PR
handoff, not in Run Artifacts:

- base SHA and whether the worktree was clean;
- minutes from delegation to first focused command;
- minutes from first focused command to first green result;
- number of writer/reviewer correction passes;
- number of reruns and whether a rerun was the wrong lane;
- changed-path count outside the owning seam;
- time for a fresh agent to reproduce the focused check from the handoff;
- blocked attempts that correctly stopped before provider/browser/reset access;
- whether the CI summary/artifact let a reviewer identify the first failure
  without rerunning the lane.

Compare these with the preceding small set of source tickets as an operational
counterfactual, labelling the comparison as an inference rather than a causal
experiment. Do not optimize for a vendor readiness level, token count, or agent
session length if that increases retries, broadens touched paths, or obscures an
unknown state. Do not copy these fields into benchmark cohorts, checkpoints,
evaluation tables, or paper claims.

## Explicit rejections

- **No new orchestrator, agent framework, event bus, or reviewer service.** The
  existing issue contract, worktree policy, feature ownership, and two-pass
  review bound are enough for these source slices.
- **No second manifest, progress database, hash hierarchy, or attestation
  layer.** Use git, existing Run Definitions/Checkpoints, and existing Run
  Artifacts according to their owners.
- **No duplicate instruction trees.** Do not add `.factory`,
  `.github/copilot-instructions.md`, or global AGENTS/CLAUDE copies unless a
  concrete supported consumer later requires a pointer-only file.
- **No global or side-effectful hooks.** Hooks cannot run paid requests,
  browser/API actions, reset calls, credential probes, or shared-state mutation.
- **No automatic provider fallback, readiness probe, budget cutoff, or retry
  loop.** A frozen provider blocker remains an explicit `blocked` state.
- **No upload of prompts, provider responses, browser traces, reset logs, or Run
  Artifacts to hosted CI.** Source-only diagnostics must be bounded, private to
  the repository's normal permissions, and clearly separate from runtime
  evidence.
- **No all-tests-every-edit rule.** Use the narrowest relevant focused command,
  then the existing acceptance/live gate when the change requires it. Broaden
  only when a failure or changed seam justifies it.

These rejections follow the primary-source guidance to keep always-loaded
instructions short, use deterministic hooks carefully, and make feedback
inspectable. They also preserve WARP's stronger safety and evidence contracts;
they are not arguments against future infrastructure when a measured failure
cannot be addressed by an existing feature-owned seam.
