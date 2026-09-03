# Provider-blocked DX vertical-slice research

- Research baseline: 2026-08-30
- Planning date: 2026-09-03
- Source authority inspected: `origin/main` at `46b8fc3a`
- Scope: source inspection and primary-source online research only; no provider,
  browser, infrastructure, dependency, or benchmark execution

This note narrows provider-independent DX work that can reduce incorrect reuse,
duplicate paid work, lost cost observations, and remote run-root ambiguity. It
does not add another evidence system or claim that source tests prove live
integration.

## Primary-source findings

### Cache identity

[Bazel's remote-cache documentation](https://bazel.build/remote/caching)
defines an action using declared inputs, command line, outputs, and environment.
It also identifies undeclared environment and tools as causes of incorrect
shared hits. The relevant lesson for WARP is narrow: environment values that
actually change generated content or its validator belong in the existing
cache identity. It is not a reason to hash unrelated runtime state.

On `origin/main`, `WORLDSIM_PHASE1_DIVERSITY_SALT` changes the contract-bound
generation prompt, while `WORLDSIM_PHASE1_FORBIDDEN_REFERENCES` changes both
that prompt and slot validation. Neither is included in the current Phase 1
site-cache fingerprint. Consequently, setting either value can leave an old
cache classified reusable.

The existing fingerprint functions are already the owning module and seam.
The smallest extension is a feature-owned helper in the contract-bound
generation module that returns:

- the exact nonempty diversity-salt string, because its bytes reach the prompt;
- the sorted normalized forbidden-reference set, because order, case, and
  separator whitespace do not change the validator's semantics.

Only a site whose direct plan or any card slice uses contract-bound generation
should consume those inputs. The unsliced host-action path reads the same
environment values and must not be omitted. Model-only sandbox caches should
not miss because of an environment value they never read.

Compatibility policy:

- With both current values absent, an existing valid cache remains eligible.
  Absence requests neither a new semantic batch nor a forbidden-reference
  restriction.
- With either current value present, legacy metadata cannot prove a match, so
  the cache is stale and must regenerate.
- Do not rewrite old metadata or add another manifest. The existing status
  inspector should expose the ordinary fingerprint mismatch.

This is a current-constraint compatibility policy, not proof that an old run
used no salt. Old metadata cannot reveal whether a nonempty value was once
present. The accepted policy treats a currently absent salt as no request for a
fresh distinct batch and a currently absent forbidden set as no restriction;
it does not claim byte-identical historical prompt provenance.

The focused counterfactual is one cache fixture: hold every other input fixed,
change each value independently, and show that the production inspector moves
from reusable to stale. Reordering or duplicating equivalent forbidden phrases
must not cause a miss. A model-only card plan must remain unaffected.

### Single-owner locking

[Python's `fcntl` documentation](https://docs.python.org/3/library/fcntl.html)
provides nonblocking exclusive `flock` operations on Unix.
[Linux's `flock(2)` documentation](https://man7.org/linux/man-pages/man2/flock.2.html)
clarifies that these locks are advisory, attach to an open file description,
release when its file descriptors close, and have filesystem-dependent NFS and
SMB behavior. WARP therefore may claim only cooperating-process exclusion on
the same underlying filesystem, not universal cross-host ownership.

Phase 1 currently lacks the lifecycle lock already used by Phase 2 and Phase 4.
Two CLI processes sharing one state root can both make paid calls and race over
site caches, `benign_tasks.json`, resume metadata, and `cost_report.json`.

The smallest module is a Phase 1-local `phase_1_run_lock(state_dir)` with the
same nonblocking interface as the Phase 2 precedent, acquired by the existing
CLI lifecycle seam. Direct Python callers remain composable and are explicitly
not covered. No PID registry, stale-file cleanup, heartbeat, pause protocol, or
generic lock registry is warranted.

Tests should prove same-root contention fails before the operation, distinct
roots proceed, an exception releases the lock, and terminating the holder lets
the next cooperating process acquire it. Tests must not imply that identical
path strings on unrelated hosts coordinate.

### Remote state-root precedence

[Python's subprocess documentation](https://docs.python.org/3/library/subprocess.html)
states that a supplied `env` mapping becomes the child's environment.
[python-dotenv's documentation](https://bbc2.github.io/python-dotenv/)
states that `load_dotenv(override=True)` makes `.env` values win over existing
process values, whereas the default preserves existing values.

The remote launcher currently records its resolved `--state-dir`, sets the
legacy `WORLDSIM_STATE_DIR`, but uses `setdefault` for canonical
`WARP_TASKGEN_STATE_DIR`. Taskgen resolves the canonical name first. An ambient
canonical value can therefore send the child to a different root from job
metadata and status. The CLI's global `load_dotenv(override=True)` can then
replace both launcher values.

The recommended precedence is:

1. explicit remote-wrapper `--state-dir`;
2. inline assignment, but only when the wrapper option is absent;
3. inherited canonical `WARP_TASKGEN_STATE_DIR`, then the legacy alias;
4. `.env` fallback;
5. the ordinary default root.

Implement this without changing dotenv precedence for credentials: set both
aliases unconditionally when the wrapper option is present, reject a command
that also assigns either state-root name, and carry one non-secret explicit-root
marker that restores both aliases immediately after `.env` loading. Do not
globally change `load_dotenv(override=True)` for secrets and provider
configuration. Helpers that read either alias then observe one root, including
nested WARP workers.

The counterfactual test matrix supplies distinct sentinel roots through the
wrapper, inline command, ambient environment, and `.env`, then asserts the
child's actual `get_state_dir()`, metadata, and status all identify the expected
root. An explicit-wrapper/inline conflict must fail before launching a child.

### Cost durability and backpressure

[OpenRouter's usage-accounting documentation](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
says completed responses include token counts and charged cost, and that a
generation ID may support later lookup. [Anthropic's error documentation](https://platform.claude.com/docs/en/api/errors)
shows that billing, permissions, rate limits, spend limits, overload, and even
post-HTTP-200 stream failures have distinct error behavior. A failed or
interrupted client call therefore cannot safely be counted as zero.

[Python documents](https://docs.python.org/3/library/os.html#os.replace) that a
same-filesystem replace is atomic. WARP's existing atomic JSON writer already
uses a same-directory temporary file and replacement, but does not make a
power-loss/fsync guarantee. The appropriate claim is parseable old-or-new
process-crash behavior, not hardware-durable accounting.

Phase 1 currently records returned summaries in memory and saves the existing
`cost_report.json` only at successful phase completion. A later exception can
therefore lose earlier returned-call costs, while malformed prior JSON is
warned about and may later be overwritten from an empty tracker.

The smallest vertical behavior keeps `cost_report.json` as the only artifact:

- after every returned successful paid call, record its summary and atomically
  save immediately;
- after a caught paid-call exception, record one null-cost failure and save;
- label totals as an observed-cost lower bound and report null-cost entries as
  unknown rather than zero;
- refuse another Phase 1 paid dispatch when the existing report is malformed,
  with an actionable error rather than a traceback;
- render the observed total, recorded-call count, and unknown count in the
  existing local and remote status surfaces.

A process killed while a provider request is in flight remains unknowable from
local evidence. Adding start/update lifecycle IDs for every provider attempt
would spread ordering constraints across callers and still would not guarantee
the final provider invoice. That machinery is not justified for this slice.

The accepted `$3,000` value remains operational backpressure, not a promised
invoice cap. [OpenRouter's own workspace-budget documentation](https://openrouter.ai/docs/guides/features/workspaces/workspace-budgets)
notes that even provider-side enforcement permits already in-flight requests
to complete and overshoot. WARP should preserve its frozen retry-inclusive
estimate, existing per-call guards, and observed ledger without adding an
automatic run-level cutoff.

### Provider readiness

[OpenRouter documents](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key)
that an ordinary current-key request returns the key's configured limit,
remaining limit, usage, and reset policy. This could detect the exact key-limit
failure without a model call. In WARP's Phase 1 route, however, the operative
key may exist only in a named Modal secret. Reading it requires launching a
small networked sandbox with that secret.

That check should not become part of default Phase 1 behavior. A successful key
metadata request does not prove that the frozen model and transport will accept
the later request, while a new mandatory network dependency can block a route
that would otherwise work. Omit it from the immediate slices. If repeated
failures justify it later, use an explicit advisory no-model probe inside the
same Modal secret context; never print the key, claim full readiness, change
provider/model identity, or fall back to another credential automatically.

### Phase 2 checkpoint visibility

Phase 2 planning, text fill, and feasibility already own strong checkpoint
validators. Current `status` usually projects optional pipeline-state counters
instead of applying those validators, so a paused or crashed run can show
unknown counts despite reusable task-local evidence.

Before the first Phase 2 run, extend the existing status interface—not a new
command—to report compatible, pending, stale/malformed, and not-inspected
counts through the owning feature validators. Missing reconstruction context
must remain `not_inspected`, never zero. Status stays read-only and the pipeline
state remains lifecycle authority.

### Agent-development readiness

[Factory's Agent Readiness model](https://docs.factory.ai/agent-readiness/overview)
emphasizes fast local tests, exact build commands, maintained documentation,
reproducible environments, observability, security controls, and discoverable
work. Its [AGENTS.md guidance](https://docs.factory.ai/harness/agents-md) further
recommends short, concrete, versioned instructions with exact focused and full
verification commands, nested only when a subtree has genuinely different
rules. [OpenAI's harness-engineering report](https://openai.com/index/harness-engineering/)
likewise treats the root agent guide as a map, repository-local documentation as
the system of record, and inspectable state plus mechanically enforced boundary
invariants as the basis for longer autonomous work. Anthropic's
[Claude Code workflow guidance](https://www.anthropic.com/engineering/claude-code-best-practices)
adds two relevant practices: give an agent a clear testable target and use
independent contexts for source work and review.

WARP already implements most of the useful foundation:

- the root `AGENTS.md` is a short router and the Taskgen `CLAUDE.md` points to
  behavior-owned branch guides;
- `agent_docs/verification.md` defines focused, default, live, and specialized
  checks, including quiet failure-preserving wrappers;
- `scripts/accept_taskgen.sh` routes changed paths into split CI lanes and also
  exercises installed-package behavior;
- `scripts/readiness_audit.py` enforces measured repository risks rather than a
  generic maturity score;
- the repository already requires one isolated topic worktree and one PR per
  focused change, while benchmark mutation remains restricted to configured
  sandboxes.

The remaining development-experience improvement is therefore a small execution
contract in each existing source ticket, not another framework. Before a ticket
is delegated, record in its issue or existing execution plan:

1. the owning feature module and narrow seam;
2. the behavioral invariant and the concrete failure it prevents;
3. one positive and one relevant negative/counterfactual test;
4. the exact focused command, then the existing shipping gate;
5. the existing artifact or status field that makes the result inspectable;
6. the boundary between offline source proof and later sandbox evidence; and
7. the condition that warrants escalation rather than another blind retry.

This is ticket content, not a new manifest or permanent schema. It makes an
implementation task independently executable and reviewable while preserving
feature-local ownership. New diagnostics in the approved cache, lock, cost,
remote-root, and Phase 2 status slices should identify the failed invariant,
relevant path or state root, and next safe command where one exists. JSON and
human output should describe the same state, including explicit `unknown`,
`malformed`, `blocked`, or `not_inspected` values instead of inferring success
from absence.

No extra source slice is justified merely to improve a readiness score. In
particular, do not install Factory QA, add `.factory` orchestration, duplicate
AGENTS/CLAUDE instruction trees, create a readiness dashboard, introduce a
generic event bus, add global hooks, or upload benchmark evidence to a third
party. Factory itself presents its readiness report as a diagnostic with
prioritized actions; its higher autonomy levels are not WARP correctness
requirements. WARP should extend mechanical enforcement only for a failure
observed in these concrete slices.

## Dependency-ordered slices

1. Correct contract-bound cache identity before another Phase 1 resume
   ([#240](https://github.com/jasmineee-li/warp/issues/240)).
2. Establish Phase 1 same-filesystem single ownership
   ([#242](https://github.com/jasmineee-li/warp/issues/242)).
3. Make returned-call costs durable and visible without claiming a hard cap
   ([#243](https://github.com/jasmineee-li/warp/issues/243)).
4. Make explicit remote state-root precedence truthful before the next remote
   Phase 1 launch.
   ([#241](https://github.com/jasmineee-li/warp/issues/241)).
5. Add Phase 2 checkpoint visibility before the first Phase 2 paid call, in
   planning, text-fill, and feasibility order
   ([#244](https://github.com/jasmineee-li/warp/issues/244),
   [#245](https://github.com/jasmineee-li/warp/issues/245), and
   [#246](https://github.com/jasmineee-li/warp/issues/246)).
6. Revisit Phase 4 host/instance preflight only before Phase 4 execution.

The cache, lock, and remote slices can be developed independently. The cost
slice should stack after the cache change because both touch Phase 1 generation
callers. Phase 2 status should stack after cost status because both extend the
same human status formatter.

For agent execution, every ticket uses the existing repository loop: confirm the
base and clean worktree, reproduce the counterfactual, implement through the
owning feature seam, run the focused command, run the existing acceptance lane,
obtain an independent diff review, and merge only the green focused PR. This is
an application of existing process, not an orchestration product or new evidence
format.

## Explicit non-goals

- no universal workflow or lock engine;
- no second resume planner or operator dashboard;
- no provider fallback or credential mutation;
- no new manifest, attestation hierarchy, or append-only attempt ledger;
- no Phase 1 per-slice task checkpoints;
- no automatic `$3,000` cutoff;
- no broad prompt-hash or “hash everything” framework;
- no paper edits or live benchmark claims.
