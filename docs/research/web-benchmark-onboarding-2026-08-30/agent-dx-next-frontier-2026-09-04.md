# Agent-DX next frontier after Taskgen CI split — 2026-09-04

- **Date:** 2026-09-04 (America/New_York)
- **Fixed point:** `origin/main` at `eac23227565c71ee3003eb2c1bc977fe04660dea`
- **Scope:** read-only WARP source inspection, first-party documentation, one
  redacted read-only OpenRouter capacity probe, and one attempted exact
  production canary that stopped at the provider's quota error; repository
  changes are planning documentation only.
- **Decision status:** ranked candidates for follow-up; no implementation is
  implied.

This note asks what is still worth doing after Taskgen CI changes #259–#261.
It ranks only bounded candidates with a clear owning seam. Claims marked
“WARP evidence” come from the checked-in source or linked hosted runs. Claims
marked “general guidance” are mechanics from first-party documentation, not a
WARP performance claim.

**Decision:** no new CI performance PR is justified by the current evidence.
The remaining high-value work is operational legibility, not another test
framework: make provider capacity explicit before a paid canary, make the
effective Phase 1 backend reconstructible for new Runs, and make the canonical
source-local commands unambiguous. The leaf-test change below remains a
measurement candidate only; one trace suggests roughly 9–10 seconds of ideal
context-lane scheduling headroom, not a measured or promised workflow gain.

## Baseline: #259–#261 are complete, not proposals

The current gate already has the intended safety shape:

- [#259 routes the root gate internally](https://github.com/jasmineee-li/warp/commit/4ec961309aabb5d848f5eb77558560363ffe8cec),
  keeping the required check visible and failing open when the base cannot be
  resolved.
- [#260 exposes pytest durations](https://github.com/jasmineee-li/warp/commit/66be5bf5d13757bdf14171d9f4cea1a9aeb40e4b)
  through the existing acceptance wrapper.
- [#261 keeps the package proof and aggregate, splits context/feature core
  lanes, retains the remote lane, and raises `max-parallel` to 3](https://github.com/jasmineee-li/warp/commit/eac23227565c71ee3003eb2c1bc977fe04660dea).
  The implementation is in the [workflow](https://github.com/jasmineee-li/warp/blob/eac23227565c71ee3003eb2c1bc977fe04660dea/.github/workflows/taskgen-acceptance.yml#L16-L106)
  and [acceptance wrapper](https://github.com/jasmineee-li/warp/blob/eac23227565c71ee3003eb2c1bc977fe04660dea/scripts/accept_taskgen.sh#L268-L324).

The two decisive hosted runs show variance, not a single stable critical lane:

| Run | Feature job | Context job | Remote | Package proof | Aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| [PR #261](https://github.com/jasmineee-li/warp/actions/runs/33918813364) | ~90 s (pytest 55.30 s) | ~76 s (pytest 57.67 s) | ~37 s | ~53 s | ~4 s |
| [merge push](https://github.com/jasmineee-li/warp/actions/runs/33919087262) | ~91 s | ~103 s | ~36 s | ~57 s | ~4 s |

Checkout is roughly 9–11 s and cached setup-uv is 1–3 s. The context lane's
`test_leaf_modules_import_in_either_order` took 45.07 s in the PR run because
one pytest item launches ten fresh interpreters; the other import-order items
were 15.12, 14.53, and 13.43 s. The source is
[here](https://github.com/jasmineee-li/warp/blob/eac23227565c71ee3003eb2c1bc977fe04660dea/packages/warp-taskgen/tests/phase_4/test_leaf_context_boundary.py#L169-L183).
Thus, splitting that item could improve balance but cannot honestly promise a
20-second end-to-end reduction: on the PR run feature was already slower, and
on the push the context/feature difference was only about 12 s. The old
pre-#259 root-gate timings are historical and must not be used as today's
critical path.

The #261 failure run also matters for agent feedback: [seven feature-lane
`test_state.py` failures](https://github.com/jasmineee-li/warp/actions/runs/33917606170)
were caused by leaked state/facade patches and were fixed by the existing
autouse isolation. This supports retaining correct test isolation. It does not
by itself establish a need for another shard, retry class, or CI-summary change.

## First-party mechanics checked on 2026-09-04

- GitHub matrix combinations are independent jobs; jobs run concurrently by
  default, `max-parallel` caps active combinations, and matrix `fail-fast`
  cancels siblings after a failure. See [workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
  The existing ref-scoped [`concurrency`](https://docs.github.com/en/actions/concepts/workflows-and-actions/concurrency)
  cancellation should remain. `$GITHUB_STEP_SUMMARY` is the supported concise
  diagnosis surface; [workflow commands](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands)
  document the per-step 1 MiB/20-summary limits and masking behavior. [Artifacts
  and caches](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
  are different: artifacts persist selected outputs, while caches are for
  reusable dependencies/intermediate files.
- Pytest's [`--durations`](https://docs.pytest.org/en/stable/how-to/usage.html#profiling-test-execution-duration)
  reports slow setup/test nodes. [`parametrize`](https://docs.pytest.org/en/stable/how-to/parametrize.html)
  creates independently selectable items with useful IDs. xdist's
  [`worksteal`](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)
  is intended for uneven durations, but [every worker must collect the same
  ordered node list](https://pytest-xdist.readthedocs.io/en/stable/how-it-works.html);
  this constrains dynamic or non-deterministic sharding.
- uv's [`--locked` sync/run](https://docs.astral.sh/uv/concepts/projects/sync/)
  rejects stale lock state; its [project layout](https://docs.astral.sh/uv/concepts/projects/layout/)
  keeps a managed `.venv` beside the project; and its [cache guidance](https://docs.astral.sh/uv/concepts/cache/)
  permits concurrent reads/writes and recommends CI caching for package
  artifacts, not a mutable environment. The setup-uv release history records
  v10.0.0 (2026-08-12) sensitive-event cache behavior and v10.0.1 (2026-08-14)
  transient manifest-timeout handling ([releases](https://github.com/astral-sh/setup-uv/releases));
  neither is a measured WARP optimization.
- Factory's [AGENTS.md](https://docs.factory.ai/harness/agents-md) and
  [readiness](https://docs.factory.ai/agent-readiness/overview) guidance favor
  short exact commands, reproducible environments, and nested instructions
  only where scope genuinely differs. Anthropic's [Claude Code best
  practices](https://code.claude.com/docs/en/best-practices) emphasize explicit
  verification criteria, clean state, and fresh review context. OpenAI's
  [harness engineering](https://openai.com/index/harness-engineering/) likewise
  treats legible feedback loops and inspectable state as the high-leverage
  agent surface. These are process guidance, not evidence that a vendor score
  predicts WARP throughput.

## Ranked candidates

### P0 operational slice — Check real OpenRouter capacity before a paid canary

- **Observed failure prevented:** on 2026-09-04 the configured credential
  authenticated to `GET /api/v1/key` with HTTP 200, but the redacted response
  reported no usable capacity. The exact one-row production generation canary
  then failed with HTTP 403 `quota_exceeded` before producing model output. A
  credential-presence check cannot distinguish this state from usable
  capacity. Exact account limits, usage and expiry remain local rather than
  entering a public planning artifact.
- **Smallest owning seam:** add an operator-invoked, read-only capacity step to
  [execution-plan E3](execution-plan.md#e3-provider-boundary-resumption-check)
  immediately before its two real one-row canaries. It calls OpenRouter's
  documented [current-key endpoint](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key)
  with the same credential and emits only provider, auth result,
  provider-reported limit/usage/remaining capacity, reset, expiry, and
  timestamp. It must fail closed on an unknown response and must never run in
  default CI, `status`, or automatic `resume`. Start with the runbook step;
  promote it to a Taskgen command only if a second caller appears.
- **Interface and locality:** this is a true-external diagnostic, not a model
  adapter. It does not select a model, alter provider fallback, inspect prompts,
  or declare task-generation readiness. The production canaries remain the
  only evidence that the SDK envelope and WARP extraction/compiler/validator
  path work.
- **Proportionate validation:** today's real zero-capacity response is the
  negative case. After quota or a key is restored, one real nonzero response is
  the positive case, followed by the existing direct and sandbox one-row
  canaries. If code is later justified, parser tests need only minimal redacted
  response shapes plus an unknown-schema failure; do not implement a fake
  provider or generated-output fixture bank.

### P1 unresolved specification frontier — Bind effective Phase 1 routes for future Runs

- **Observed failure prevented:** Run15 reports 0/100 reusable tasks under the
  ambient environment and 20/100 reusable Postmill tasks only after restoring
  `WORLDSIM_PHASE1_CONTRACT_BOUND_API=1`; 80 GitLab tasks are absent either
  way. The backend selector affects the Phase 1 cache identity, but the Run
  Definition does not record it, so a fresh agent must reconstruct intent from
  shell history. This is a real resume-identity incident, not merely a possible
  future ambiguity.
- **Route shape, not a global boolean:** Phase 1 chooses the backend for each
  site/card slice. One plan may therefore be mixed. The narrow target is one
  pure Phase 1 resolver that returns a stable active-card route map such as
  `card_id -> contract_bound_api | sandbox` after applying card ownership,
  required profiles, and the operator opt-in. Generation and cache identity
  consume that result instead of independently rereading the environment.
- **Run-identity seam still needs a decision:** the CLI currently binds the
  Run Definition before `phase_1_tasks.run()` loads or compiles the task-card
  plan. Binding the route map to a future Run Definition therefore requires
  moving a read-only plan-resolution step before identity binding and reusing
  the same resolver; computing it twice or adding a scalar pass-through wrapper
  would destroy locality. The narrower alternative is to persist the map only
  in existing Phase 1 resume metadata, which improves diagnosis but cannot
  reject a mismatch before dispatch. The specification must choose explicitly.
- **Compatibility decision required before implementation:** existing defined
  Runs such as Run15 lack the field. Model missing as `unknown`, specify whether
  exact resume requires an explicit operator route assertion or a Derived Run,
  and test the policy against Run15's actual envelope. Do not silently backfill,
  rewrite, or reject Run15. Because the current cache fingerprint hashes the
  raw environment string, the specification must also state the cache
  version/migration rule before source work begins. Until those choices are
  accepted, this is not an authorized source ticket despite its observed value.
- **Proportionate validation after specification:** test pure route resolution
  for env-selected, profile-required, sandbox, and mixed plans; new-Run
  persistence if chosen; exact matching resume; mismatch rejection before any
  paid dispatch; and the chosen missing-field behavior. Status projection is
  diagnostic, not the enforcement test. This work makes intended routes
  reconstructible; it does not create the 80 absent GitLab rows or prove their
  quality.

### Conditional diagnostic (not an approved perf PR) — Parameterize the five leaf import-order checks

- **Measured baseline / failure prevented (WARP evidence):** one item at
  `test_leaf_context_boundary.py:169-183` performs five module checks × two
  fresh `sys.executable` processes, so xdist cannot rebalance the 45.07-second
  item while other workers finish. This creates slow, uneven context feedback;
  it does not indicate a missing assertion.
- **Smallest owning seam:** parameterize `LEAF_MODULES` into five stable IDs
  and retain the same two subprocess statements, `cwd`, `PYTHONPATH`, and
  assertions. The existing `-n 4 --dist worksteal` command and lane stay
  unchanged. Do not batch in one interpreter: the fresh-interpreter import
  invariant is the proof. Note that four xdist workers may still launch eight
  child interpreters concurrently, so hosted stability matters as much as the
  nominal split.
- **Validation / rollback:** first obtain repeated ordinary hosted timings on
  the same SHA. If they still show this item as a dominant indivisible node,
  run the focused parameterized item, compare stable `--collect-only` IDs and
  the exact two-process-per-case behavior, then run the full context lane.
  Compare repeated matched before/after runs and report both context-lane and
  whole-workflow critical-path medians; do not land if an assertion disappears,
  flake rate rises, or the timing change is noise.
- **Expected gain / uncertainty:** one PR trace has about 195 seconds of heavy
  context-node work, whose ideal four-worker floor is about 49 seconds versus
  57.67 seconds observed. That is an estimate of scheduling headroom, not
  worker-level evidence or a validated bound. Parameterization cannot reduce
  subprocess CPU; feature or package work may remain critical, making the
  end-to-end gain zero.
- **Dependencies / rejected complexity:** pytest-xdist already meets the
  required version (`>=3.8.0`); no provider or browser is involved. Reject a
  third core shard, a duration manifest/custom scheduler, `-n` grid tuning, or
  same-process batching until a measured trace proves need.

### P1 documentation slice — Put the first commands in the canonical Taskgen guide

- **Observed failure prevented:** package-focused checks run from
  `packages/warp-taskgen/`, while repository shipping acceptance runs from the
  root. This distinction exists across the router and verification guide but
  is not presented as one first-command card; retained-worktree inspection has
  already hit a wrong/uninitialized environment before collection.
- **Smallest owning seam:** add a short card to the existing
  `packages/warp-taskgen/CLAUDE.md`, which already owns the working loop and
  root acceptance command: canonical locked sync, one focused package test,
  one named acceptance lane, and the shipping command, each with its required
  working directory. The verification guide remains the detail owner and other
  plans link to the card instead of copying it.
- **Proportionate validation:** truth-check each command's `--help` or
  route/collection-only behavior from a clean worktree. No helper script,
  environment manager, hook, or five-ticket scorecard is warranted.

### P2 candidate — Publish only CI summary data that the current lane owns

- **Observed gap:** the workflow has no `$GITHUB_STEP_SUMMARY`
  ([workflow lines 29–83](https://github.com/jasmineee-li/warp/blob/eac23227565c71ee3003eb2c1bc977fe04660dea/.github/workflows/taskgen-acceptance.yml#L29-L83)).
  A completed lane knows its route, lane name, result, and static focused rerun
  command, but a fresh reviewer must currently reconstruct those from the
  workflow. Reduced rediscovery is an unmeasured hypothesis; the seven-failure
  #261 run proves a failure class, not that reviewers chose the wrong rerun.
- **Smallest owning seam:** an `if: always()` step may summarize route, lane,
  outcome, and safe rerun command for lanes that started or completed. Do not
  promise a summary for siblings cancelled by `fail-fast`. The current quiet
  wrapper deletes its temporary capture, so first-failure text and durations
  are out of scope unless the run step later emits a bounded, allowlisted,
  redacted report. That report would be a separate interface decision, not a
  reason to persist full logs.
- **Proportionate validation:** one workflow fixture for a known failure and
  one route-skip fixture are sufficient to confirm the summary and unchanged
  aggregate result. Observe normal use afterward; do not manufacture five CI
  failures or build a dashboard, telemetry service, artifact stream, or second
  runner. GitHub's 2026-06-25 [parallel steps](https://github.blog/changelog/2026-06-25-actions-steps-can-now-be-run-in-parallel/)
  are unnecessary because the matrix already supplies concurrency.

## External boundary status (not a CI substitute)

The checked-in [2026-08-08 OpenRouter endpoint research](https://github.com/jasmineee-li/warp/blob/eac23227565c71ee3003eb2c1bc977fe04660dea/docs/research/openrouter-l3-endpoint-selection-2026-08-08.md)
is historical: that replacement-key probe authenticated, returned 391
policy-filtered models, completed the unchanged
`anthropic/claude-sonnet-4-6` tool request, passed the locality wrapper, and
observed HTTP 200 responses in a two-task Phase 4 canary. It does not establish
capacity on 2026-09-04.

Current redacted evidence is:

| 2026-09-04 check | Result | Establishes |
| --- | --- | --- |
| OpenRouter `GET /api/v1/key` | HTTP 200; redacted capacity response reports no usable capacity | Credential is authentic but presence is not sufficient for dispatch |
| Exact direct one-row production canary | HTTP 403 `quota_exceeded`; no model output; no browser/Site | Current route cannot generate a row; failure is before WARP output validation |

This is provider/account evidence, not task quality, exposure, admission, or
source-CI evidence. Do not silently switch provider/model when resuming Phase
1. A nonzero capacity result (or a newly authorized usable key) is necessary
but still insufficient: the unchanged direct and sandbox one-row production
canaries must pass before the 80 absent GitLab rows are dispatched.

The technical specification still calls the contract-bound Phase 1 backend
setting an **unpersisted** cache-identity input and tells status to warn the
operator to restore the original context ([spec lines 476–490](https://github.com/jasmineee-li/warp/blob/eac23227565c71ee3003eb2c1bc977fe04660dea/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md#L476-L490)).
That is the specification work required before any route-map source slice;
today's quota failure is a separate external condition.

## Output-contract testing and explicit non-goals

Unit tests may replace transport, but they must continue through WARP's
production extraction, compiler, validator, metadata, and cache owners. Do not
reimplement a provider in tests, add seven hand-authored response fixtures, or
ship a no-provider CLI that can pass while the real SDK/sandbox envelope is
incompatible. The real boundary remains two one-row production canaries outside
ordinary CI; a sanitized replay is justified only after a recurring real
boundary failure, and it must live beside the untouched production extractor.

This note does **not** propose a third shard, a shared `.venv`, caching
credentials/browser/generated state, wheel build artifacts, selective remote
routing, weaker `fail-fast`/cancellation, a workflow-level path filter, a
progress database, hooks that mutate state, a provider fallback, browser/reset
execution, or paper/publication changes. GitHub's cache rate limits and
retention changes, and uv's cache guidance, further argue for keeping any future
diagnostics bounded rather than uploading every run ([cache rate-limit
changelog](https://github.blog/changelog/2026-01-16-rate-limiting-for-actions-cache-entries/);
[retention changelog](https://github.blog/changelog/2026-08-27-actions-retention-will-cover-checks-workflow-runs-and-statuses/)).

Recommended order is:

1. add the read-only OpenRouter capacity step to the existing canary runbook
   now, and stop before a model call while remaining capacity is zero;
2. decide whether the per-card Phase 1 route map belongs in Run Definition or
   existing Phase 1 resume metadata, including pre-dispatch resolution and the
   Run15 missing-field policy; do not implement it while those choices remain
   open;
3. add the short canonical Taskgen command card;
4. treat CI summary output as an optional, separately measurable legibility
   slice; and
5. take no CI performance action until repeated hosted evidence identifies a
   critical-path change, with leaf parameterization remaining only a bounded
   candidate.

None of these changes is benchmark evidence. Each must retain the existing
aggregate, package proof, state isolation, exact cache attribution, and
provider/live evidence boundaries.
