# WARP Taskgen AgentLab Runner

Isolated sidecar for AgentLab/BrowserGym comparison runs and the experimental
WARP Taskgen Phase 4 AgentLab runner. The primary sidecar distribution and
console script are `warp-taskgen-agentlab-runner`; the legacy
`worldsim-agentlab-runner` entrypoint remains as a compatibility alias for
existing run artifacts and operator scripts.

This package intentionally has its own dependency graph. Root WARP Taskgen uses
Browser Use 0.12.6, which requires `openai==2.16.0`; current AgentLab requires
`openai<2`. Keeping AgentLab here avoids resolver conflicts in the orchestrator
environment.

Run contract:

```bash
uv run --project packages/worldsim-agentlab-runner \
  warp-taskgen-agentlab-runner run /path/to/agentlab_request.json
```

The command prints the native result as the final JSON object on stdout.
Upstream BrowserGym/AgentLab may print informational lines before it. The
parent WARP Taskgen process parses the final JSON object. Benchmarks with the
`comparison_ingestion` capability write a separate `comparison_result.json`;
the historical WebArena comparison path retains its `result.json` sentinel for
one compatibility cycle. The legacy `worldsim-agentlab-runner` entrypoint is
kept as a compatibility alias.

WARP Taskgen Phase 4 uses the sibling command:

```bash
uv run --project packages/worldsim-agentlab-runner \
  warp-taskgen-agentlab-runner phase4-run /path/to/agentlab_phase4_request.json
```

That path runs admitted WARP Taskgen tasks through AgentLab/BrowserGym while root
WARP Taskgen keeps ownership of seeding, rewards, PVPO gates, TP/VEA,
placement-fix, variants, resume, and summaries.

Parent CLI:

```bash
uv run warp-taskgen agentlab run \
  --instances instances.scale.json \
  --browsergym-task-name webarena_verified.42 \
  --site gitlab \
  --agent-model gpt52 \
  --agent-provider openrouter
```

Named model profiles can be listed with:

```bash
uv run warp-taskgen agentlab models
```

Current named OpenRouter profiles are `opus47`, `sonnet46`, `gemini25pro`,
`kimik25`, `gpt52`, and `glm5`. Direct native routes remain available for
compatible providers, for example `--agent-provider openai --agent-model gpt-5.2`.
Named OpenRouter profiles pin the intended provider and disable OpenRouter
fallbacks where possible; `gpt52` also disables/excludes OpenRouter reasoning
to match the Browser Use OpenRouter parity arm.

There are two separate modes. `run` is the benchmark-native comparison runner.
`phase4-run` is the only AgentLab path that may be compared as WARP Taskgen
Phase 4 data. It must preserve Browser Use-equivalent auth, PVPO, network,
history, final-response, screenshot, and resume artifacts before a live sweep is
treated as parity data. Do not describe native `run` outputs as AgentLab
Phase 4 parity results.

Current proof boundary:

- Local tests cover request controls, `service_workers="block"` context kwargs,
  AgentLab-native action projection, Needham XML byte equivalence, timeout/error
  placeholder artifacts, AgentLab browser-step timeout retry audits, and resume
  completeness.
- r5 live proof on 2026-05-07 completed one Reddit AgentLab `phase4-run` task
  with PVPO, Needham, HAR/network evidence, private reward traces, and a
  successful `result.json`.
- r8a live proof on 2026-05-08/09 moved the Phase 4 path to native BrowserGym
  launch plus page-surface-stable PVPO at W48. Treat new host/model matrices as
  parity data only after their own artifact, PVPO, network, auth, timeout/retry,
  and resume gates pass.

Phase 4 request controls intentionally use scoped interception rather than
global Playwright headers. Storage-state auth is normalized and host-validated
by root WARP Taskgen before the sidecar sees it; origin rewrites are same-scheme and
repair URL-bearing headers before adding scoped auth. The sidecar blocks service
workers so Playwright routing can observe first-party requests.

`network_trace.json` remains WARP Taskgen's flat evaluator trace. `network.har` is a
standards-shaped HAR 1.2 artifact with ISO timestamps, request/response
metadata, `cache`, and `timings` fields for audit tooling. Persisted request
copies and network artifacts redact auth headers, cookies, and token-like
headers.

`agentlab_step_timeline.jsonl` carries compact per-step network deltas for
operator debugging. Delta fields use recorder row indexes and mean "request
rows initiated since the previous timeline mark"; response and failure fields
may enrich those rows later. The timeline keeps counts, methods, statuses, and
redacted latest URLs only. Full request/response evidence stays in
`network_trace.json`, `network.har`, and private reward traces.

AgentLab actions stay AgentLab-native. The Phase 4 exporter may parse
AgentLab's Python-call action strings into ordered tool-call records for
Needham XML, TP/VEA, and outcome-taxonomy consumers, but it must preserve the
raw AgentLab text and must not relabel those actions as Browser Use actions.

By default, AgentLab Phase 4 uses BrowserGym's native browser launch. The
page-surface-stable PVPO recorder observes that runner-owned page with a scoped
CDP session at capture time; it does not route BrowserGym through a dedicated
browser endpoint. Use native BrowserGym launch for Phase 4 runs.

AgentLab `phase4-run` accepts the whole-task subprocess timeout, LLM-call
timeout, and step timeout. The sidecar forwards `llm_timeout` to its model
transport and wraps setup/action/browser steps with `step_timeout`. It also
sets BrowserGym's page and context default action/navigation deadlines so a
single stuck Playwright action does not consume the full task timeout. Defaults
are 60s for actions and 45s for navigations; override with
`WORLDSIM_AGENTLAB_ACTION_TIMEOUT_S` and
`WORLDSIM_AGENTLAB_NAVIGATION_TIMEOUT_S` only for debugging. Browser steps are
also capped independently at 120s by default via
`WORLDSIM_AGENTLAB_BROWSER_STEP_TIMEOUT_S`, while setup and LLM action calls can
still use the larger `--agent-step-timeout`.
`--agent-task-timeout` remains the outer infrastructure guard for startup and
cleanup hangs.

At high worker fan-out, `phase4-run` serializes only the short BrowserGym
benchmark metadata construction step behind a host-local file lock. This avoids
transient empty metadata reads while keeping browser execution parallel.

When a BrowserGym action step hits the safe browser-step timeout before any
mutating network evidence, root Phase 4 may retry the AgentLab task and write
`agentlab_infra_retries.json` in the trace directory. The default retry count is
controlled by `WORLDSIM_AGENTLAB_BROWSER_STEP_TIMEOUT_RETRIES`. Timeouts after
mutating evidence are not safe to replay and should be treated as task/runtime
failures until manually inspected.
