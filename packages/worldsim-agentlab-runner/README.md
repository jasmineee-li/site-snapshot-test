# WorldSim AgentLab Runner

Isolated sidecar for AgentLab/BrowserGym comparison runs and the experimental
WorldSim-v5 Phase 4 AgentLab runner.

This package intentionally has its own dependency graph. Root WorldSim uses
Browser Use 0.12.6, which requires `openai==2.16.0`; current AgentLab requires
`openai<2`. Keeping AgentLab here avoids resolver conflicts in the orchestrator
environment.

Run contract:

```bash
uv run --project packages/worldsim-agentlab-runner \
  worldsim-agentlab-runner run /path/to/agentlab_request.json
```

The command prints the canonical result as the final JSON object on stdout.
Upstream BrowserGym/AgentLab may print informational lines before it. The
parent WorldSim process parses the final JSON object and owns conversion into
canonical `result.json`.

WorldSim-v5 Phase 4 uses the sibling command:

```bash
uv run --project packages/worldsim-agentlab-runner \
  worldsim-agentlab-runner phase4-run /path/to/agentlab_phase4_request.json
```

That path runs admitted WorldSim tasks through AgentLab/BrowserGym while root
WorldSim keeps ownership of seeding, rewards, PVPO gates, TP/VEA,
placement-fix, variants, resume, and summaries.

Parent CLI:

```bash
uv run python -m worldsim.main agentlab run \
  --instances instances.scale.json \
  --browsergym-task-name webarena_verified.42 \
  --site gitlab \
  --agent-model gpt52 \
  --agent-provider openrouter
```

Named model profiles can be listed with:

```bash
uv run python -m worldsim.main agentlab models
```

Current named OpenRouter profiles are `opus47`, `sonnet46`, `gemini25pro`,
`kimik25`, `gpt52`, and `glm5`. Direct native routes remain available for
compatible providers, for example `--agent-provider openai --agent-model gpt-5.2`.
Named OpenRouter profiles pin the intended provider and disable OpenRouter
fallbacks where possible; `gpt52` also disables/excludes OpenRouter reasoning
to match the Browser Use OpenRouter parity arm.

There are two separate modes. `run` is the benchmark-native comparison runner.
`phase4-run` is the only AgentLab path that may be compared as WorldSim-v5
Phase 4 data. It must preserve Browser Use-equivalent auth, PVPO, network,
history, final-response, screenshot, and resume artifacts before a live sweep is
treated as parity data. Do not describe native `run` outputs as AgentLab
Phase 4 parity results.

Current proof boundary:

- Local tests cover request controls, `service_workers="block"` context kwargs,
  AgentLab-native action projection, Needham XML byte equivalence, timeout/error
  placeholder artifacts, and resume completeness.
- r5 live proof on 2026-05-07 completed one Reddit AgentLab `phase4-run` task
  with PVPO, Needham, HAR/network evidence, private reward traces, and a
  successful `result.json`.
- A later r5 rerun proved parent whole-task timeout recovery and browser
  recycle, but timed out before final sidecar runtime was written. The next
  successful live proof must show
  `browser_runtime.browsergym_context_kwargs.service_workers == "block"`.

Phase 4 request controls intentionally use scoped interception rather than
global Playwright headers. Storage-state auth is normalized and host-validated
by root WorldSim before the sidecar sees it; origin rewrites are same-scheme and
repair URL-bearing headers before adding scoped auth. The sidecar blocks service
workers so Playwright routing can observe first-party requests.

`network_trace.json` remains WorldSim's flat evaluator trace. `network.har` is a
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
`pvpo_cdp_url` or `connect_over_cdp` browser. Legacy CDP attachment is available
only for explicit debugging with `WORLDSIM_AGENTLAB_LEGACY_CONNECT_OVER_CDP=1`.

AgentLab `phase4-run` accepts the whole-task subprocess timeout, LLM-call
timeout, and step timeout. The sidecar forwards `llm_timeout` to its model
transport and wraps setup/action/browser steps with `step_timeout`.
`--agent-task-timeout` remains the outer infrastructure guard for startup, CDP,
and cleanup hangs, and it is still the deadline to rely on for remote-job
failure recovery. r5 live proof on 2026-05-07 showed a stalled AgentLab step
could still run past `--agent-step-timeout 60` and was recovered only by
`--agent-task-timeout 900`, so report per-step deadline parity only after a
passing live proof.
