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

The command prints one JSON object to stdout. The parent WorldSim process owns
conversion into canonical `result.json`.

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

AgentLab actions stay AgentLab-native. The Phase 4 exporter may parse
AgentLab's Python-call action strings into ordered tool-call records for
Needham XML, TP/VEA, and outcome-taxonomy consumers, but it must preserve the
raw AgentLab text and must not relabel those actions as Browser Use actions.

AgentLab `phase4-run` parity should rely only on the whole-task subprocess
timeout today. The sidecar accepts `llm_timeout` and `step_timeout` request
fields for experimental deadline handling, but r5 live proof on 2026-05-07
showed a stalled AgentLab step could run past `--agent-step-timeout 60` and was
recovered only by `--agent-task-timeout 900`. Do not claim per-step deadline
parity until a passing live proof exists.
