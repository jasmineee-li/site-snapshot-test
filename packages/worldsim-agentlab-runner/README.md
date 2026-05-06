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
`phase4-run` is the WorldSim-v5 runtime and must preserve Browser Use-equivalent
auth, PVPO, network, history, final-response, screenshot, and resume artifacts
before a live sweep is treated as parity data.
