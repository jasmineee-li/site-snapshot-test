# Phase 4 Expanded Model Sweep, 2026-05-01

## Purpose

This sweep compares browser-agent model behavior over one frozen admitted cohort:

- Source run: `logs/task_bank_live_verify_phase0_path_repair_20260430T1900Z`
- Task filter: `--task-origin new_task`
- Cap: `--max-tasks-per-site 16`
- Sites: `gitlab,reddit`
- Instance file: `instances.scale.json`
- Sandbox judge: `claude-sonnet-4-6`
- Browser-use deadlines: `--agent-llm-timeout 240 --agent-step-timeout 300`

The sweep only varies the Phase 4 web-agent provider/model/tier. It must not
regenerate tasks, mutate rewards, alter Phase 2c admission, change strategy
catalogs, or change Phase 4 encounter science mid-sweep.

## Runner

Config:

```bash
configs/phase4_model_sweeps/phase4_20260501_expanded.json
```

Dry run:

```bash
uv run python scripts/run_phase_4_model_sweep.py \
  --config configs/phase4_model_sweeps/phase4_20260501_expanded.json \
  --dry-run
```

Live run:

```bash
uv run python scripts/run_phase_4_model_sweep.py \
  --config configs/phase4_model_sweeps/phase4_20260501_expanded.json
```

The runner syncs once, launches one registered r5 job at a time through
`scripts/remote_job_start.sh`, checks with `scripts/remote_job_status.sh`, and
tails only when logs become stale. It writes compaction-safe local state under:

```text
logs/phase4_model_sweep_<timestamp>Z/
```

State files:

- `sweep_state.json`
- `handoff.md`

Raw Phase 4 artifacts remain under `logs/` and must not be committed.

## Model Conditions

Completed condition, included for summary provenance:

- `gpt52`: `--agent-provider openai --agent-model gpt-5.2 --agent-service-tier priority`

Sequential remaining conditions:

- `sonnet46`: `--agent-provider anthropic --agent-model claude-sonnet-4-6`
- `opus47`: `--agent-provider anthropic --agent-model claude-opus-4-7`
- `glm5`: `--agent-provider openrouter --agent-model z-ai/glm-5`
- `gemini25pro`: `--agent-provider openrouter --agent-model google/gemini-2.5-pro`
- `kimi-k25`: `--agent-provider openrouter --agent-model moonshotai/kimi-k2.5`
- `minimax-m27`: `--agent-provider openrouter --agent-model minimax/minimax-m2.7`

## Failure Discipline

Stop the sweep on first unresolved failure. Preserve artifacts and diagnose
before rerunning.

Reruns are allowed only for diagnosed infrastructure, auth, quota, topology,
provider, model-slug, stale-log, or evaluator failures. Do not rerun legitimate
model behavior such as compliance, resistance, task-broke outcomes, or low ASR.

Retry budgets recorded in config:

- GPT, Sonnet, Opus: 5 per model family
- GLM, Gemini, Kimi, MiniMax: 3 per model

The runner records these budgets but does not automatically consume them. That
keeps reruns diagnosis-gated instead of hiding failure modes inside an outer
retry loop.

## Kimi Reasoning

The primary Kimi condition uses `moonshotai/kimi-k2.5` without an explicit
OpenRouter `reasoning` override. Browser Use's current OpenRouter adapter does
not preserve `reasoning_details` across multi-turn tool calls, so enabling
explicit reasoning would create an adapter confound. A separate controlled
condition can test explicit Kimi reasoning after the adapter records and replays
reasoning details in the agent loop.
