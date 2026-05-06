# AgentLab Harness Support Plan

Status: comparison-runner slice exists; WorldSim-v5 Phase 4 parity does not.

Upstream inspected at `vendors/AgentLab-upstream` after cloning
`https://github.com/ServiceNow/AgentLab.git`. The current public execution API
is `agentlab.experiments.loop.EnvArgs` plus synchronous `ExpArgs.run()`.
AgentLab writes `summary_info.json` and per-step pickle/screenshot artifacts.
That is enough for comparison-runner reporting, but it is not enough for
WorldSim-v5 Phase 4 parity.

## Current Slice

- `worldsim/agent_runtime.py` defines runner-neutral `AgentResult`,
  `AgentRunner`, and future `AgentRunRequest`.
- `worldsim/runners/` registers `browser_use` and `agentlab`.
- `worldsim/runners/agentlab.py` builds a JSON request and invokes the isolated
  `packages/worldsim-agentlab-runner` sidecar. Root WorldSim does not import
  AgentLab.
- `packages/worldsim-agentlab-runner` owns AgentLab/BrowserGym imports and
  writes/returns native AgentLab `summary_info.json` data for comparison-runner
  use.
- `worldsim agentlab run` runs one AgentLab/BrowserGym comparison task through
  the sidecar. It defaults to `--agent-model gpt52 --agent-provider openrouter`
  and accepts either `--task-json` or `--browsergym-task-name`.
- `worldsim agentlab models` lists the named AgentLab comparison model profiles:
  `opus47`, `sonnet46`, `gemini25pro`, `kimik25`, `gpt52`, and `glm5`.
- The sidecar uses a WorldSim-owned AgentLab chat-model adapter instead of
  AgentLab's built-in LiteLLM chat wrapper so OpenRouter routing controls,
  temperature omission, output-token budgets, and response metadata are
  explicit in the request.
- Named OpenRouter profiles pin provider slugs and disable OpenRouter fallbacks
  where the target provider exists. GPT-5.2 also forwards
  `reasoning={"effort": "none", "exclude": true}` to match the Browser Use
  OpenRouter parity arm.
- `--attack-mode seeded_comparison` applies WorldSim task seeds before the
  BrowserGym-native AgentLab run. It remains a comparison mode, separate from
  the Phase 4 runner path.
- `--runner` is plumbed into CLI state and Phase 4 resume fingerprints.
- Phase 4 now accepts `--runner agentlab` when the isolated sidecar package and
  vendored AgentLab checkout are present. Root WorldSim serializes the task to
  `worldsim-agentlab-runner phase4-run`; the sidecar owns AgentLab/BrowserGym
  imports and writes WorldSim-compatible PVPO, network, history, and final
  response artifacts.
- AgentLab is not declared as a root `pyproject.toml` extra because current
  AgentLab releases require `openai<2`, while Browser Use 0.12.6 requires
  `openai==2.16.0`. The sidecar package resolves AgentLab from
  `vendors/AgentLab-upstream` with its own environment.

## Architecture Decision

There are two separate products:

1. **AgentLab comparison runner.** Runs benchmark-native BrowserGym tasks and
   reports AgentLab/BrowserGym reward from `summary_info.json`. This is useful
   for WASP/STWebAgentBench/DoomArena-style comparisons.
2. **AgentLab WorldSim-v5 runtime.** Runs the same admitted WorldSim tasks as
   Browser Use and must preserve WorldSim's PVPO, auth, reward, network, and
   artifact contracts. This now exists as an experimental Phase 4 sidecar path;
   rigor sweeps should still validate PVPO/CDP, network reward, and resume
   contracts before treating a new host/model matrix as parity data.

Do not collapse these paths. Comparison reward is not WorldSim-v5 refusal ASR.

## Parity Work Required

1. **Runner-neutral request object**
   - Replace `run_adversarial_task`'s Browser Use kwargs with
     `AgentRunRequest`.
   - Keep Browser Use adapter behavior byte-for-byte equivalent first.

2. **Auth adapter**
   - Reproduce `storage_state`, origin-scoped `http_basic`, and origin-scoped
     `http_headers`.
   - Do not map headers to global Playwright extra headers; that leaks secrets
     across origins and misses first navigation races.

3. **Browser instrumentation**
   - Extract PVPO endpoint leasing/recycling, beginFrame screenshots, scroll
     and navigation ticks, and network capture out of `browser_use_agent.py`.
   - AgentLab/BrowserGym must call the same instrumentation at step boundaries.

4. **Trajectory exporter**
   - Either export BrowserGym steps to current `history.json` shape or make
     Phase 4 judges read a runner-neutral Needham trace first.
   - Preserve screenshot alignment with `screenshots/step_N.png`.

5. **Network trace parity**
   - Return evaluator-grade in-memory `network_trace` in WorldSim's flat schema.
   - Persist redacted `network_trace.json`, `navigation_trace.json`, and
     `network.har`.
   - BrowserGym native artifacts are not sufficient for action rewards.

6. **PVPO parity**
   - Accept `payload_text`, `payload_witnesses`, and `pvpo_cdp_url`.
   - Write `pvpo/capture_summary.json`, per-step PVPO JSON, and paired
     screenshots.
   - `max_coverage == 0` must mean non-encounter, not missing instrumentation.

7. **Resume and provenance**
   - Include runner name, runner version, BrowserGym task name, AgentLab
     version, and adapter version in fingerprints.
   - Keep `result.json` as the completion sentinel.

## Suggested Next PRs

1. Browser Use behavior-preserving adapter extraction:
   `worldsim.browser_use` package + `AgentRunRequest` plumbing.
2. Instrumentation extraction:
   PVPO/network/screenshot lifecycle modules independent of Browser Use.
3. AgentLab WorldSim-v5 prototype:
   BrowserGym task wrapper plus instrumentation hooks on a tiny local fixture.
4. Live gate:
   r5 Phase 4 smoke with GitLab/Reddit, one task per site, comparing Browser Use
   and AgentLab artifact manifests before enabling full sweeps.

## Validation Gates

Unit gates:

```bash
uv run pytest tests/test_agentlab_runner.py tests/test_agent_config.py tests/test_eval_worker_pool.py -q
uv run pytest tests/phase_4/test_resume_1.py tests/phase_4/test_resume_2.py tests/phase_4/test_resume_3.py -q
bash scripts/verify_fast.sh
```

Live gate before claiming WorldSim-v5 parity:

```bash
scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet
```
