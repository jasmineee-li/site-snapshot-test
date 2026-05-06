# AgentLab Harness Support Plan

Status: comparison-runner slice exists; experimental WorldSim-v5 Phase 4 runner
exists behind the isolated sidecar. Treat new AgentLab Phase 4 sweeps as parity
data only after the live artifact/PVPO/network/auth/resume gate passes on the
target host/model matrix.

Upstream inspected at `vendors/AgentLab-upstream` after cloning
`https://github.com/ServiceNow/AgentLab.git`. The current public execution API
is `agentlab.experiments.loop.EnvArgs` plus synchronous `ExpArgs.run()`.
AgentLab writes `summary_info.json` and per-step pickle/screenshot artifacts for
comparison runs. The WorldSim-v5 Phase 4 sidecar additionally writes
WorldSim-compatible history, final response, network, HAR, PVPO, screenshot, and
browser-runtime artifacts.

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

## Parity Work Status

1. **Runner-neutral request fields**
   - Phase 4 now serializes the Browser Use runtime fields that AgentLab needs:
     start URLs, site prompt, benchmark root, task site, payload text/witnesses,
     PVPO CDP URL, instance id, and origin rewrites.

2. **Auth adapter**
   - Phase 4 resolves `storage_state` before crossing the sidecar boundary and
     sends scoped `http_basic` / `http_headers` controls for request
     interception. Do not reintroduce global Playwright `extra_http_headers`;
     it leaks secrets across origins and misses first-navigation races.
   - Storage state is host-validated, normalized, copied into the task
     directory, and augmented for same-site origin aliases before the sidecar
     starts. This mirrors Browser Use's timing fix: Chromium chooses
     cookies/localStorage before request rewriting can redirect a canonical
     absolute URL back to the bound replica.

3. **Browser instrumentation**
   - AgentLab captures PVPO beginFrame screenshots and network events in the
     sidecar. Browser Use still owns the mature async implementation; live
     parity gates should compare artifact manifests and PVPO capture summaries.

4. **Trajectory exporter**
   - AgentLab exports `history.json`, `final_response.json`, `needham_trace.*`,
     and canonical `screenshots/step_N.png`.

5. **Network trace parity**
   - AgentLab returns in-memory `network_trace` and persists redacted
     `network_trace.json`, `navigation_trace.json`, and `network.har`. Keep
     reward fixtures covering query params, POST bodies, response headers, and
     cookies before claiming parity for action-reward sweeps.
   - `network_trace` must stay flat so root WorldSim can convert it through the
     same HAR adapter used by Browser Use. `network.har` is now a HAR 1.2-shaped
     audit artifact rather than the evaluator input source of truth.

6. **PVPO parity**
   - AgentLab accepts `payload_text`, `payload_witnesses`, and `pvpo_cdp_url`,
     and writes `pvpo/capture_summary.json`, per-step PVPO JSON, and paired
     screenshots. `max_coverage == 0` must still mean non-encounter, not missing
     instrumentation.

7. **Resume and provenance**
   - Phase 4 includes runner name in fingerprints and keeps `result.json` as
     the completion sentinel. Live gates should prove resume skips completed
     AgentLab task dirs and reruns incomplete dirs.
   - Whole-task subprocess timeout writes minimal error artifacts and attempts
     PVPO browser recycle from the parent. `--agent-llm-timeout` and
     `--agent-step-timeout` currently fail fast for AgentLab because they are not
     wired into AgentLab model/step execution.

## Suggested Next PRs

1. Browser Use behavior-preserving adapter extraction:
   `worldsim.browser_use` package + `AgentRunRequest` plumbing.
2. Instrumentation extraction:
   PVPO/network/screenshot lifecycle modules independent of Browser Use.
3. AgentLab live parity harness:
   fake sidecar composition test plus localhost auth/rewrite/network fixture.
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
