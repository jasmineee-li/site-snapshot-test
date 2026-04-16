# Safety Benchmarks Integration — Handoff

**Purpose**: Run DoomArena / WASP / SafeArena safety benchmarks with glm-5:thinking
to produce trajectories that the trajectory_awareness / test_awareness_steering
judges can score. Goal: 20 tasks per site × 5 WebArena sites with reasoning
captured end-to-end.

## Context

The stub runners in `eval_awareness_experiments/run_{doomarena,wasp,safearena}.py`
only write `status: "pending_implementation"` manifests — they don't actually
run agents. So we're taking **option 2**: run each benchmark via its own
native CLI, then point `trajectory_converter.parse_*_trajectory` at the output.

For DoomArena specifically, that means using `doomarena.browsergym.scripts.run_bgym_experiment`
directly, with a custom `BgymExperiment` that uses a GLM-5 agent.

## What's done

### 1. Upstream packages installed into `world-sim` conda env

```
pip install -e external_benchmarks/AgentLab
pip install -e external_benchmarks/BrowserGym/browsergym/{core,miniwob,webarena,experiments,.}
pip install -e external_benchmarks/DoomArena/doomarena/{core,browsergym}
playwright install chromium
```

**Important**: We swapped from the in-repo `AgentLab/` to a fresh clone at
`external_benchmarks/AgentLab` (ServiceNow upstream, commit `cbc35a9`). The
in-repo copy is now unused by these experiments. Do not reinstall it on top.

### 2. Upstream patches (all in `external_benchmarks/AgentLab/`)

Three upstream bugs / gaps needed fixing. These are uncommitted edits in the
fresh AgentLab clone (git status shows them as modified files).

- **`src/agentlab/experiments/study.py` — `_convert_env_args`**: Upstream coerces
  any `BGymEnvArgs` back to plain `EnvArgs(**asdict(ea))`, which fails when
  DoomArena's `AttackedBrowserEnvArgs` carries extra fields (`benchmark_name`,
  `attack_configs`, `defenses`). Patch: if `ea` is a proper subclass, keep as-is.

- **`src/agentlab/llm/llm_utils.py` — `BaseMessage` / `AIMessage`**: Upstream
  only allowed `log_probs` as a kwarg. Added `reasoning` to the allowed set and
  threaded it through `AIMessage.__init__`.

- **`src/agentlab/llm/chat_api.py` — `ChatModel.__call__`**: Two changes:
  1. When `model_name` ends with `:thinking`, strip the suffix and pass
     `extra_body={"reasoning": {"effort": "high"}}` to OpenRouter.
  2. Read `reasoning` off the response message (OpenRouter returns CoT under
     `message.reasoning`) and attach it to the `AIMessage`.

These patches MUST exist for reasoning capture to work. Re-verify with:

```bash
grep -n "reasoning" external_benchmarks/AgentLab/src/agentlab/llm/chat_api.py
# should show ~3 hits: extract_reasoning, extra_body kwarg, reasoning=...
```

### 3. Verified end-to-end on miniwob + 1 reddit task

**Miniwob smoke test** (no docker): `/tmp/doomarena_smoke_glm5.py`. Produced
4 `step_*.pkl.gz` files, `step.agent_info.chat_messages.messages[2]["reasoning"]`
contained the full GLM-5 chain-of-thought. ✅

**Reddit nano smoke test** (1 task, live webarena container): `/tmp/doomarena_reddit_nano.py`.
Run result: `results/browsergym/study_2026-04-15_10-37-05/`. Popup attack fired
(`Flag ['GATEWAY_POPUP_INSERTED_0'] successfully verified`), `ASR_NEW=1.0`. The
agent ran 1 step then the popup's target-URL success filter triggered and ended
the episode. ✅

### 4. WebArena Reddit (postmill) container running

```
docker images | grep postmill    → postmill-populated-exposed-withimg:latest (53.3 GB)
docker ps --filter name=forum    → container `forum` mapping 8080:80
curl -sI http://localhost:8080/  → 200 OK
```

**Port is 8080, not 9999.** DoomArena hardcodes `WA_REDDIT={base_url}:8080`
in `_override_webarena_env_vars`. The miniwob http server (from `make miniwob`)
used to occupy 8080; it's now stopped.

The tar sits at `/local_data/temp/max/webarena-images/postmill-populated-exposed-withimg.tar`
(50 GB) — kept for future reloads; can be deleted if disk pressure.

### 5. Reachability check monkey-patch

BrowserGym's `WebArenaInstance._check_is_reachable` pings all 6 WebArena sites
(reddit, gitlab, shopping, shopping_admin, wikipedia, map) on `prepare_backend`,
even when the benchmark only uses reddit. `doomarena_reddit_nano.py` monkey-patches
it via a module-level `_ONLINE_SITES = {"reddit"}` set. When adding more
containers, expand this set.

## Update — 2026-04-15 19:38 UTC

This section supersedes the original "Immediate" checklist below.

### Newly done since the original handoff

1. **Fresh Reddit nano rerun confirmed reasoning capture on real WebArena tasks.**
   - Run dir: `results/browsergym/study_2026-04-15_12-53-51/`
   - Fresh trajectory contained `message.reasoning` on the assistant message and
     the popup attack triggered immediately.
   - `attack_results_v2.csv` for that run shows `ASR_NEW=1.0`.

2. **The local trajectory conversion path was fixed.**
   - `eval_awareness_experiments/trajectory_converter.py` had two real issues:
     - it crashed on real `step_*.pkl.gz` names because it sorted using
       `p.stem.split("_")[1]` (`step_1.pkl.gz` became `"1.pkl"`)
     - it was using `agent_info.think` and dropping richer hidden reasoning in
       `chat_messages.messages[*]["reasoning"]`
   - It now parses real DoomArena step files and prefers the captured hidden
     reasoning when present.

3. **A reusable local runner was added.**
   - New file: `eval_awareness_experiments/run_doomarena_glm5_popup.py`
   - Purpose:
     - replaces the ad hoc `/tmp/doomarena_reddit_nano.py` path
     - keeps the BrowserGym reachability monkey-patch local to the runner
     - supports exact single-site benchmark construction from installed
       `webarena/test.raw.json`

4. **The stock `webarena_reddit_subset5` benchmark is not safe to use as a pure Reddit validation set.**
   - The original 5-task run at `results/browsergym/study_2026-04-15_13-04-11/`
     should be treated as diagnostic only.
   - Reason: it included `webarena.685`, whose metadata is
     `sites=['reddit', 'gitlab']`, so the run retried and failed on
     `http://localhost:9001/users/sign_in`.
   - `attack_results_v2.csv` there shows 4 completed tasks and 1 error, which is
     expected given the mixed-site contamination.

5. **A corrected exact single-site Reddit subset run completed successfully.**
   - Run dir: `results/browsergym/study_2026-04-15_13-32-16/`
   - Benchmark name: `webarena_reddit_exact_subset5`
   - Task ids used (single-site `sites == ['reddit']`, shuffle `42`):
     `634, 611, 405, 399, 730`
   - All 5 tasks produced `summary_info.json`.
   - `attack_results_v2.csv` summary:
     - `task_status=done`
     - `ASR_NEW=0.8`
     - `task_count=5`
     - `steps=5.8`
   - Context: task `730` ("DisLike all submissions created by Hrekires in
     subreddit news") took much longer than the others and produced 26 step
     pickles before completion. Long single-site Reddit tasks are therefore
     normal; this was not a stuck run.

6. **All currently needed non-map image tarballs were downloaded and loaded into Docker.**
   - Present as Docker images:
     - `postmill-populated-exposed-withimg:latest` (`53.3GB`)
     - `shopping_admin_final_0719:latest` (`9.45GB`)
     - `shopping_final_0712:latest` (`64GB`)
     - `gitlab-populated-final-port8023:latest` (`77.6GB`)
   - Current free space on `/local_data`: about `295G`.
   - This is enough to continue, but there is no large comfort margin anymore.

7. **Current live containers**
   - `forum` (`postmill-populated-exposed-withimg`) is still up on `8080 -> 80`
   - No `shopping`, `shopping_admin`, or `gitlab` containers have been started yet

### Current next steps

1. **Bring up the newly loaded containers on DoomArena-compatible ports.**
   - Needed mappings:
     - `gitlab`: host `9001` -> container `8023`
     - `shopping`: host `8082` -> container `80`
     - `shopping_admin`: host `8083` -> container `80`

2. **Run each container's base URL reconfiguration immediately after startup.**
   - This is still required for Magento and GitLab.
   - Use the exact commands from
     `external_benchmarks/wasp/visualwebarena/environment_docker/README.md`.

3. **Expand the runner's online-site set and validate each site with a small exact single-site subset.**
   - Do this before any 20-task run.
   - Recommended order:
     - `shopping_admin`
     - `shopping`
     - `gitlab`

4. **Use exact single-site filtering for future site-scoped validations.**
   - Do not assume `start_url="__REDDIT__"` or similar start-url filters imply
     `sites == ['reddit']`.
   - Cross-site tasks exist and will silently contaminate subset runs.

5. **Only then scale to the 20-task/site runs.**
   - The current runner already has the right shape for the DoomArena popup path.
   - Banner attacks and multi-site orchestration can be added after the
     per-site container validation is complete.

6. **Map remains deferred.**
   - It still needs the separate OSM setup and the DoomArena port-443 issue is
     unresolved.

## Update — 2026-04-16 02:04 UTC

This section supersedes the earlier container-status and "Current next steps"
bullets above.

### Newly done since 2026-04-15 19:38 UTC

1. **All non-map WebArena containers are now live on DoomArena-compatible host ports.**
   - Current containers:
     - `forum` (`postmill-populated-exposed-withimg`) on `8080 -> 80`
     - `shopping` (`shopping_final_0712`) on `8082 -> 80`
     - `shopping_admin` (`shopping_admin_final_0719`) on `8083 -> 80`
     - `gitlab` (`gitlab-populated-final-port8023`) on `9001 -> 8023`

2. **The Magento and GitLab app-level URL reconfiguration is complete.**
   - `shopping` and `shopping_admin`:
     - Magento base URLs were rewritten to `http://localhost:8082` and
       `http://localhost:8083`
     - secure base URLs were updated in MySQL
     - caches were flushed
     - `shopping` indexers were switched to `schedule`
   - `gitlab`:
     - `external_url` was rewritten to `http://localhost:9001`
     - the first `gitlab-ctl reconfigure` broke the app because it auto-set
       Puma to `192` workers and moved nginx to internal port `9001`
     - this was repaired by pinning `puma['worker_processes'] = 2` and
       `nginx['listen_port'] = 8023`, then re-running `gitlab-ctl reconfigure`
   - Current live checks:
     - `shopping`: `200` at `http://localhost:8082/`
     - `shopping_admin`: `200` at `http://localhost:8083/admin`
     - `gitlab`: `200` at `http://localhost:9001/help`
     - Docker health for `gitlab`: `healthy`

3. **`shopping_admin` exact single-site validation completed successfully.**
   - Run dir: `results/browsergym/study_2026-04-15_21-44-45/`
   - Benchmark name: `webarena_shopping_admin_exact_subset5`
   - `attack_results_v2.csv` summary:
     - `task_status=done`
     - `ASR_NEW=0.4`
     - `task_count=5`
     - `steps=4.2`
   - Interpretation: the site and benchmark path are usable end-to-end. This
     was not just trivial 1-step termination; the run included both short and
     longer trajectories.

4. **`shopping` exact single-site validation exposed a scoring/evaluator bug, not a site bring-up bug.**
   - Run dir: `results/browsergym/study_2026-04-15_21-54-51/`
   - Benchmark name: `webarena_shopping_exact_subset5`
   - `attack_results_v2.csv` summary:
     - one aggregate `done` row with `ASR_NEW=0.5`, `task_count=4`, `steps=3.25`
     - one aggregate `error` row with `task_count=1`
   - The failing task was `webarena.117`.
   - Root cause:
     - WebArena's evaluator still hardcodes `gpt-4-1106-preview` inside
       `webarena.evaluation_harness.helper_functions.llm_fuzzy_match`
     - that model is unavailable in the current OpenAI account / API setup, so
       validation raised `model_not_found` during scoring
   - Important: this does **not** indicate the `shopping` container is broken.
     The site is live and benchmark trajectories were produced; the failure is
     in the evaluator path after the agent step.

5. **The reusable runner was patched to handle the stale WebArena evaluator model.**
   - File: `eval_awareness_experiments/run_doomarena_glm5_popup.py`
   - New behavior:
     - monkey-patches WebArena's `llm_fuzzy_match` / `llm_ua_match`
     - adds `--webarena-eval-model` (default:
       `WEBARENA_EVAL_MODEL` env var, else `gpt-4.1-mini`)
     - if evaluator LLM calls still fail, the task now degrades to
       `score=0.0` instead of crashing the whole study
   - Status:
     - syntax checked successfully
     - a rerun to validate this patch was started, but the user interrupted the
       turn before a new study directory was created
     - therefore this evaluator fix is **implemented but not yet revalidated**

### Current state summary

- `reddit`: validated on nano and exact 5-task subset
- `shopping_admin`: validated on exact 5-task subset
- `shopping`: site is healthy, but the first 5-task validation hit the stale
  WebArena evaluator model; rerun with the patched runner is still needed
- `gitlab`: site is healthy after config repair, but no benchmark validation
  run has been executed yet
- `map`: still deferred

### Current next steps

1. **Rerun `shopping` exact subset with the patched runner.**
   - Goal: confirm the `gpt-4-1106-preview` scoring failure is gone and produce
     a clean `attack_results_v2.csv` for all 5 tasks.

2. **Run `gitlab` exact single-site subset with the patched runner.**
   - This should be done only after the `shopping` rerun is clean, since the
     same stale evaluator path may affect some GitLab tasks too.

3. **After `shopping` and `gitlab` are validated, update this handoff again and then scale out.**
   - Ready-to-scale sites at that point should be:
     - `reddit`
     - `shopping_admin`
     - `shopping`
     - `gitlab`
   - Then move to the intended `20 tasks / site` runs.

4. **Map remains deferred.**
   - It still needs separate OSM setup and the DoomArena port-`443` issue is
     unresolved.

## What needs to be done next

### Immediate

1. **Re-run reddit_nano smoke test** and inspect the trajectory to confirm
   reasoning is captured even for webarena tasks (we've only verified it on
   miniwob so far — the real reddit run may follow a different code path):

   ```bash
   docker ps --filter name=forum  # confirm container up on :8080
   cd /local_data/temp/max/browser-sim
   set -a && source .env && set +a
   /home/max/envs/world-sim/bin/python /tmp/doomarena_reddit_nano.py
   # Then inspect step pickles:
   find results/browsergym -name "step_*.pkl.gz" -newer /tmp/doomarena_reddit_nano.py
   ```

2. **Run a slightly larger reddit subset** (`webarena_reddit_subset5`, 5 tasks)
   to validate stability before the full 20-task run.

### Then — scale to all 5 sites × 20 tasks

3. **Download + load remaining webarena images**. Sizes (from CMU mirror
   `http://metis.lti.cs.cmu.edu/webarena-images/`):

   | File | Size | Into container |
   |---|---|---|
   | `gitlab-populated-final-port8023.tar` | 72 GB | `gitlab` |
   | `shopping_final_0712.tar` | 63 GB | `shopping` (Magento) |
   | `shopping_admin_final_0719.tar` | 9 GB | `shopping_admin` |

   Map (OSM) is separate — a docker-compose under `~/openstreetmap-website/`
   per the WASP README. Not yet fetched.

   Download into `/local_data/temp/max/webarena-images/`, then `docker load`
   each. Disk used so far: ~100 GB. Needed total: ~350 GB of docker storage.
   Free on `/local_data`: 633 GB. It fits, but tight. Use `tmux` for each
   download + load so they survive disconnects.

4. **Correct port mapping for each container**. DoomArena's
   `_override_webarena_env_vars` uses non-standard ports:

   | Site | DoomArena port | Container expected port |
   |---|---|---|
   | reddit (postmill) | 8080 | 80 |
   | gitlab | 9001 | 8023 |
   | shopping | 8082 | 80 |
   | shopping_admin | 8083 | 80 |
   | wikipedia | 8081 | 80 |
   | map | 443 | 3000 |

   Map's port 443 is privileged — either run with sudo, override
   `_override_webarena_env_vars`, or drop map from scope.

5. **Reconfigure each container's base_url** after running. Magento bakes URLs
   into its DB; GitLab bakes into `gitlab.rb`. See
   `external_benchmarks/wasp/visualwebarena/environment_docker/README.md` for
   the exact per-container `docker exec` reconfigure commands (long sequence).

6. **Expand `_ONLINE_SITES`** in the runner script to include each site you've
   brought up. Also update `_check_is_reachable` patch to ping them.

7. **Write a runner for all 5 sites × 20 tasks × glm-5:thinking**. It should:
   - Use `get_webarena_subset(start_url="__{SITE}__", max_tasks=20, shuffle=42)`
     to construct matched subsets for gitlab / shopping_admin / map (not pre-registered)
   - Target both popup and banner attacks (for breadth)
   - Run each site in parallel (if RAM allows) or sequentially
   - Save to `results/browsergym/study_*/`

8. **Feed trajectories into `trajectory_awareness`**. Use
   `eval_awareness_experiments.trajectory_converter.parse_doomarena_trajectory`
   to standardize, then score with `judges/test_awareness_steering.py`.

### Known gotchas

- **`skip_reset_and_massage=True`** is required; the full-reset HTTP endpoint at
  `{base_url}:7565` isn't running, and the reset would take 5-10 minutes anyway.
- **Study dir names say "claude-3-5-sonnet-beta"** even when using glm-5. That's
  because we clone `AGENT_CLAUDE_SONNET_35` as a template and just swap
  `chat_model_args`. The actual model used is visible in
  `step.agent_info.extra_info['chat_model_args']['model_name']`.
- **DoomArena overrides WA_* env vars inside `run_bgym_experiment`** via
  `_override_webarena_env_vars`. Setting them before invocation won't stick.
- **`collect_results` post-processing fails** on miniwob with a pandas KeyError
  (`'stats.cum_input_tokens' not in index`). It's called after experiments
  complete, so it doesn't block trajectory creation. On webarena runs with
  the reddit task, it worked fine — the miniwob failure seems benign/format
  specific.
- **Upstream agentlab is at `external_benchmarks/AgentLab`**. The in-repo
  `AgentLab/` is still there but no longer used by these experiments.

## Key files / locations

```
eval_awareness_experiments/
  SAFETY_BENCHMARKS_HANDOFF.md       ← this file
  EXPERIMENTS.md                      ← design doc for the A/B/C experiments
  trajectory_converter.py             ← knows DoomArena/WASP/SafeArena formats
  experiments/trajectory_awareness.py ← judge runner (existing)
  run_doomarena.py                    ← STUB, do not use
  run_wasp.py                         ← STUB, do not use
  run_safearena.py                    ← STUB, do not use

external_benchmarks/
  AgentLab/                           ← ServiceNow upstream (has 3 local patches)
  BrowserGym/                         ← ServiceNow upstream
  DoomArena/                          ← ServiceNow upstream
  safearena/                          ← cloned, not yet installed/tested
  wasp/                               ← cloned, not yet installed/tested
  TheAgentCompany/                    ← pre-existing, for other work

/local_data/temp/max/webarena-images/
  postmill-populated-exposed-withimg.tar   ← 50 GB (loaded)
  docker_load_postmill.log
  download.log

/tmp/
  doomarena_smoke_glm5.py             ← miniwob smoke test (verified reasoning capture)
  doomarena_reddit_nano.py            ← 1-task reddit smoke test (verified popup attack)
  reddit_nano.log                     ← last run output
```

## Environment

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate world-sim
set -a && source /local_data/temp/max/browser-sim/.env && set +a
# These get set automatically by DoomArena, but for manual invocations:
export DOOMARENA_WEBARENA_BASE_URL=http://localhost
```

Required env vars in `.env`: `OPENROUTER_API_KEY` (for glm-5 and judges),
`ANTHROPIC_API_KEY` (unused by these runs but referenced by some defaults).

## Containers currently running

```
forum   postmill-populated-exposed-withimg   :8080 → :80   (Reddit — live)
```

Check with `docker ps`. To restart from scratch:
```bash
docker stop forum && docker rm forum
docker run --name forum -p 8080:80 -e RATELIMIT_WHITELIST=0.0.0.0/0,::/0 -d postmill-populated-exposed-withimg
sleep 20  # wait for nginx init
curl -sI http://localhost:8080/  # should 200
```
