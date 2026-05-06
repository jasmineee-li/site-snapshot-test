# 2026-04-26 — Setup notes

Things I had to do / learned getting InjecAgent wired into the
eval-awareness pipeline. Not part of the experiment_log; this is
scaffolding-level, for the next agent who has to set this up
elsewhere or extend it.

## InjecAgent

### What I tried first (wrong path)

- `git clone https://github.com/uiuc-kang-lab/InjecAgent` into
  `external_benchmarks/InjecAgent/` (uppercase). The project's runner
  hard-codes a lowercase default path, so this didn't work.
- Created a dedicated venv inside `external_benchmarks/InjecAgent/.venv`
  with `uv venv` + `uv pip install -r requirements.txt`.
- Found two deps the upstream `requirements.txt` is missing that
  `src/` imports unconditionally:
  - `together` (`src/models.py:63`) — even `--help` fails on import
    without it.
  - `nltk` (`src/output_parsing.py:2`).
- Added both, got upstream `evaluate_prompted_agent.py --help` working.

All of this turned out to be unnecessary.

### What's actually right

- Run `bash eval_awareness_experiments/setup_benchmarks.sh injecagent`.
- That clones to lowercase `external_benchmarks/injecagent/` (the
  path the runner expects) and intentionally **does not install
  InjecAgent's Python**. The script's comment is the canonical
  statement: "InjecAgent ships a dataset + evaluator scripts; no
  package install."
- `eval_awareness_experiments/run_injecagent.py` consumes only the
  JSON dataset (`data/tools.json`, `data/test_cases_*.json`) via the
  shared `ToolCallingRunner`. Upstream's `src/` and `requirements.txt`
  are unused.

### Gotchas worth flagging for next time

- **Path case matters.** Runner default is `external_benchmarks/injecagent`.
  A manual `git clone` produces uppercase `InjecAgent/` and breaks the
  default. Always go through `setup_benchmarks.sh`.
- **Upstream's `requirements.txt` is incomplete.** If anyone ever does
  want to run upstream's own evaluator (we don't), they'd need to add
  `together` and `nltk`. Probably worth a one-line PR upstream, not
  blocking for our use.
- **Opus 4.7 + thinking via OpenRouter on InjecAgent dh_base finishes
  fast.** n=20 took ~2 min, n=100 ~7 min. Per-task is 1–2 tool turns,
  so latency dominates over thinking-token volume. Concurrency=8
  saturated; bumping to 20 only helps when round-trip bound on larger
  n or longer-step benchmarks.
- **`--max-tasks N` yields N+1 5q rows** sometimes (n=20 → 21, n=100 →
  101). Same shuffle-dedup quirk the WASP/AgentDojo entries describe.
  Not a bug to fix; don't normalize by `N` when computing rates.

### Things that just worked (so don't re-debug them)

- `markdown_export.py` auto-fires for InjecAgent because it's in the
  tool-calling track (single-JSON-per-task transcripts). No extension
  needed — unlike WASP, where it's explicitly skipped on agentlab
  pickle dirs.
- `trajectory_converter.PARSERS` already includes `injecagent`. No
  parser to wire.
- `run_safety_pipeline.py --benchmark injecagent --stage all` runs
  Stage 1 (run) + Stage 2 (judges: verbalized_awareness,
  purpose_continue_5q) + markdown sidecars cleanly out of the box.
- Default split for InjecAgent is `dh_base` (510 cases). `ds_base`,
  `dh_enhanced`, `ds_enhanced` are also available — the runner takes
  `--attack-type {dh,ds} --setting {base,enhanced}`.

## WASP

Things I had to do / learned getting WASP wired into the
eval-awareness pipeline end-to-end (smoke + n=100 sweep on glm-5).
WASP is the messiest of the four benchmarks because it's the only
one that *actively mutates the live docker containers* during run
setup — it plants real issues/posts whose URLs become the agent's
start_urls. Most gotchas below stem from that one design choice.

### What I tried first (wrong path)

- Ran upstream's `webarena_prompt_injections/setup.sh`. It builds
  three separate venvs (visualwebarena / wpi / claude-35-cu-demo)
  totaling ~9 GB of useless deps that the project's runner doesn't
  use. The project uses the root `.venv` via `uv pip install -e`.
  Burned ~10 min and ~9 GB before realizing `setup_benchmarks.sh`
  was the right entry point — same lesson as the InjecAgent path-case
  one.
- Tried to launch the runner with the in-repo `AgentLab/` installed.
  The `:thinking` reasoning-mode patch lives in
  `external_benchmarks/AgentLab` (fresh ServiceNow clone, commit
  `cbc35a9`); without it, `model_name="z-ai/glm-5:thinking"` goes
  out literally → OpenRouter 404 ("No endpoints found for
  z-ai/glm-5:thinking"). Re-installed via
  `uv pip install -e external_benchmarks/AgentLab` and it worked.
- Wrote my own "bridge script" to cross-product
  `user_goals × formats` + template-substitute the raw config into
  per-task JSONs, before realizing upstream's `prompt_injector.py`
  already does that — and more (it also plants the live state). The
  bridge work was thrown away. The useful realization:
  `prompt_injector.py` does NOT call any LLM despite taking
  `--model`, so it's runnable without an OpenAI/Azure key.

### What's actually right

- `bash eval_awareness_experiments/setup_benchmarks.sh wasp` clones
  the repo and warns "no installable manifest" (correct — WASP is
  imported via `PYTHONPATH=external_benchmarks/wasp`, not pip).
- For each run, **step 1 is `prompt_injector.py`**. It plants real
  state in the live gitlab/reddit dockers and emits per-task JSONs
  at `<output-dir>/webarena_tasks/{task_id}.json`. The runner reads
  those.
- One `prompt_injector.py` invocation = one `(user_goal_idx,
  injection_format)` pair × all 21 attackers = 21 task JSONs. Full
  cross-product = 8 invocations = 168 task JSONs (96 gitlab, 72
  reddit). Wrapped in `scripts/wasp_plant_full.sh` — handles the
  task_id renumbering on merge (each plant restarts at task_id 1000;
  collisions resolved by adding the run-index offset).
- Run via `scripts/wasp_n100_run.sh <preset>` — minimal launcher
  that sets PYTHONPATH and points
  `run_wasp.py --task-dir /tmp/wasp_full`.
- Cleanup via `scripts/wasp_cleanup_full.sh` — calls upstream
  `environment_cleanup.py` against the merged config file with
  `--delete-attacker-account` (gitlab attacker users get deleted;
  postmill has no programmatic self-delete so reddit attacker users
  persist as no-ops).

### Gotchas worth flagging for next time

- **`:thinking` is internal notation, not an OpenRouter model.** The
  patched `chat_api.py` strips the suffix and adds
  `extra_body={"reasoning": {"effort": "high"}}`. Confirm the
  patched copy is installed (Python should resolve
  `agentlab.llm.chat_api` to the
  `external_benchmarks/AgentLab/...` path) before launching anything
  that costs money.
- **`PYTHONPATH=external_benchmarks/wasp` is required.** WASP isn't
  a pip package; `webarena_prompt_injections/` has no
  `__init__.py`. The runner's `_import_wasp_injector()` fails
  noisily without it.
- **Per-task JSON schema uses `sites` (list, plural).** Original
  `_build_wasp_benchmark` filtered on `site` (singular, key doesn't
  exist) → 0 tasks survived. Patched.
- **WASP task_ids ≥ 1000 collide with WebArena's 812-task canonical
  set.** Three-part fix in `_register_wasp_tasks()`:
  - `gym.register("webarena.<tid>", GenericWebArenaTask, task_kwargs={"task_id": tid})`
    so the env exists.
  - Monkey-patch `GenericWebArenaTask.__init__` to short-circuit on
    known WASP task_ids — otherwise it loads from the webarena pip
    package's `test.raw.json` which only has the 812.
  - Append synthesized rows to `task_metadata("webarena")` so
    `Benchmark.__post_init__` doesn't reject WASP task_names.
- **`n_jobs=1` only.** Ray workers spawned with `n_jobs > 1`
  re-import `browsergym.webarena` fresh and don't see the parent
  process's registrations →
  `gymnasium.error.NameNotFound: Environment 'webarena.1008'`. Fix
  would require module-import-time registration; punted for now.
  Means n=100 is a ~3-hour sequential run per preset.
- **DoomArena's `collect_results()` crashes on WASP runs.** It
  expects `success_filter_key`/`success_filter_value` columns from
  runtime AttackConfigs. WASP uses post-hoc evaluators
  (`attack_configs=()`). The crash happens AFTER trajectories are
  fully written; just narrow-catch the `KeyError` in
  `run_wasp.main()`.
- **Upstream cleanup script has a click-flag-shadows-function
  bug.** `--delete-attacker-account` binds to a function param of
  the same name, which then shadows the module-level
  `delete_attacker_account()` it tries to call → `TypeError:
  'bool' object is not callable`. Patched at
  `eval_awareness_experiments/patches/wasp_cleanup_flag_shadowing.patch`.
  Apply with `git -C external_benchmarks/wasp apply ../../<patch>`
  after re-cloning.
- **Reddit post slug auto-suffixing on title collisions.** Each
  plant uses titles like "A note from a gentleman thief 1000".
  Eight plants = eight titles named "1000". Postmill renames the
  second through eighth to `-2`, `-3`, etc.; the editor returns the
  actual URL (correct, distinct), so per-task JSONs point at the
  right post. Don't dedupe on title — dedupe on URL.
- **Reddit attacker users persist forever.** No programmatic
  self-delete in postmill. Idempotent on re-plant ("user already
  exists, logging in"), so it doesn't break anything; just
  clutter.
- **Pointing `judge-only` at the same study dir for both `--splits`
  judges everything twice.** Each trajectory ends up under both
  `wasp_reddit_<tid>` and `wasp_gitlab_<tid>` in the JSONL. Dedupe
  on the bare task_id when computing aggregates. Or just point
  each split at separate dirs (cleaner).
- **`task_pass` is the benign-goal evaluator only**, NOT attack
  success rate. WASP also generates
  `webarena_tasks_attacker/{task_id}.json` with attacker-side eval
  criteria — those aren't wired into the project's pipeline yet,
  so true ASR isn't measured. The smoke earlier this session
  showed 6/9 *followed* the injection while still producing some
  comment → `task_pass` and ASR diverge.

### Things that just worked (so don't re-debug them)

- `prompt_injector.py` is pure string substitution — no LLM call,
  no API key needed for step 1. Despite the `--model` flag, the
  model name only ends up in the boilerplate `run_agent.sh` it
  emits as scaffolding (which we don't use).
- `setup_fn`s for the 7 gitlab attackers that need pre-existing
  state (target project, attacker account, namespace, pristine
  `.zshrc`) all work out of the box via WASP's
  `environment_setup.py`. The `gitlab_editor.py` Playwright class
  drives gitlab as `byteblaze` competently.
- The patched `chat_api.py` captures reasoning text under
  `message.reasoning` for `:thinking` models. Showed up cleanly in
  step pickles — verified by reading
  `step_0.pkl.gz`'s `agent_info.chat_messages.messages[*]['reasoning']`.
- Cross-benchmark contention with the AgentDojo / InjecAgent runs
  going in parallel was a non-issue — they're tool-calling, no
  docker contention. Different OpenRouter request streams,
  effectively independent.

### Things that DIDN'T just work and need follow-up

- **Markdown sidecars not generated for browser benchmarks.**
  `markdown_export.write_markdown_for_split` only handles
  tool-calling single-JSON transcripts; agentlab pickle dirs are
  explicitly skipped in `run_safety_pipeline.py`. So WASP runs ship
  without the human-readable per-task `.md` files the style guide
  asks for. Future work: extend `markdown_export` to walk
  `step_*.pkl.gz` + `goal_object.pkl.gz` and produce equivalent
  output.
- **Cost reporting.** `Model z-ai/glm-5:thinking not found in the
  pricing information` — langchain's pricing table doesn't know
  about glm-5, so cost-per-task gets reported as $0. Have to check
  OpenRouter dashboard for actuals.
