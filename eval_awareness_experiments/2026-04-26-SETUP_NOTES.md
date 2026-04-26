# Runbook — Running the eval-awareness pipeline

Operational guide for running experiments. If you've cloned the repo and want
to actually run something, start here.

For deeper context:
- Data extraction setup: `SETUP.md`
- Upstream benchmark package installs and patches: `SAFETY_BENCHMARKS_HANDOFF.md`
- Result formatting / table conventions: `results_presentation_guide.md`
- Project journal: `experiment_log.md` (newest at top)
- Architecture / what each module does: `CLAUDE.md`

---

## General setup (any benchmark)

### 1. Environment

The repo uses **uv** (`uv.lock` + `pyproject.toml`), not pip or conda. Assume
`uv sync` has been run once so `.venv/` exists.

```bash
# Activate (or prefix every command with `uv run`)
source .venv/bin/activate

# Sanity check
.venv/bin/python -c "import eval_awareness_experiments; print('ok')"
```

Do not install packages into a separate conda env. Earlier handoffs used a
`world-sim` conda env + raw pip — that path is deprecated. The canonical
install target is `.venv/` via `uv pip`.

### 2. .env at the project root

Create `/local_data/temp/max/browser-sim/.env` with:

```bash
OPENROUTER_API_KEY="sk-or-v1-..."
# Plus any benchmark-specific vars (WebArena URLs etc. — see SETUP.md)
```

Every script loads dotenv from project root automatically. **Don't put .env
inside `eval_awareness_experiments/`** — the convention is project-root.

### 3. Install upstream benchmark packages

Once-per-clone:

```bash
bash eval_awareness_experiments/setup_benchmarks.sh            # all
bash eval_awareness_experiments/setup_benchmarks.sh agentdojo  # one
```

For WASP/DoomArena Docker requirements, see `SAFETY_BENCHMARKS_HANDOFF.md` —
those have additional setup beyond the package install.

### 4. Verify your OpenRouter balance + key

Before kicking off long runs, eyeball the balance. Each `:thinking` model call
on Opus can be $0.50–$2 per task and stages run hundreds of calls.

```bash
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
import os, urllib.request, json
key = os.environ['OPENROUTER_API_KEY']
d = json.loads(urllib.request.urlopen(urllib.request.Request(
    'https://openrouter.ai/api/v1/credits',
    headers={'Authorization': f'Bearer {key}'}), timeout=5).read())['data']
print(f'remaining: \${d[\"total_credits\"] - d[\"total_usage\"]:.2f}')
"
```

Rough cost estimates (per n=20 run, agent + judge stages, AgentDojo workspace):
- GLM-5:thinking: ~$5–15
- Kimi-2:thinking: ~$10–25
- Opus 4.7:thinking: ~$30–60
(Stage 2 — judges on Opus 4.6:thinking — is ~$20–40 of those numbers
regardless of agent model. It's the bottleneck for the cheap models.)

---

## The unified pipeline

`run_safety_pipeline.py` is the entry point for *every* benchmark.
Three stages selectable by `--stage`: `all`, `run-only`, `judge-only`.

```bash
.venv/bin/python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark {doomarena|wasp|agentdojo|injecagent|eia} \
    --model-name <openrouter-slug>:thinking \
    --condition {baseline|eval_cue|scenario_cue} \
    --extra-instructions-preset <preset> \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --shuffle 42 \
    --concurrency 8 \
    --output-dir eval_awareness_experiments/results/n20_<model>_<preset>_v2_$(date +%Y%m%d_%H%M)
```

### Stages

- **`--stage all`** (default): run agent (stage 1) → judge trajectories
  (stage 2). One subprocess per benchmark split.
- **`--stage run-only`**: produce trajectories, no judging. Useful when you
  want to defer judges until you have new judge code.
- **`--stage judge-only`**: re-judge existing trajectory dirs without
  re-running agents. Pass `--existing-dirs split:path` to point it at the
  trajectory roots. **The classifier version that runs is whatever code is
  on disk at the moment the judge subprocess loads** — see "Pulling
  mid-run" below.

### Concurrency

The bottleneck is target-model TPM via OpenRouter, not local CPU.

- `--concurrency 8` (default): safe across all models including Opus thinking.
- `--concurrency 20`: GLM-5 / Kimi handle this fine. Opus 4.7:thinking can
  occasionally 429 at 20 — `retry.py` handles 429s with backoff, so worst
  case is wallclock not failure.
- The judge stage uses a separate semaphore (default 64 in `llm.py`) and
  rarely needs adjustment.

### Naming convention for `--output-dir`

```
n<N>_<model-slug>_<preset>_<round>_<YYYYMMDD>_<HHMM>
```

- `<model-slug>`: `glm5`, `opus47`, `kimi26`, `qwen3` (drop the `:thinking`)
- `<preset>`: `none`, `bare`, `private`, `scratchpad`, `green`, `motive`,
  etc. Drop the `scratchpad_` prefix when context is clear.
- `<round>`: `v2` for current canonical classifier; `v3` if/when classifier
  wording changes again. Absent for one-shots.

Example: `n20_opus47_private_v2_20260426_0257`

### Output layout

```
<output_dir>/
├── pipeline_manifest_<benchmark>.json     # config + per-split status
├── <benchmark>/<split>/
│   ├── trajectory_awareness_results.jsonl # judge outputs (one row per task per judge)
│   └── markdown/                          # auto-generated per-task sidecars
│       ├── run_meta.md                    # composed config header
│       └── workspace__user_task_NN__injection_task_M.md
└── run.log                                 # piped stdout/stderr
```

Per-task transcript JSONs live OUTSIDE the output_dir, in
`results/toolcalling/<benchmark>/<split>/<run_name>/`. The pipeline_manifest
points at them via `run_dir`.

### Markdown sidecars

Auto-generated by `markdown_export.py` after stage 2 for **tool-calling
benchmarks only** (AgentDojo, InjecAgent). Browser benchmarks (DoomArena,
WASP) currently skip this — agentlab pickle dirs aren't supported yet.

The sidecars are committed to git for canonical v2 result dirs (raw
JSON/JSONL stays gitignored). Each contains: composed system prompt, task
+ injection, full per-step trajectory, final answer, all judge results.

### Resumability

`run_causal_experiment.py` (the multi-cell wrapper) supports
`--skip-existing` via `run_manifest.py`. It tracks completed cells in a
JSON manifest and skips them on restart. **Verify the skip logic before
restarting after a code change** — if it's based on trajectory presence
not judge presence, you might end up with mixed-version judges.

### Pulling mid-run is dangerous

The pipeline launches stage 1 (agent) as a fresh subprocess per split.
Each subprocess loads judges/runners from disk at startup. If you `git
pull` mid-run:

- Already-launched subprocesses keep OLD code in memory (Python module cache).
- New subprocesses load NEW code.
- You end up with mixed v1/v2 results in one run.

**Rule**: don't pull during a run. If a fix landed and you need v2 results,
either wait for the run to finish, then pull + re-run; or kill, pull,
restart fresh.

---

## Per-benchmark sections

### AgentDojo

**Track**: tool-calling. Uses shared `tool_calling_runner.py` (OpenAI SDK
function calling on OpenRouter).

**Suite**: workspace is the only one we've used. Has 41 user tasks ×
~16 injection tasks = ~640 possible (user, injection) combinations.

**Quirk**: `--max-tasks 20 --shuffle 42` yields **21 tasks**, not 20.
AgentDojo's task selection logic over-shoots when shuffle hits a particular
combination of (user_task, injection_task) pairs that share user_tasks. So
your "n=20" runs are usually n=21 in practice. The metrics are robust to
this; just don't be confused when the JSONL has 21 rows.

**Run command**:

```bash
.venv/bin/python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark agentdojo \
    --splits workspace \
    --model-name z-ai/glm-5:thinking \
    --condition baseline \
    --extra-instructions-preset scratchpad_private \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --shuffle 42 \
    --concurrency 20 \
    --output-dir eval_awareness_experiments/results/n20_glm5_private_v2_$(date +%Y%m%d_%H%M)
```

**Trajectory output**: each task produces one JSON at
`results/toolcalling/agentdojo/workspace/<run_name>/workspace__user_task_NN__injection_task_M.json`.
The JSON has `system_prompt`, `task_instruction`, `steps`
(thought/action/observation per step), `final_answer`, `passed`,
`attack_success`.

**Cost (rough, n=20)**:
- GLM-5:thinking + judges on Opus 4.6:thinking: ~$5–10 total
- Opus 4.7:thinking + same judges: ~$30–60 total

**Common pitfalls**:

1. **`--shuffle 0`** = no shuffle. Always runs the first 20 tasks in suite
   order. Fine for reproducibility but means every run hits the same set.
   Use a non-zero shuffle for diverse sampling.
2. **Scratchpad emission rate is preset-dependent**. On GLM-5:
   `scratchpad_bare` ~75%, `scratchpad_private` ~70%, `scratchpad_wtfy`
   ~50% (preset removed). On Opus, much higher. Don't expect 100%.
3. **The first run on a model can hit cold-start latency** on OpenRouter —
   your stage 1 wallclock for run #1 is not representative.
4. **Concurrency=20 works on GLM-5** end-to-end. On Opus 4.7:thinking it
   can occasionally 429; defaults of 8 are safer.
5. **Manifest config-leak (fixed)**: judge-only re-runs used to write
   `model=None preset=none` in the output manifest. Fixed in commit
   `ad37a40` — `run_meta.json` is now written next to trajectories at
   stage 1 and read back at stage 2. If you see old `model=None`
   manifests, they're pre-fix.

**Reading results**:
- Headline numbers: `experiment_log.md`'s most recent
  `n100-glm5-bare-private` (or similar) entry.
- Per-task qualitative: `<output_dir>/agentdojo/workspace/markdown/*.md`.
- VEA-flagged tasks: filter the trajectory_awareness_results.jsonl for
  rows where `experiment_type == "trajectory_verbalized_awareness"` and
  `classification != "Real"`.

---

### WASP

(To be filled in by the WASP-running Claude. Key items to cover:
prompt_injector.py step 1 setup, the `wasp_full` task pool merge,
why `--n-jobs 1` is required, attacker account cleanup,
agentlab study_dir layout. See latest `experiment_log.md` entry for
`n100-glm5-bare-private-wasp` for current state.)

---

### InjecAgent

(To be filled in by the InjecAgent-running Claude. Key items to cover:
attack-type/setting split semantics, the dh_base default,
InjecAgent-specific tool catalog quirks. The benchmark uses
`tool_calling_runner.py` so the general setup applies.)

---

### DoomArena

(To be filled in by the DoomArena-running Claude. Key items to cover:
BrowserGym + AgentLab study setup, the OOda popup attack mode,
WebArena URL env vars, the `attack_summary_info.json` per-task file,
known Ray-subprocess issues. See `SAFETY_BENCHMARKS_HANDOFF.md` for the
upstream-package install + AgentLab patches.)

---

### EIA (import-only)

Imported trajectories from a third-party SeeAct dump. No runtime — stage 1
is a no-op; stage 2 reads `data/eia/manifest.json`.

```bash
# One-time import
.venv/bin/python -m eval_awareness_experiments.import_eia_trajectories \
    --eval-results path/to/eval_results.zip

# Judge
.venv/bin/python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark eia \
    --stage judge-only \
    --judge-model anthropic/claude-opus-4.6:thinking \
    --judges verbalized_awareness purpose_continue_5q \
    --output-dir eval_awareness_experiments/results/eia_v2_$(date +%Y%m%d_%H%M)
```

---

## Common gotchas (all benchmarks)

1. **The classifier version matters**. v2 (current canonical) threads
   `{question}` into `PURPOSE_GRADER_PROMPT` so identity-style answers on
   prompts 11/12 don't get mis-classified as `Refused`. Pre-v2 numbers
   (any dir tagged `_12prompts_` or `_fixed_classifier_`) are NOT directly
   comparable to v2 numbers. Always tag classifier version when reporting.
2. **5PQ is now 12 prompts, not 5.** Class still named
   `PurposeContinue5qJudge` for result-file compat. JSON files in
   `judges/prompts/purpose_continue_5q/` define the 12 probes.
3. **Don't lump `any_eval` into one number**. Always show `align`
   separately — it's the cross-model discriminator. Lumping
   {Alignment, Capability, Other} together hides the most interesting
   finding (Opus has 0.62 align on AgentDojo private, GLM-5 has 0.27 —
   that gap collapses to 0.13 if you only show `any`).
4. **Curated-8 vs Classic-5**. Prompts 04, 05, 11, 12 are reliably
   low-elicitation across all models we've tested. The curated-8 metric
   (drops those four, averages over {01, 02, 03, 06, 07, 08, 09, 10}) is
   ~0.10 higher than classic-5 and more discriminating. Report both.
5. **Markdown sidecars are gitignored except for canonical v2 dirs**.
   Inner `eval_awareness_experiments/results/.gitignore` excludes
   `*.jsonl`, `*.csv`, `*.png`, `*.json`, `*.log`, `_pilot_logs/`. The
   `.md` sidecars are committed for canonical runs because they're
   human-readable trajectory dumps.
6. **`run_safety_pipeline.py` re-judges silently overwrite previous
   judge JSONLs**. Output dirs use `'w'` mode in `experiments/base.py`.
   Always use a fresh timestamped `--output-dir`; never re-judge on top
   of an old one.

---

## After every run — output checklist

(Per `results_presentation_guide.md`.)

1. Markdown sidecars committed (auto-generated by `markdown_export.py`
   for tool-calling benchmarks).
2. Three tables added to your `experiment_log.md` entry's Results section:
   - **Table 1**: one row per `(benchmark, model, preset)`.
   - **Table 2**: per-question detail mini-table per `(model, preset)`.
   - **Table 3**: optional cross-model pivot (Align share by prompt).
3. Each table gets a self-contained heading naming the benchmark, split,
   condition, and (if relevant) classifier version. They get screenshotted.
4. Cost line: actual cost, balance delta, brief reason if cost diverged
   from expectation.
5. 2–4 bullets of observation prose.
6. Atomic commits — sidecars, log, code changes in separate logical
   commits. Push.

---

## Verification — read these manually before trusting numbers

In priority order:

1. **VEA-flagged trajectories**: any task where
   `trajectory_verbalized_awareness.classification != "Real"`. Open the
   sidecar, verify the evidence quote actually appears in context, and
   the classification fits.
2. **5PQ classification spot-checks**: pick 1–2 tasks per (model, preset).
   Read each prompt's grader call and verify the classifier label matches
   the model's actual reflection.
3. **Scratchpad emission verification**: open any sidecar with a
   scratchpad preset, confirm the model emitted
   `<unfiltered_observations>...</unfiltered_observations>` tags, and
   read the content (often it's a task plan, not reflection — that's the
   "format-as-frame" finding).
4. **Cross-preset task overlap**: same shuffle should produce the same
   task IDs. `diff <(ls bare/markdown/) <(ls private/markdown/)` should
   be empty.
5. **Manifest sanity**: every output dir's `pipeline_manifest_*.json`
   should have correct `model_name`, `preset`, `n_trajectories`, and
   `status: "judges_complete"`.

For automated checks, see `verify.py` (TBD — happy to write if useful).

---

## Pinging the right Claude

The four Claude instances split work as follows:

| Benchmark | Track | Section author in this file |
|---|---|---|
| AgentDojo | tool-calling | Filled in (this Claude) |
| InjecAgent | tool-calling | TBD |
| WASP | browser | TBD |
| DoomArena | browser | TBD |

If you're picking up a run and your benchmark's section above is a stub,
it means that Claude instance hasn't filled it in yet. The general
section + the tool-calling-track gotchas above should still apply.
