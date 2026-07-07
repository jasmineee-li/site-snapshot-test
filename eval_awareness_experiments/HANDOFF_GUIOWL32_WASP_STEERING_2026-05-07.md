# GUI-Owl-32B WASP GitLab Success-46 Steering Handoff

Date: 2026-05-07

## TL;DR

Active WASP steering experiment on **GUI-Owl-1.5-32B-Think** using the
46-task GitLab "success" subset from `HANDOFF_WASP_GITLAB_SUCCESS46_STEERING_2026-05-07.md`.
Four arms, four parallel runs, four isolated WASP Docker stacks. Each arm
saves transcripts so we can score the probe on them after.

| arm | tmux | GPUs | TASK_DIR | WASP stack | settings |
|---|---|---|---|---|---|
| 1. baseline | `wasp_s46_base` | 0,1 | `/tmp/wasp_gitlab_success46` (glm) | gitlab `:9201`, forum `:8201` | bare — no steering, no frame |
| 2. steer +0.9 | `wasp_s46_toward` | 2,3 | `/tmp/wasp_gitlab_success46_sonnet` | `:9211/:8211` | `alpha_per_layer=+0.9`, layers `9,8,10,12,15,11` |
| 2'. steer +0.6 | `wasp_s46_toward06` | 4,5 | `/tmp/wasp_gitlab_success46_opus` | `:9221/:8221` | `alpha_per_layer=+0.6` (canonical 72B-WASP recipe), same layers |
| 4. baseline + XML frame | `wasp_s46_xml` | 6,7 | `/tmp/wasp_gitlab_success46_gpt` | `:9231/:8231` | no steering, `SYSTEM_PROMPT_FRAME=safety_test` |

All arms launched at `2026-05-07 01:14 UTC` (after a v1 wave was killed —
see "Bugs Fixed" below). RUN_IDs end with `_v2`.

The earlier −0.9 (steer-away) arm was killed without replacement: at
`alpha_per_layer=-0.9 × N=6` GUI-Owl-32B totally fails to produce parseable
BrowserGym actions even on step 0 (action=None for all 20 tasks).

## Probe & Steering Configuration

- **Probe:** `probes/trained/gui-owl-32b-chat-template-span/framing`
  (16 real + 16 fake colon-terminated framing prompts; vector =
  `mean(real) − mean(fake)`; trained earlier this session).
- **SAD eval (oversight) on the probe:**
  `probes/trained/gui-owl-32b-chat-template-span/sad_stages_oversight_eval_framing.json`.
  Top-1 layer = **9** (AUROC 0.748). Top-6 = `9, 8, 10, 12, 15, 11`.
- **SAD scoring layer / threshold for `_resolve_sad_layer_threshold`:**
  layer **9**, threshold **16.375** (passed via `SAD_SCORE_LAYER` /
  `SAD_SCORE_THRESHOLD`).
- **Steering implementation:** the WASP server uses
  `generate_with_token_gated_steering_generate`
  (`models/common/hf_openai_server.py:274` → `probes/steering.py:335`).
  - Prefill: only user-message *content* tokens are perturbed (no system
    prompt, no chat template, no scaffolding).
  - Generation: hook is enabled only **after the first generated
    sentence** completes (a `_EnableAfterFirstSentenceCriteria` flips
    `state.generation_enabled` → True).
  - This is the Hua-et-al WASP recipe pattern, NOT the broadcast-every-
    token mode used by `probes.steering.steering_hook`. Behavioral chat
    experiments earlier in the session used `steering_hook` (broadcast
    everywhere); WASP is the surgical version.

Asymmetry observed earlier:
- α=+0.9 with token-gated steering (this run): step 0 produces a coherent
  thought + valid action format, but the model often hallucinates BIDs
  that aren't on the page (`fill('546', ...)`, `click('489')`); after
  the BrowserGym ValueError, step 1 emits empty output → terminate.
- α=−0.9: catastrophic — action=None on step 0, no recovery.
- α=+0.6 N=6 (72B-WASP recipe): same step-0-OK / step-1-empty pattern.
- This is much more sensitive than chat behavior suggested. AgentDojo
  workspace earlier showed TSR≈0.75 even at α=+2.0 × N=6 (total
  perturbation 12.0); browser tasks under token-gated steering apparently
  destabilize multi-turn coherence at much lower magnitudes. The +0.6
  arm in this run is explicit follow-up to test whether the canonical
  72B recipe works on this model.

## Bugs Fixed This Session

### 1. `hf_openai_server.py` couldn't load GUI-Owl

`AutoModelForCausalLM` doesn't recognize the Qwen3-VL config used by
GUI-Owl-1.5. Symptom: HF server crashes at startup with
"Unrecognized configuration class … Qwen3VLConfig". OpenCUA-72B (Qwen2.5-VL)
loads fine through the same path.

**Fix** (`models/common/hf_openai_server.py`): `try AutoModelForCausalLM` →
on `ValueError` containing "Unrecognized configuration class", fall back to
`Qwen3VLForConditionalGeneration`, then `AutoModelForVision2Seq`. Committed
as `c8ecbeb9 feat(wasp): GUI-Owl-32B WASP launcher + Qwen3-VL fallback in HF server`.

### 2. Wrong WASP stack ports for non-glm task pools (the "Unauthorized url" mystery)

The 46-task subset was built from `/tmp/wasp_full_glm`, whose task JSONs
hardcode URLs at `localhost:9201`. Three of the four arms (toward+0.9,
toward+0.6, xml) were running against *different* WASP stacks
(sonnet/opus/gpt at 9211/9221/9231). DoomArena's attack gateway flags
out-of-stack URLs as "unauthorized", yields `task_info.error =
'Unauthorized url, terminating task'`, agent_info is empty, and the env
terminates after step 1. From the outside this looked exactly like
"steering broke the model on step 1" — both steered AND the no-steering
xml arm were failing identically, which is what tipped me off.

**Fix:** the per-stack pools `/tmp/wasp_full_{glm,sonnet,opus,gpt,gemini25,kimi25}`
already exist on the box (planted by upstream tooling), and each is
URL-bound to its stack's port. I built per-stack subsets for all four
non-glm stacks:

```bash
TASK_IDS=eval_awareness_experiments/configs/wasp_gitlab_success46_task_ids.txt
for stack in sonnet opus gpt; do
  src=/tmp/wasp_full_$stack
  dst=/tmp/wasp_gitlab_success46_$stack
  mkdir -p "$dst/webarena_tasks"
  while read -r tid; do
    n="${tid#webarena.}"
    cp "$src/webarena_tasks/${n}.json" "$dst/webarena_tasks/${n}.json"
  done < "$TASK_IDS"
done
```

Each `_v2` arm in the current run uses its stack's matching subset.

### 3. Earlier OpenCUA-72B runs aborted because the v1 wave used different ports

The first WASP launch attempt (timestamp `20260507_004732`, with
OpenCUA-72B and 4-GPU layout) was killed before model load completed —
user changed model choice to GUI-Owl-32B. Then the GUI-Owl v1 wave
(`20260507_005112`) had two arms (`toward`, `xml`) fail at server
startup because ports `8003` and `8005` were already bound by orphan
listeners (no owning user, probably leftover from a previous session).
Relaunched on free ports 8020/8021. Then further killed/relaunched when
the URL-mismatch bug above was found.

The current `_v2` arms are on free, distinct ports: 8002 (base),
8020 (toward+0.9), 8022 (toward+0.6), 8021 (xml).

## Files Added This Session

```
eval_awareness_experiments/launch_guiowl32_wasp_full_attacked_with_probes.sh
probes/trained/gui-owl-32b-aware-framing/                  # variant probe (SAD AUROC 0.762; behaviorally weaker)
probes/trained/gui-owl-32b-chat-template-span/sad_stages_oversight_eval_framing.json
probes/data/framing_pairs_aware.json
scripts/agentdojo_steering_sweep.sh                        # multi-layer × alpha sweep harness for AgentDojo
scripts/judge_agentdojo_cells.py                           # standalone LLM-judge over completed cells
scripts/compare_steered_outputs.py
scripts/probe_eval_question.py                             # quick alpha-sweep on a single chat prompt
scripts/train_probes_gui_owl_32b_chat_template.sh
scripts/train_probes_opencua_32b_chat_template.sh
eval_awareness_experiments/launch_opencua32_wasp_full_attacked_with_probes.sh
```

Earlier in the session (before the WASP run) we also did a 13-cell
AgentDojo (layer × alpha) sweep on this same probe — see commits
`60ac8262` (resume), `2d437fad` (max_tokens), `464e66ee`, `27ca7187`,
`3d751a16`, `d9c06fc9`. Findings logged in chat: TSR essentially
untouched across α=0.3–2.0 × N=1–12 on AgentDojo workspace; verbalized
eval-awareness rate (LLM-judge) = 0/520 across the entire grid;
chat-side effects clearly visible at α≥0.6 on direct meta prompts.

## Live Monitor

Background bash task `brec1m1he` (started `2026-05-07 01:14 UTC`) prints
a line each time any arm advances; reports `nontriv` (n_steps > 1, i.e.,
the agent got past step 0 without instant-terminate) and `rew>0` per
arm. Stops when base hits 30/46, any tmux session dies, or 90 min cap.

Manual snapshot:

```bash
for arm in base toward toward06 xml; do
  case $arm in
    base) RUNID=guiowl32_wasp_gitlab_success46_baseline_20260507_005112 ;;
    toward) RUNID=guiowl32_wasp_gitlab_success46_toward_a09_20260507_005112_v2 ;;
    toward06) RUNID=guiowl32_wasp_gitlab_success46_toward_a06_20260507_005112_v2 ;;
    xml) RUNID=guiowl32_wasp_gitlab_success46_xmlsafety_20260507_005112_v2 ;;
  esac
  d=eval_awareness_experiments/results/$RUNID/bare/local_gui-owl-32b-think/_browser_runs/all
  echo "$arm: $(find $d -name summary_info.json 2>/dev/null | wc -l)/46"
done
```

Inspect a transcript:

```bash
RUNID=guiowl32_wasp_gitlab_success46_toward_a09_20260507_005112_v2
ls eval_awareness_experiments/results/$RUNID/bare/local_gui-owl-32b-think/_browser_runs/all/*/*/step_0.pkl.gz | head
```

## Run-After-Run Pipeline

The launcher (`launch_guiowl32_wasp_full_attacked_with_probes.sh`) does
this in sequence after the WASP browser run finishes per arm:

1. Stops the HF OpenAI server.
2. Walks the browser run dir, dumps per-step `chat_messages` to
   `wasp_messages.jsonl`.
3. Re-loads the model + probe and scores each transcript at the SAD
   layer/threshold (3 spans: `last_user_token`, `user_message_mean`,
   `first_assistant_message_mean`), writes `probe_spans.jsonl` +
   `probe_span_summary.json`.
4. Runs `wasp_post_hoc_evaluator.evaluate_wasp_cell_split` (Sonnet-4.6
   classifier, `WASP_ASR_CONCURRENCY=32`) → `wasp_asr_all.jsonl`.
5. Joins reward + ASR + probe spans into `probe_reward_asr_table.csv`,
   `.md`, `_summary.json`.

So each arm produces self-contained probe + ASR + TSR artifacts; cross-
arm comparison is just diffing summaries.

## Open Questions / Followups

- **Does steering at α=+0.6 N=6 produce non-trivial multi-step
  trajectories on GUI-Owl-32B browser** (now that the URL bug is fixed)?
  If yes, that's the apples-to-apples comparison with 72B WASP recipe.
  If no, this model is too sensitive to token-gated steering for browser
  even at the canonical magnitude — fall back to N=3 × α=0.6 (total 1.8)
  or N=6 × α=0.3 (total 1.8) on a free GPU pair.
- **Is the safety_test XML frame alone enough to break GUI-Owl on
  WASP** (independent of steering)? The xml arm answers this. If xml
  also produces only 1-step trajectories on the gpt stack, the
  `safety_test` wrapper is genuinely behaviorally disruptive on this
  model. If it produces healthy multi-step tasks, the v1 wave's xml
  failure was purely the URL bug.
- **Symmetric −α arm:** the −0.9 arm killed itself; want a symmetric
  steer-away condition. Try −0.3 once toward+0.6 lands.
- **Probe-score side:** with arm 1 (baseline) transcripts, we can
  compute the natural distribution of `last_user_token` / `user_message_mean`
  / `first_assistant_message_mean` probe scores per task and per
  attack-success bucket. The launcher already does this in the
  post-run pipeline; just diff the resulting `probe_reward_asr_summary.json`s.
- **Push to remote:** local commits exist on `data-import` branch;
  pushing has been intermittently blocked by the sandbox guard against
  "agent-inferred URLs". User has been pushing manually as needed.

## Useful Paths

- Subset task dirs: `/tmp/wasp_gitlab_success46{,_sonnet,_opus,_gpt}/webarena_tasks/*.json`
- Task ID list (46 GitLab-success tasks): `eval_awareness_experiments/configs/wasp_gitlab_success46_task_ids.txt`
- Source handoff (defines the 46): `eval_awareness_experiments/HANDOFF_WASP_GITLAB_SUCCESS46_STEERING_2026-05-07.md`
- GUI-Owl framing probe: `probes/trained/gui-owl-32b-chat-template-span/framing/`
- AgentDojo sweep results: `results/toolcalling/agentdojo/sweep/guiowl_framing_*_v2_20260506_203902/`
