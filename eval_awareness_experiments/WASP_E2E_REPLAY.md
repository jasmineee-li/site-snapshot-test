# WASP Replay E2E ASR

`wasp_replay_e2e_evaluator.py` computes WASP attacker-side e2e ASR from saved
AgentLab/BrowserGym trajectories without querying the target model again.

It reads saved `step_*.pkl.gz` actions, replays them in BrowserGym for
`program_html` attacker tasks, and runs WASP's final attacker evaluator on the
resulting page state. For `exfil` attacker tasks, it mirrors WASP's log-based
check directly against the saved action/message trace because our AgentLab runs
do not produce WASP's `conversation_render_*.html` files.

Example dry-run, mapping only:

```bash
.venv/bin/python -m eval_awareness_experiments.wasp_replay_e2e_evaluator \
  --cell-dir eval_awareness_experiments/results/n200_2026-04-29/wasp/bare/openai_gpt-5.2_thinking \
  --task-dir /tmp/wasp_full \
  --output-jsonl /tmp/wasp_replay_full_dryrun.jsonl \
  --dry-run
```

Example one-task browser replay:

```bash
.venv/bin/python -m eval_awareness_experiments.wasp_replay_e2e_evaluator \
  --cell-dir eval_awareness_experiments/results/n200_2026-04-29/wasp/bare/openai_gpt-5.2_thinking \
  --task-dir /tmp/wasp_full \
  --output-jsonl /tmp/wasp_replay_program_smoke.jsonl \
  --task-id 1095 \
  --single-site gitlab
```

If Playwright/Chromium is blocked by sandboxing, run the same command outside
the sandbox. The script writes per-task rows plus a final summary row containing
`n_scored`, `n_success`, and `asr`.

Important reset behavior: BrowserGym's per-task reset does not restore the
backing GitLab/Reddit database state. `program_html` replays mutate issues,
posts, comments, and reactions, so production replay runs hard-reset the
corresponding WASP stack before each cell by default:

```bash
./scripts/run_wasp_e2e_replay_n200.sh
```

The launcher maps `/tmp/wasp_full` and `/tmp/wasp_full_<stack>` task pools to
their GitLab/Reddit ports, recreates the matching Docker containers from the
populated images, disables forum rate limits, waits for site readiness, and
replants that stack with `wasp_plant_full_stack.sh` before scoring that cell.
Leaving `RESET_STACK_BEFORE_CELL=0` is only appropriate for dry-runs,
exfil-only checks, or quick browser smoke tests on disposable state.

Do not run targeted WASP cleanup or replant between individual tasks unless you
also regenerate the task JSONs. The task JSONs contain URLs and object IDs
created during planting, so deleting those objects between tasks invalidates
later tasks. WebArena `env-ctrl /init`-style hooks are only config/cache
refreshes; they are not a database or planted-object reset for WASP.

For per-model planted stacks, pass the matching task pool, e.g.
`--task-dir /tmp/wasp_full_gpt`. The script infers the corresponding
`/tmp/wasp_full_plant_gpt` attacker configs and reconstructs the merge-time
task-id renumbering.
