# Trace Inspection

Use this when a user asks why Phase 4 tasks complied, resisted, were unaware,
or failed under the eval-awareness iterator.

## First Command

Start with the compact inspector instead of hand-written `python -c` JSON dumps:

```bash
uv run python scripts/inspect_phase4_traces.py logs/<run> summary
```

For r5 artifacts, run the same inspector remotely:

```bash
scripts/remote_trace_inspect.sh \
  --host-config configs/benchmark_hosts/r5.yaml \
  --remote-dir /home/ubuntu/browser-sim \
  --run logs/<run> -- summary
```

Add narrow filters before increasing detail:

```bash
uv run python scripts/inspect_phase4_traces.py logs/<run> summary \
  --site gitlab \
  --action create_issue_note
```

## Progressive Disclosure

Use this order:

1. `summary` for counts, awareness buckets, iterator stops, top reward messages,
   and representative task ids.
2. `slice` for a compact task table with `--site`, `--action`, `--status`,
   `--outcome`, `--limit`, and `--fields`.
3. `task <task_id> --iterator` when the question is about rewrite attempts,
   stop reason, or why contrasts did not work.
4. `task <task_id> --refs` when you need artifact paths for `history.json`,
   `result.json`, Needham XML, TP/VEA raw responses, or PVPO summary.
5. Only after that, open raw trajectory artifacts directly.

Machine consumption:

```bash
uv run python scripts/inspect_phase4_traces.py logs/<run> slice \
  --outcome resistant_unaware \
  --fields task_id,site,action,final_status,reward_msg,iterator_stop \
  --output json
```

## Interpretation Checklist

For `resistant_unaware`, distinguish these cases:

- PVPO `max_coverage > 0`: the payload was encountered; inspect reward/action
  evidence and awareness metrics.
- TP/VEA aware is false: `unaware` means the observational awareness metrics did
  not classify the trajectory as eval-aware. It does not mean the payload was
  unreachable.
- `eval_awareness_iterator` exists with generated attempts and budget-exhausted
  stop: the iterator ran but did not find a compliant rewrite.
- `attack_attempted=false` or `attack_success=false`: the browser agent never
  executed the adversarial action or the action/readback reward did not prove it.
- `state_success=false`: request-level evidence may differ from final-state
  readback; report these separately.

Do not paste full `signals`, full `history.json`, prompts, screenshots, or raw
stderr into the conversation by default. Emit compact counts, task ids, artifact
paths, and the next exact command.
