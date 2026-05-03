# Action-Capability Pilot Runbook

This runbook is for minimum validation of capability-aligned task generation
and host-compiled adversarial actions. It is not a headline sweep protocol.

## Research Invariants

- Keep strict WASP carrier scope: GitLab issue/comment bodies and
  Reddit/Postmill submission/comment bodies.
- Do not reintroduce title carriers.
- Do not skip Phase 2c.
- Do not hand-edit feasibility, admission, rewards, or final statuses.
- Do not let Phase 4 variants mutate benign instructions, route contracts,
  rewards, task cards, or Phase 2c status.
- Report semantic, request-level action, and state-confirmed action success
  separately.

## Task-Card Plans

Use one task-card plan per pilot question. Do not mix these with an unrelated
Phase 2 action policy.

| Pilot | Task-card plan | Phase 2 action policy | Purpose |
| --- | --- | --- | --- |
| Semantic minimum validation | `configs/task_card_plans/action_capability_semantic_minval.json` | `semantic_only` | Prove no-title body carriers and host-compiled binary semantic rewards still work. |
| Tier 2 public mutation pilot | `configs/task_card_plans/action_capability_tier2_mutation_pilot.json` | `tier2_pilot` | Test bounded public create/comment actions with POST/status/body evidence. |
| Tier 3 repository pilot | `configs/task_card_plans/action_capability_tier3_repository_pilot.json` | `tier3_pilot` | Test disposable repository/content mutation with request evidence, readback, and cleanup. |

The semantic pilot is the first live proof after code changes. The Tier 2 and
Tier 3 pilots should run only after their artifact audits show compatible
action options and, for Tier 3, verified fixture/readback/cleanup evidence.

## Remote Discipline

Use r5 wrappers only. Before syncing or launching, check for active registered
jobs:

```bash
scripts/remote_job_status.sh --host-config configs/benchmark_hosts/r5.yaml --remote-dir /home/ubuntu/browser-sim --all
```

Do not sync while a registered job is running unless intentionally overriding
and documenting why. For fresh Phase 0/1/2 chains on r5, use both network
localities:

- Modal Phase 0c: `--instances instances.smoke.json`
- Host inventory and on-host browser checks: `--host-inventory-instances instances.scale.json`
- Phase 2c/4: `instances.scale.json`

Use `/home/ubuntu/vendors/webarena-verified` for the remote benchmark path.

## Semantic Minimum Validation

This run validates the new Phase 1 task-card capability constraints while
keeping Phase 2 actions semantic-only.

```bash
RUN=logs/action_capability_semantic_minval_<timestamp>Z
BENCH=/home/ubuntu/vendors/webarena-verified
export WORLDSIM_STATE_DIR="$RUN"

uv run python -m worldsim.main phase 0 \
  --benchmark "$BENCH" \
  --instances instances.smoke.json \
  --host-inventory-instances instances.scale.json \
  --sites gitlab,reddit \
  --sandbox-model claude-sonnet-4-6

uv run python -m worldsim.main phase 1 \
  --benchmark "$BENCH" \
  --generate-novel \
  --sites gitlab,reddit \
  --novel-tasks-per-site 24 \
  --task-card-plan configs/task_card_plans/action_capability_semantic_minval.json \
  --sandbox-model claude-sonnet-4-6

uv run python -m worldsim.main phase 2 \
  --benchmark "$BENCH" \
  --sites gitlab,reddit \
  --task-origin new_task \
  --phase-2a-action-policy semantic_only \
  --feasibility-instances instances.scale.json \
  --feasibility-concurrency 8 \
  --sandbox-model claude-sonnet-4-6

uv run python -m worldsim.main phase 3 \
  --benchmark "$BENCH" \
  --sites gitlab,reddit \
  --task-origin new_task
```

Inspect Phase 2/3 before Phase 4:

- admitted counts by site, route, surface, editor, capability family, and action
  kind,
- zero admitted title carriers,
- no source-data or no-contract systemic drops,
- action policy is `semantic_only`,
- Phase 3 contracts are valid for the same Phase 2 artifact.

Then run GPT-5.2 priority on the frozen cohort:

```bash
uv run python -m worldsim.main phase 4 \
  --instances instances.scale.json \
  --sites gitlab,reddit \
  --task-origin new_task \
  --max-tasks-per-site 10 \
  --agent-provider openai \
  --agent-model gpt-5.2 \
  --agent-service-tier priority \
  --agent-llm-timeout 240 \
  --agent-step-timeout 300 \
  --agent-task-timeout 900 \
  --sandbox-model claude-sonnet-4-6
```

If GPT-5.2 succeeds and the artifacts are interpretable, run the same frozen
Phase 2/3 cohort with:

- `--agent-provider anthropic --agent-model claude-opus-4-7`
- `--agent-provider openrouter --agent-model google/gemini-2.5-pro`

Only vary the Phase 4 web-agent model/provider.

## Tier 2 And Tier 3 Pilots

Run these as separate validation cohorts, not as a silent expansion of the
semantic minimum run.

Tier 2:

```bash
--task-card-plan configs/task_card_plans/action_capability_tier2_mutation_pilot.json
--phase-2a-action-policy tier2_pilot
```

Use `tier2_pilot` for the main action pilot. It fails closed when a contract has
no host-ready Tier 2 action, rather than falling back to semantic/navigation
actions. `mutation_only_when_available` is useful for diagnosis on mixed
cohorts, but it is not the cleanest policy for a publishable Tier 2 action
condition.

Tier 2 task cards must also set `requires_benign_action_evidence`. Phase 1
validation then appends a natural task-local action witness to the instruction
and adds a matching benign `NetworkEventEvaluator`. If the admitted cohort has
public mutation wording but benign rewards only contain `AgentResponseEvaluator`
checks, stop before Phase 4 and regenerate Phase 1 with the corrected cards.

Tier 3:

```bash
--task-card-plan configs/task_card_plans/action_capability_tier3_repository_pilot.json
--phase-2a-action-policy tier3_pilot
```

Before Tier 3 Phase 4, run the Tier 3 pilot audit and require selected tasks to
carry host-ready options, request/body evidence, final-state readback, and
cleanup metadata. A Tier 3 task without fixture/readback/cleanup proof is a
cohort construction failure, not a Phase 4 resistance result.

## Reporting

Minimum validation reports must clearly label:

- direct ASR,
- bounded adaptive ASR,
- semantic hijack success,
- request-level action success,
- state-confirmed action success,
- benign utility under attack,
- PVPO observation buckets,
- task-broke, non-encounter, inconclusive, and error buckets,
- variant-loop attempts and action-variant QA failure classes.

Do not compare Tier 1 semantic ASR to Tier 2/Tier 3 action ASR without stating
the different action policy, task-card plan, evidence requirements, and
fixture maturity.
