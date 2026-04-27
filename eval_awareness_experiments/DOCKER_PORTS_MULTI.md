# Docker port mapping — multi-stack setup (per-model isolation)

Extension of `DOCKER_PORTS.md` to support running all 6 models in
parallel within an arm. Each model gets its own dedicated stack of
WebArena services, so concurrent agent sessions across models don't
collide on docker state.

Setup script: `scripts/setup_doomarena_per_model_dockers.sh up`
(stop with `... stop`, full remove with `... rm`).

## Layout (DoomArena)

| stack | model | gitlab | forum (reddit) | shopping | shopping_admin |
|---|---|---|---|---|---|
| 1 (existing) | GLM-5 | gitlab_doom :9002 | forum_doom :8081 | shopping :8082 | shopping_admin :8083 |
| 2 | Sonnet-4.6 | gitlab_sonnet :9012 | forum_sonnet :8091 | shopping_sonnet :8092 | shopping_admin_sonnet :8093 |
| 3 | Opus-4.7 | gitlab_opus :9022 | forum_opus :8101 | shopping_opus :8102 | shopping_admin_opus :8103 |
| 4 | GPT-5.2 | gitlab_gpt :9032 | forum_gpt :8111 | shopping_gpt :8112 | shopping_admin_gpt :8113 |
| 5 | Gem-Flash | gitlab_flash :9042 | forum_flash :8121 | shopping_flash :8122 | shopping_admin_flash :8123 |
| 6 | Gem-Pro | gitlab_pro :9052 | forum_pro :8131 | shopping_pro :8132 | shopping_admin_pro :8133 |

## WASP layout (unchanged)

WASP keeps using the original `gitlab` :9001 + `forum` :8080 stack.
WASP's per-model parallelism would need a similar duplication if we
extend this to WASP — not done yet.

## Env var routing

Per-model env vars (paste into a shell or set per-subprocess via
`env VAR=val ...`):

```bash
# GLM-5 (uses existing stack 1 — same as DOCKER_PORTS.md)
export GITLAB="http://localhost:9002"
export REDDIT="http://localhost:8081"
export SHOPPING="http://localhost:8082"
export SHOPPING_ADMIN="http://localhost:8083"

# Sonnet-4.6
export GITLAB="http://localhost:9012"
export REDDIT="http://localhost:8091"
export SHOPPING="http://localhost:8092"
export SHOPPING_ADMIN="http://localhost:8093"

# Opus-4.7
export GITLAB="http://localhost:9022"
export REDDIT="http://localhost:8101"
export SHOPPING="http://localhost:8102"
export SHOPPING_ADMIN="http://localhost:8103"

# GPT-5.2
export GITLAB="http://localhost:9032"
export REDDIT="http://localhost:8111"
export SHOPPING="http://localhost:8112"
export SHOPPING_ADMIN="http://localhost:8113"

# Gemini-3-Flash
export GITLAB="http://localhost:9042"
export REDDIT="http://localhost:8121"
export SHOPPING="http://localhost:8122"
export SHOPPING_ADMIN="http://localhost:8123"

# Gemini-3.1-Pro
export GITLAB="http://localhost:9052"
export REDDIT="http://localhost:8131"
export SHOPPING="http://localhost:8132"
export SHOPPING_ADMIN="http://localhost:8133"
```

## How the new launcher uses these

`scripts/launch_pilot_doomarena_model_parallel.sh` (or equivalent):
fires one process per (arm, model) pair, totaling 4 arms × 6 models
= 24 simultaneous streams. Each stream runs ONE benchmark + arm + model
on its own dedicated docker stack. Within each stream, splits run in
parallel (4 splits using the same model's stack).

Total simultaneous browser sessions at peak: 4 arms × 6 models × 4
splits = **96 browser sessions**, distributed across 24 docker stacks
(4 sessions per stack — each stack handles all 4 sites for 4 arms of
the same model).

Wait, that's not right — each stack only has ONE of each site, so 4
arms hitting the same model's gitlab = 4 sessions on `gitlab_<model>`
container. Same as before per container, just spread across 6
containers per site instead of 1.

## Resource cost (measured at setup)

- Disk: fresh containers share image layers; ~0 GB initial. Will grow
  ~500 MB-2 GB per container over a multi-hour run.
- RAM: each gitlab uses ~10 GB resident, others smaller. 5 new gitlabs
  × ~10 GB + 15 other containers × ~1-2 GB = ~70-80 GB total. Host
  has 1.9 TiB free; fine.
- Boot time: gitlab containers take 2-4 min to fully boot; postmill /
  shopping respond within seconds.

## Cleanup

```bash
# stop only (preserves writable layer if you want to re-use later)
./scripts/setup_doomarena_per_model_dockers.sh stop

# stop + remove (frees writable layer + ~10 GB if heavily used)
./scripts/setup_doomarena_per_model_dockers.sh rm
```

## Known issues

- WASP isn't yet duplicated per-model. WASP's `_register_wasp_tasks`
  also has Ray-subprocess-incompatibility issues separate from the
  docker-isolation issue (see WASP entry in `experiment_log.md`).
- `shopping_final_0712:latest` (64 GB image) is the heaviest — not
  every WebArena workload uses shopping, so if disk gets tight you
  could skip duplicating it (point all 6 stacks at the same shared
  `shopping` :8082). Sharing is safe for read-mostly workloads but
  introduces collision risk on write-heavy tasks.
