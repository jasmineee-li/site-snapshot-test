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

## Layout (WASP)

Setup script: `scripts/setup_wasp_per_model_dockers.sh up`
(stop with `... stop`, full remove with `... rm`).

WASP's original `gitlab` :9001 + `forum` :8080 stack is still available
for single-stream runs. Per-model WASP runs use separate WASP-only
containers and must be planted separately:

```bash
./scripts/setup_wasp_per_model_dockers.sh up
./scripts/setup_wasp_per_model_dockers.sh health
./scripts/wasp_plant_per_model_dockers.sh
```

| stack | model | gitlab | forum (reddit) | task dir |
|---|---|---|---|---|
| glm | GLM-5 | gitlab_wasp_glm :9201 | forum_wasp_glm :8201 | /tmp/wasp_full_glm |
| sonnet | Sonnet-4.6 | gitlab_wasp_sonnet :9211 | forum_wasp_sonnet :8211 | /tmp/wasp_full_sonnet |
| opus | Opus-4.7 | gitlab_wasp_opus :9221 | forum_wasp_opus :8221 | /tmp/wasp_full_opus |
| gpt | GPT-5.2 | gitlab_wasp_gpt :9231 | forum_wasp_gpt :8231 | /tmp/wasp_full_gpt |
| gemini25 | Gemini-2.5-Pro | gitlab_wasp_gemini25 :9241 | forum_wasp_gemini25 :8241 | /tmp/wasp_full_gemini25 |
| kimi25 | Kimi-K2.5 | gitlab_wasp_kimi25 :9251 | forum_wasp_kimi25 :8251 | /tmp/wasp_full_kimi25 |
| w01 | generic worker | gitlab_wasp_w01 :9301 | forum_wasp_w01 :8301 | /tmp/wasp_full_w01 |
| w02 | generic worker | gitlab_wasp_w02 :9311 | forum_wasp_w02 :8311 | /tmp/wasp_full_w02 |
| w03 | generic worker | gitlab_wasp_w03 :9321 | forum_wasp_w03 :8321 | /tmp/wasp_full_w03 |
| w04 | generic worker | gitlab_wasp_w04 :9331 | forum_wasp_w04 :8331 | /tmp/wasp_full_w04 |
| w05 | generic worker | gitlab_wasp_w05 :9341 | forum_wasp_w05 :8341 | /tmp/wasp_full_w05 |
| w06 | generic worker | gitlab_wasp_w06 :9351 | forum_wasp_w06 :8351 | /tmp/wasp_full_w06 |
| w07 | generic worker | gitlab_wasp_w07 :9361 | forum_wasp_w07 :8361 | /tmp/wasp_full_w07 |
| w08 | generic worker | gitlab_wasp_w08 :9371 | forum_wasp_w08 :8371 | /tmp/wasp_full_w08 |
| w09 | generic worker | gitlab_wasp_w09 :9381 | forum_wasp_w09 :8381 | /tmp/wasp_full_w09 |
| w10 | generic worker | gitlab_wasp_w10 :9391 | forum_wasp_w10 :8391 | /tmp/wasp_full_w10 |
| w11 | generic worker | gitlab_wasp_w11 :9401 | forum_wasp_w11 :8401 | /tmp/wasp_full_w11 |
| w12 | generic worker | gitlab_wasp_w12 :9411 | forum_wasp_w12 :8411 | /tmp/wasp_full_w12 |
| w13 | generic worker | gitlab_wasp_w13 :9421 | forum_wasp_w13 :8421 | /tmp/wasp_full_w13 |
| w14 | generic worker | gitlab_wasp_w14 :9431 | forum_wasp_w14 :8431 | /tmp/wasp_full_w14 |
| w15 | generic worker | gitlab_wasp_w15 :9441 | forum_wasp_w15 :8441 | /tmp/wasp_full_w15 |
| w16 | generic worker | gitlab_wasp_w16 :9451 | forum_wasp_w16 :8451 | /tmp/wasp_full_w16 |
| w17 | generic worker | gitlab_wasp_w17 :9461 | forum_wasp_w17 :8461 | /tmp/wasp_full_w17 |
| w18 | generic worker | gitlab_wasp_w18 :9471 | forum_wasp_w18 :8471 | /tmp/wasp_full_w18 |

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

WASP per-model env vars:

```bash
# GLM-5
export GITLAB="http://localhost:9201"
export REDDIT="http://localhost:8201"
export WASP_TASK_DIR="/tmp/wasp_full_glm"

# Sonnet-4.6
export GITLAB="http://localhost:9211"
export REDDIT="http://localhost:8211"
export WASP_TASK_DIR="/tmp/wasp_full_sonnet"

# Opus-4.7
export GITLAB="http://localhost:9221"
export REDDIT="http://localhost:8221"
export WASP_TASK_DIR="/tmp/wasp_full_opus"

# GPT-5.2
export GITLAB="http://localhost:9231"
export REDDIT="http://localhost:8231"
export WASP_TASK_DIR="/tmp/wasp_full_gpt"

# Gemini-2.5-Pro
export GITLAB="http://localhost:9241"
export REDDIT="http://localhost:8241"
export WASP_TASK_DIR="/tmp/wasp_full_gemini25"

# Kimi-K2.5
export GITLAB="http://localhost:9251"
export REDDIT="http://localhost:8251"
export WASP_TASK_DIR="/tmp/wasp_full_kimi25"
```

Extra WASP workers `w01` through `w18` are generic queue workers, not
additional models. `launch_wasp_per_model.sh` keeps the six model list separate
from the worker stack list, so `WORKER_STACK_FILTER=all` gives 24 workers for
the 24 model x arm cells without duplicating model cells. With
`RESET_STACK_BEFORE_CELL=1`, each worker recreates and replants its own task
pool before running the cell.

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
./scripts/setup_wasp_per_model_dockers.sh stop

# stop + remove (frees writable layer + ~10 GB if heavily used)
./scripts/setup_doomarena_per_model_dockers.sh rm
./scripts/setup_wasp_per_model_dockers.sh rm
```

## Known issues

- WASP per-model stacks must be planted before use. The generated task
  JSONs contain stack-specific URLs, so `/tmp/wasp_full_gpt` should be
  used only with `gitlab_wasp_gpt`/`forum_wasp_gpt`, etc.
- WASP's `_register_wasp_tasks` also has Ray-subprocess-incompatibility
  issues separate from docker isolation (see WASP entry in
  `experiment_log.md`).
- `shopping_final_0712:latest` (64 GB image) is the heaviest — not
  every WebArena workload uses shopping, so if disk gets tight you
  could skip duplicating it (point all 6 stacks at the same shared
  `shopping` :8082). Sharing is safe for read-mostly workloads but
  introduces collision risk on write-heavy tasks.
