# Node3 Handoff: Extra DoomArena Docker Workers

Date: 2026-05-09

Goal: run DoomArena on node3 while this machine runs WASP. We want enough
full WebArena/DoomArena stacks to run the 24 DoomArena cells in parallel:
6 models x 4 arms (`bare`, `xml_safety`, `xml_scenario`, `xml_control`).

## Current Local State

On this machine we already have enough WASP-only workers:

- 6 model-named WASP workers: `glm`, `sonnet`, `opus`, `gpt`, `gemini25`,
  `kimi25`
- 18 generic extra WASP workers: `w01` through `w18`
- `launch_wasp_per_model.sh` has been updated so model specs and worker specs
  are separate; `WORKER_STACK_FILTER=all` means 24 generic workers without
  duplicating model cells.

Do not copy the WASP run to node3 unless needed. Use node3 for DoomArena.

## DoomArena Worker Target

Each DoomArena worker needs four containers:

- GitLab: `gitlab-populated-final-port8023:latest`
- Forum/Reddit: `postmill-populated-exposed-withimg:latest`
- Shopping: `shopping_final_0712:latest`
- Shopping admin: `shopping_admin_final_0719:latest`

Existing script: `scripts/setup_doomarena_per_model_dockers.sh`

Existing launcher: `eval_awareness_experiments/launch_doomarena_per_model.sh`

The launcher already uses a generic queue, but its model list and worker list
are currently tied together in `MODEL_STACKS`. Before adding generic extra
workers, patch it the same way WASP was patched:

- keep a six-model `MODEL_SPECS` list
- keep a separate `WORKER_STACKS` list
- select `SELECTED_MODELS` from `MODEL_SPECS`
- select `SELECTED_WORKERS` from `WORKER_STACKS`

This avoids accidentally duplicating model cells when adding extra workers.

## Suggested Extra Worker Ports

If node3 has no conflicts, add 16 generic DoomArena workers, giving 24 total
when combined with the 8 existing local-style workers:

| stack | gitlab | reddit | shopping | shopping_admin |
|---|---:|---:|---:|---:|
| d01 | 9302 | 8301 | 8302 | 8303 |
| d02 | 9312 | 8311 | 8312 | 8313 |
| d03 | 9322 | 8321 | 8322 | 8323 |
| d04 | 9332 | 8331 | 8332 | 8333 |
| d05 | 9342 | 8341 | 8342 | 8343 |
| d06 | 9352 | 8351 | 8352 | 8353 |
| d07 | 9362 | 8361 | 8362 | 8363 |
| d08 | 9372 | 8371 | 8372 | 8373 |
| d09 | 9382 | 8381 | 8382 | 8383 |
| d10 | 9392 | 8391 | 8392 | 8393 |
| d11 | 9402 | 8401 | 8402 | 8403 |
| d12 | 9412 | 8411 | 8412 | 8413 |
| d13 | 9422 | 8421 | 8422 | 8423 |
| d14 | 9432 | 8431 | 8432 | 8433 |
| d15 | 9442 | 8441 | 8442 | 8443 |
| d16 | 9452 | 8451 | 8452 | 8453 |

The `shopping_admin` URL passed to the launcher should include `/admin`, same
as the current launcher does.

## Resource Budget

Measured here, approximate writable-layer cost:

- WASP stack: ~7-16 GB
- DoomArena full stack: ~35-45 GB

Budget 45 GB per extra DoomArena worker. Sixteen extra workers need roughly
720 GB writable-layer headroom. RAM is usually fine on these boxes; GitLab is
the largest process, around 10-15 GiB resident per container here.

Before creating stacks on node3:

```bash
df -h /local_data /var/lib/docker 2>/dev/null || df -h /local_data
docker info --format 'DockerRootDir={{.DockerRootDir}} Driver={{.Driver}}'
docker system df
```

## Run Command Shape

Use fresh output/log paths. Example:

```bash
tmux new-session -d -s doomarena_n200_reset_node3_20260509 \
  'cd /local_data/temp/max/browser-sim && \
   mkdir -p logs/n200_doomarena_node3_2026-05-09 && \
   N_TASKS=200 \
   MODEL_STACK_FILTER="glm sonnet opus gpt gemini25 kimi25" \
   WORKER_STACK_FILTER=all \
   ARM_FILTER="bare xml_safety xml_scenario xml_control" \
   SPLITS="reddit shopping shopping_admin gitlab" \
   OUTPUT_BASE=eval_awareness_experiments/results/n200_doomarena_node3_2026-05-09 \
   LOG_DIR=logs/n200_doomarena_node3_2026-05-09 \
   BROWSER_RELAUNCH_INCOMPLETE=1 \
   SKIP_EXISTING=1 \
   ./eval_awareness_experiments/launch_doomarena_per_model.sh \
     > logs/n200_doomarena_node3_2026-05-09/launch.log 2>&1'
```

## Reset Caveat

The current DoomArena runner historically used `skip_reset_and_massage=True`
because the WebArena full-reset endpoint was not running. For a clean paper
rerun, node3 should either:

- stand up the WebArena full-reset service and set DoomArena to use the normal
  reset/massage path, or
- recreate the relevant Docker containers before each model x arm cell, like
  the WASP reset path does.

Adding more workers improves parallelism, but it does not by itself solve the
DoomArena reset cleanliness issue.
