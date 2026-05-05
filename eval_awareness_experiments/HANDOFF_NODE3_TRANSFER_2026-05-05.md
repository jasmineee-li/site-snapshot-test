# Handoff - node3 transfer setup - 2026-05-05

## Goal

Move the browser-sim probe/benchmark working environment to `node3` under:

- repo: `/local_data/temp/max/browser-sim`
- Docker data root: `/local_data/docker`
- Hugging Face cache: `/local_data/group_dir/huggingface/hub`

As of the latest update, the node3 repo/env, model caches, Docker stacks, and
WASP planted task pools are ready for benchmark work. Do not start new training,
vLLM serving, or benchmark runs unless explicitly resumed.

## Current node3 state

### Filesystem and Docker

- SSH to `node3` works from the current machine.
- `/` was full; Docker build cache was pruned and root now has ample space.
- Docker now uses `/local_data/docker`.
- Existing Docker config preserved the NVIDIA runtime.
- Old Docker root was moved aside, not deleted:
  - `/var/lib/docker.migrated.20260505011236`
- Latest observed disk after copying the Python environment, Playwright browsers,
  models, and Docker state:
  - `/`: `891G` free
  - `/local_data`: `4.8T` free

Verification:

```bash
ssh node3 'docker info --format "{{.DockerRootDir}}"; df -h / /local_data'
```

Expected Docker root:

```text
/local_data/docker
```

### Repo

Repo is present on node3:

```text
/local_data/temp/max/browser-sim
```

It was copied from local with large result outputs excluded:

- excluded `eval_awareness_experiments/results/`
- excluded Python/tool caches

The repo-local Python environment was copied afterwards:

```text
/local_data/temp/max/browser-sim/.venv  8.3G
```

Verified on node3:

```text
Python 3.12.7
/local_data/temp/max/browser-sim/.venv/bin/python
agentlab/browsergym/browsergym.webarena imports ok
```

Observed repo ref:

```text
branch: data-import
HEAD: 430ed62001b46ad3ecdd594aef9d3f7d0b1f840f
```

Important repo caveat for the node3 Codex agent:

- The node3 working tree was created by rsync, not by a clean clone.
- It is intentionally dirty because large tracked result outputs were excluded
  from the copy and now appear as deleted in `git status`.
- The current handoff doc was copied directly onto node3, and the doc update
  was also pushed to `origin/data-import` as commit `01369989`.
- Do not use `git reset --hard` in this checkout unless the user explicitly
  decides to discard the rsynced working tree state. If a clean Git checkout is
  needed, prefer making a separate fresh clone under a new directory.

The rsync included local code/probe files that were untracked locally at the
time, including:

- `probes/chat_template_experiment.py`
- `probes/eval_sad_stages_oversight.py`
- `scripts/compute_eval_awareness_table.py`
- `eval_awareness_experiments/HANDOFF_ASR_OUTCOME_SPLIT_2026-05-04.md`
- `eval_awareness_experiments/launch_n200_toolcalling_bare.sh`

### Docker Images

The transferred image tar files already existed on node3 and were loaded from:

```text
/local_data/temp/max/webarena-images/gitlab-populated-final-port8023.tar
/local_data/temp/max/webarena-images/postmill-populated-exposed-withimg.tar
/local_data/temp/max/webarena-images/shopping_final_0712.tar
/local_data/temp/max/webarena-images/shopping_admin_final_0719.tar
```

Loaded tags:

```text
gitlab-populated-final-port8023:latest        77.6GB
postmill-populated-exposed-withimg:latest     53.3GB
shopping_final_0712:latest                    64GB
shopping_admin_final_0719:latest              9.45GB
```

Latest Docker footprint observed with `docker system df`:

```text
Images          237.2GB
Containers      223.2GB
```

There are active containers for the WASP and DoomArena stacks; the container
size is not all reclaimable without removing those active stacks.

### Playwright Browser Cache

Copied local Playwright browser binaries to node3 so the WASP prompt injector
can run browser automation without downloading browsers:

```text
/home/max/.cache/ms-playwright  1.7G
```

Verified Chromium launch through the repo `.venv`.

### Model Cache

Copied from local cache to node3:

```text
/local_data/group_dir/huggingface/hub/models--xlangai--OpenCUA-72B
/local_data/group_dir/huggingface/hub/models--mPLUG--GUI-Owl-1.5-32B-Think
```

Observed sizes:

```text
137G  models--xlangai--OpenCUA-72B
63G   models--mPLUG--GUI-Owl-1.5-32B-Think
```

Set these before model work on node3:

```bash
export HF_HOME=/local_data/group_dir/huggingface
export HF_HUB_CACHE=/local_data/group_dir/huggingface/hub
```

## WASP State

The six WASP per-model Docker stacks were created and health-checked.
These are six independent duplicate WASP GitLab/forum site pairs, one pair per
model stack. They are expected to be up concurrently.

Script used:

```bash
cd /local_data/temp/max/browser-sim
./scripts/setup_wasp_per_model_dockers.sh up
./scripts/setup_wasp_per_model_dockers.sh health
```

Health returned HTTP 200 for every GitLab/forum pair:

| stack | GitLab | forum |
|---|---:|---:|
| glm | 9201 | 8201 |
| sonnet | 9211 | 8211 |
| opus | 9221 | 8221 |
| gpt | 9231 | 8231 |
| gemini25 | 9241 | 8241 |
| kimi25 | 9251 | 8251 |

WASP task planting has been run successfully on all six stacks.

Commands used:

```bash
ssh node3
cd /local_data/temp/max/browser-sim
./scripts/setup_wasp_per_model_dockers.sh health
PARALLEL=1 ./scripts/wasp_plant_per_model_dockers.sh
```

Logs are in:

```text
logs/wasp_plant_per_model/
```

Final planted task pools:

| stack | final task dir | task JSONs |
|---|---|---:|
| glm | `/tmp/wasp_full_glm/webarena_tasks/` | 168 |
| sonnet | `/tmp/wasp_full_sonnet/webarena_tasks/` | 168 |
| opus | `/tmp/wasp_full_opus/webarena_tasks/` | 168 |
| gpt | `/tmp/wasp_full_gpt/webarena_tasks/` | 168 |
| gemini25 | `/tmp/wasp_full_gemini25/webarena_tasks/` | 168 |
| kimi25 | `/tmp/wasp_full_kimi25/webarena_tasks/` | 168 |

No WASP planting processes were left running after completion.

## DoomArena State

Despite the user interrupt, the per-model DoomArena setup continued in the
background and completed. No more setup commands are currently running.

Base GLM-compatible stack is up:

| service | container | port |
|---|---|---:|
| GitLab | `gitlab_doom` | 9002 |
| forum/reddit | `forum_doom` | 8081 |
| shopping | `shopping` | 8082 |
| shopping_admin | `shopping_admin` | 8083 |

Additional per-model stacks were created with:

```bash
cd /local_data/temp/max/browser-sim
./scripts/setup_doomarena_per_model_dockers.sh up
```

The script reported all additional stacks healthy. HTTP 200 was observed for:

| stack | GitLab | forum | shopping | shopping_admin |
|---|---:|---:|---:|---:|
| sonnet | 9012 | 8091 | 8092 | 8093 |
| opus | 9022 | 8101 | 8102 | 8103 |
| gpt | 9032 | 8111 | 8112 | 8113 |
| flash | 9042 | 8121 | 8122 | 8123 |
| pro | 9052 | 8131 | 8132 | 8133 |
| gemini25 | 9062 | 8141 | 8142 | 8143 |
| kimi25 | 9072 | 8151 | 8152 | 8153 |

Quick recheck:

```bash
ssh node3 'docker ps --format "{{.Names}} {{.Status}} {{.Ports}}" | sort | grep -E "doom|gitlab_|forum_|shopping"'
```

The setup script does not expose a standalone `health` action. Re-running `up`
is idempotent but will also try to configure existing containers again.

## Not Done Yet

Do these only if we continue on node3:

1. Start vLLM only if node3 is needed.

   For OpenCUA-72B:

   ```bash
   cd /local_data/temp/max/browser-sim
   export HF_HOME=/local_data/group_dir/huggingface
   export HF_HUB_CACHE=/local_data/group_dir/huggingface/hub
   CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 bash scripts/serve_opencua_72b.sh
   ```

   The wrapper defaults to `PORT=8002`, `TENSOR_PARALLEL_SIZE=2`,
   `DTYPE=bfloat16`, `MAX_MODEL_LEN=32768`.

   For GUI-Owl-1.5-32B-Think:

   ```bash
   cd /local_data/temp/max/browser-sim
   export HF_HOME=/local_data/group_dir/huggingface
   export HF_HUB_CACHE=/local_data/group_dir/huggingface/hub
   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_gui_owl_32b.sh
   ```

   The wrapper defaults to `PORT=8003`, `TENSOR_PARALLEL_SIZE=1`.

2. Verify benchmark launchers against node3 ports before scaling.

   Useful files:

   - `eval_awareness_experiments/DOCKER_PORTS_MULTI.md`
   - `eval_awareness_experiments/launch_doomarena_per_model.sh`
   - `scripts/wasp_plant_per_model_dockers.sh`
   - `scripts/run_cua_eval_awareness_matrix.sh`

3. Probe/steering caveat.

   vLLM is fine for unsteered benchmark runs. Steering/probe intervention in
   browser benchmarks still needs an HF-hook generation path; the existing
   matrix script previously skipped steered WASP/DoomArena cells where that
   harness is absent.

4. Optional cleanup if node3 is not needed soon.

   Stop containers without deleting their writable layers:

   ```bash
   cd /local_data/temp/max/browser-sim
   ./scripts/setup_wasp_per_model_dockers.sh stop
   ./scripts/setup_doomarena_per_model_dockers.sh stop
   docker stop gitlab_doom forum_doom shopping shopping_admin
   ```

   Remove containers if we want to reclaim writable-layer space:

   ```bash
   cd /local_data/temp/max/browser-sim
   ./scripts/setup_wasp_per_model_dockers.sh rm
   ./scripts/setup_doomarena_per_model_dockers.sh rm
   docker rm -f gitlab_doom forum_doom shopping shopping_admin
   ```

   After confidence that Docker is stable on `/local_data/docker`, the old
   Docker root can be removed to reclaim root disk:

   ```bash
   sudo rm -rf /var/lib/docker.migrated.20260505011236
   ```

   Do not delete it until we are comfortable that no old Docker state is needed.

## 72B Probe Experiment Note

The 72B probe experiments were not run through vLLM. They used the HF probe
loader, which wraps the model with `HookedTransformerShim` and uses
`device_map="auto"` plus bf16 when loading on CUDA.

Relevant local artifact directories:

```text
probes/trained/opencua-72b-raw-sanity/
probes/trained/opencua-72b-chat-template-span/
```

Observed model snapshot in probe metadata:

```text
/local_data/group_dir/huggingface/hub/models--xlangai--OpenCUA-72B/snapshots/3cc4af5337a1d0010f934284df3c482797266547
```

The raw/original run metadata says:

- probe method: `dom`
- data: `probes/data/contrastive_dataset.json`
- best MCQ layer: 46
- layers: 0-79

The chat-template span run metadata says:

- probe method: `chat_template_mcq_dom`
- `chat_template: true`
- MCQ format: user question plus assistant answer
- extraction position: assistant answer content span
- best MCQ layer from synthetic MCQ heldout: 24
- SAD stages/oversight dev-selected layer: 13
- SAD test AUROC at dev-selected layer: about 0.708
- MCQ-to-framing cross-format AUROC at the selected layer: 0.375

Approximate commands to reproduce:

```bash
cd /local_data/temp/max/browser-sim
export HF_HOME=/local_data/group_dir/huggingface
export HF_HUB_CACHE=/local_data/group_dir/huggingface/hub

# Original/no chat-template sanity run.
CUDA_VISIBLE_DEVICES=0,1 python -m probes.train_probe \
  --model /local_data/group_dir/huggingface/hub/models--xlangai--OpenCUA-72B/snapshots/3cc4af5337a1d0010f934284df3c482797266547 \
  --data probes/data/contrastive_dataset.json \
  --output probes/trained/opencua-72b-raw-sanity/mcq \
  --layers all

# Chat-template MCQ/framing run with SAD layer selection.
CUDA_VISIBLE_DEVICES=0,1 python probes/chat_template_experiment.py \
  --model /local_data/group_dir/huggingface/hub/models--xlangai--OpenCUA-72B/snapshots/3cc4af5337a1d0010f934284df3c482797266547 \
  --output-root probes/trained/opencua-72b-chat-template-span \
  --layers all \
  --dtype bfloat16 \
  --mcq-layer-selection sad \
  --sad-structs-zip /tmp/sad/sad/stages/oversight/structs.zip
```

The exact shell transcript is not in this handoff. The commands above are
reconstructed from the scripts and artifact metadata.
