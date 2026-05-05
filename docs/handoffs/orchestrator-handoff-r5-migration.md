# Orchestrator handoff — r5 migration + Phase 3/4 rerun

**Date:** 2026-04-16
**Branch:** `feat/worldsim-v5`
**Last commit:** `b2fbc00 feat: add migration tooling for r5 scale-out (30x replica config)`
**Status:** paused awaiting user bug fixes to `scripts/generate_compose_scale.py`

This handoff supersedes `orchestrator-handoff-auth-migration-phase4-rerun.md` and
`orchestrator-handoff-paper-run-v1.md`. It is self-contained: a fresh Claude
session reading *only this doc* should be able to resume the paper run.

---

## TL;DR

We are mid-migration from a saturated m5.xlarge (4 vCPU / 16GB) to an
r5.4xlarge (16 vCPU / 128GB) so Phase 3 and Phase 4 can exploit ~16-way
parallelism instead of ~6. The r5.4xlarge is up, volumes are extracted, but
no containers are running yet. The user is actively fixing 3 bugs in
`scripts/generate_compose_scale.py` on a call and said **"wait on my word"**
before deploying anything on the new host. Do not `ssh ubuntu@3.12.221.9`
and start `docker compose up` until the user confirms their fixes are in.

Everything upstream of infrastructure (Phase 0 recon, Phase 2 injections,
validated benign tasks, auth migration, evaluator shims, runtime tokens,
gpt-5.4-mini Responses API wrapper) is already committed and correct.

---

## 1. Infrastructure state

| Thing | Value | Purpose |
|-------|-------|---------|
| m5.xlarge | `i-0353213d8b1d35117` @ `18.117.99.179` (stopped) | rollback only — do not start |
| r5.4xlarge | `i-03acfc08597207960` @ `3.12.221.9` (running) | target host for 30-replica scale-out |
| Elastic IP | `eipalloc-0739c982e65410647` → `3.12.221.9` | future-proof if we stop/start r5 |
| EBS | `vol-*` 1TB gp3 attached to r5, root | 619G used / 969G total after volume extract |
| Security group | `sg-08792057943b27a65` | currently only has m5 ports 7770/7780/8023/9999/8888/3030 + SSH; proxy ports not yet added |
| IAM role | `worldsim-ec2-benchmark-backup` | attached to r5 via instance profile; read/write on `s3://benchmark-archives/` only |
| S3 cold storage | `s3://benchmark-archives/webarena/` (us-east-2, Standard-IA, ~$3/mo) | 4 files, 265GB: `nominatim_volumes.tar`, `osm_tile_server.tar`, `osrm_routing.tar`, `wikipedia_en_all_maxi_2022-05.zim` |
| Modal | `ashton-chew-1` workspace, Claude Code via `CLAUDE_CODE_OAUTH_TOKEN` | sandboxes for Phase 0c, 2, 4 diagnosis |
| AWS account | `324025606582` (Ashton Chew's personal) | vCPU quota = 16 (matters below) |

**Why r5.4xlarge, not r5.8xlarge:** AWS personal account vCPU quota is 16.
r5.8xlarge (32 vCPU) is not obtainable without a quota increase request, so
we pivoted mid-migration. The migration plan doc still reads "r5.8xlarge"
in places — treat it as r5.4xlarge (16 vCPU, 128GB RAM, 1TB gp3) in
practice. 16 vCPU is enough for our 6-site smoke profile and probably
enough for 30 replicas if we tune; the generator already emits 30, not 16.

**Rollback marker:** `.m5_instance_id` in the repo root holds
`i-0353213d8b1d35117`. If r5 is wedged beyond recovery, start m5, update
`instances.json` `site_url` back to `18.117.99.179`, re-apply
`scripts/patch_webarena_containers.sh` if needed.

### Volumes on r5 (already extracted, verified)

All 7 docker-volume payloads were extracted from the S3 tars into
`/var/lib/docker/volumes/` on r5:

- `osrm_volumes_profile-car` (6.1GB)
- `osrm_volumes_profile-foot` (7.8GB)
- `osrm_volumes_profile-bike` (7.4GB)
- `osm_tile_server_volume` (41GB)
- `nominatim-flatnode` (**81.4GB** — anomaly: m5 had 38GB; see §7)
- `openstreetmap-website_db` (56MB)
- `wikipedia_zim` (89GB)

Total post-extract: 619GB / 969GB disk.

### S3 contents (committed reference)

See `docs/handoffs/` and memory `reference_benchmark_archives_s3.md`.
Restore script: `scripts/restore_benchmark_archives_from_s3.sh` (committed
in `ac4a81d`). Flags: `--wiki-only`, `--skip-wiki`. Intra-region restore is
10–15 min.

---

## 2. What the user is fixing RIGHT NOW (blocks Phase 3 start)

Three bugs in `scripts/generate_compose_scale.py` found during the smoke-test
design review on the call. **User said "wait on my word"** — do not attempt
to reproduce or fix these yourself.

1. **site_name canonicalization.** Generator emits `"shopping_1"`,
   `"shopping_2"` etc. in the output `instances.json` fragment. The rest of
   the pipeline (phase 3/4 `instances_for_site("shopping")`,
   `BENCHMARK_PROFILE_shopping.json` mapping, reward dispatch) keys on the
   *canonical* site name. Fix: keep `site_name: "shopping"` for every
   replica, distinguish by `instance_id` or index field, not by renaming.

2. **Auth field preservation.** Generated fragments drop `auth`, `api_auth`,
   `agent_auth` blocks that exist in the base `instances.json`. Without
   these, runtime token acquisition in `worldsim/auth_tokens.py` can't run
   and Phase 3/4 fail at seeding.

3. **Direct-port vs proxy-port ambiguity.** `site_url` in generated
   fragments must be the **direct real port** (e.g. `http://3.12.221.9:7770`)
   because Phase 3 and Phase 4 hit the site directly. The nginx proxy on
   port + 10000 is **only for Phase 0c live verification** from Modal
   sandboxes. Don't let the generator point `site_url` at the proxy.

When user confirms fixes are in, the path forward is §4 below.

---

## 3. Everything that is already done (committed)

Chronological by commit, not by logical area:

| Commit | Summary |
|--------|---------|
| `5b54d93` | relax Phase 4 seed pre-flight for missing `db_connection` |
| `2cbcf31` | add live instance verification to Phase 0c injection surface discovery |
| `962eec7` + `7ab95ab` | remove SQL seeding entirely (violates threat model: regular authenticated user cannot write DB) — -1622 LOC |
| `3004f00` | gpt-5.4-mini OpenRouter support for Browser Use |
| `118b6bd` | nginx token-auth reverse proxy for Phase 0c (Modal→EC2 network path) |
| `fd7ebd5` | runtime token generation (`worldsim/auth_tokens.py`), credentials-in-`instances.json`, fresh tokens per run |
| `1d32012` | Phase 0c prompt: allow POST probes, add delivery feasibility probing |
| `42210d7` + `205f570` | voice exemplar registry keyed on `source_field` pattern (stable across Phase 0c reruns) |
| `e45b803` | fix stale Phase 0c prompt test |
| `d1a7179` | remove extra_body overrides from gpt-5.4-mini config |
| `198608f` | strip think tags from gpt-5.4-mini structured output + bootstrap map storage_state |
| `2eb95cd` | use native `reasoning_effort` parameter instead of think-tag wrapper |
| `4a9e85b` | route OpenAI through OpenRouter BYOK, delete Responses API wrapper |
| `65d50f3` | **restore** `ChatOpenAIResponses` wrapper for native OpenAI Responses API — Chat Completions cannot suppress `<think>` tag leakage for gpt-5.4-mini; **three upstream WebArena evaluator bugs shimmed in `worldsim/rewards.py` `_apply_webarena_vendor_shims`** (value_normalizer schema normalization, AgentResponse string coercion) |
| `b7becf7` | A/B study arms: openai native Responses vs openrouter pinned to openai provider |
| `ac4a81d` | document `benchmark-archives` S3 cold storage + restore script |
| `f1f7e0d` | r5.8xlarge migration plan (empirical probing based) |
| `b2fbc00` | migration tooling: `scripts/generate_compose_scale.py`, `scripts/scale_config.yml`, `scripts/benchmark_host.sh`, `scripts/gitlab.rb.tuned` |

### Key architectural decisions, for context

- **SQL seeding is gone.** Removed in `962eec7`. Rationale: threat model is
  "authenticated regular user" — writing directly to the DB is a privileged
  action that regular users cannot perform via the UI. `db_connection` in
  `instances.json` is optional and only used for postcondition verification.
  Don't re-add SQL seeding paths.

- **Proxy is Phase 0c only.** Phases 3 and 4 use real `site_url` values.
  The proxy port + 10000 scheme + `X-Worldsim-Token` header exists solely so
  Modal sandboxes (dynamic IPs, EC2 security group blocks them) can probe
  live instances during Phase 0c site profiling. Real benchmark ports
  remain locked down to known sources.

- **gpt-5.4-mini needs OpenAI Responses API.** Chat Completions leaks
  `<think>` tags into the structured output `thinking` field, breaks JSON
  parse. `worldsim/llm_wrapper.py` `ChatOpenAIResponses` uses
  `client.responses.parse(..., reasoning={"effort":"none"})`. A/B arms in
  `worldsim/agent_config.py`: `openai` → Responses wrapper,
  `openrouter` → `ChatOpenRouter` with pinned provider and nested
  `extra_body={"extra_body": {...}}`.

- **Evaluator shims** in `worldsim/rewards.py` rewrite three upstream bugs
  in-flight so we don't fork WebArena Verified:
    1. `value_normalizer.py:149-151` crashes on `{"type":"null"}` schemas → rewrite to `{"type":"array"}`
    2. same file, object-type variant → same fix
    3. `agent_response_evaluator.py:120` requires list-of-strings → coerce single strings

- **Runtime tokens.** `worldsim/auth_tokens.py:acquire_tokens_for_instances`
  runs at start of phases 3/4, reads credentials (`auth`, `api_auth`,
  `agent_auth`) from `instances.json`, POSTs to `token_endpoint` /
  `token_generator`, and stashes fresh bearer tokens in an in-memory map
  used by seeding and postcondition verification. **This is why bug #2
  above matters** — without the credential blocks, this function can't run.

---

## 4. Forward plan after user unblocks

**Do not run these without the user's explicit "go":**

### 4a. Regenerate compose + instances.json from fixed generator

```bash
# local
cd /Users/ashtonchew/projects/browser-sim
./scripts/generate_scale_r5.sh
```

This preserves the canonical 6-instance `instances.json` and writes the
generated scale artifacts to:

- `scripts/docker-compose.scale.yml`
- `scripts/proxy_ports.conf`
- `instances.scale.json`
- `instances.scale.json.fragment`

Verify output:

- `instances.scale.json` has `site_name: "shopping"` not `"shopping_1"` (bug 1)
- Every instance entry contains `auth`, `api_auth`, `agent_auth` copied from base (bug 2)
- `site_url` is `http://3.12.221.9:<real_port>`, not the proxy port (bug 3)

### 4b. Ship compose + proxy conf to r5, bring up containers

```bash
# from repo root
./scripts/bootstrap_r5.sh
```

`bootstrap_r5.sh` regenerates the scale artifacts locally, stages
`scripts/docker-compose.scale.yml` onto the host at
`/home/ubuntu/docker-compose.yml`, runs a security-group preflight against
`instances.scale.json`, and then runs the generic bootstrap flow.

`bootstrap_ec2.sh` now uploads the pinned vendored `webarena-verified`
source snapshot, builds local amd64 site images with the env-ctrl fallback
baked in, writes `/home/ubuntu/.env` with `WORLDSIM_ADVERTISE_HOST`,
`WORLDSIM_BIND_HOST`, and `WORLDSIM_DB_BIND_HOST`, restores the
map/wiki assets, and then runs `docker compose -f /home/ubuntu/docker-compose.yml up -d`.

If you want a different remote compose location, set both
`COMPOSE_DIR_REMOTE` and `COMPOSE_FILE_REMOTE` when invoking
`bootstrap_ec2.sh`; the override file, `.env`, and `docker compose` call now
use those paths consistently.

`scripts/patch_webarena_containers.sh` and `scripts/wa_envctrl_patcher.py`
remain in the repo as an emergency fallback only if the baked-image path is
unavailable on a host that already has stale containers running.

### 4c. Security group — add proxy ports

```bash
# only if we want Phase 0c reruns; not needed for Phase 3/4
aws ec2 authorize-security-group-ingress \
  --group-id sg-08792057943b27a65 \
  --ip-permissions 'IpProtocol=tcp,FromPort=13030,ToPort=19930,IpRanges=[{CidrIp=0.0.0.0/0}]' \
  --region us-east-2
```

### 4d. Deploy proxy and mint a token

```bash
./scripts/deploy_benchmark_proxy.sh --host 3.12.221.9
# writes .proxy_token in repo; reference it from instances.scale.json:
#   "verification_proxy": { "token_file": ".proxy_token", "token_env": "WORLDSIM_VERIFICATION_PROXY_TOKEN", "scheme": "http", "port_offset": 10000 }
```

The previous m5 proxy token was removed from checked-in docs/configs. Mint a
new one on r5 to keep blast radius tight.

### 4e. Sanity check all replicas

```bash
# /init responds 200 on every real-port/env-ctrl pair
for ip in 7770 7780 8023 9999 8888 3030; do
  curl -fsS http://3.12.221.9:$((ip+1))/init || echo "FAIL $ip"
done
```

### 4f. Start Phase 3

```bash
# reset pipeline state
python3 -c "
import json, pathlib
p = pathlib.Path('logs/pipeline_state.json')
s = json.loads(p.read_text())
s['status'] = 'failed'
s['step'] = 'phase_2'
p.write_text(json.dumps(s, indent=2))
"

export $(grep -v '^#' .env | grep -v '^$' | xargs)
uv run python -m worldsim.main phase 3 \
  --benchmark vendors/webarena-verified \
  --instances instances.scale.json \
  --agent-provider openai \
  --agent-model gpt-5.4-mini \
  > logs/phase_3_rerun.log 2>&1 &
```

Phase 3 reads the current paper-run `logs/phase_2/adversarial_tasks.json`
(312 tasks, auth-migration compliant). The earlier 516-task figure came from
the separate 2026-04-15 Phase 2 v2 smoke-quality run documented in
`docs/handoffs/codex-handoff-phase-2-quality-audit.md`; it is not the task
count in the rerun artifact on disk now. Expect ~16-way worker parallelism
once the generator emits 16+ canonical-site instances.

Some Phase 2 task bodies still embed the old m5 IP under
`agent_context.auth_mechanism.storage_state.form_login.login_url`, but that
field is Phase 0d bootstrap metadata, not a Phase 3/4 runtime input.
Phase 3/4 resolve agent auth from `instances*.json`, so those stale URLs do
not need task-body rewriting for the r5 rerun.

### 4g. Phase 4, ASR analysis, archive

After Phase 3 completes:

```bash
uv run python -m worldsim.main phase 4 \
  --benchmark vendors/webarena-verified \
  --instances instances.scale.json \
  --agent-model claude-sonnet-4-6 \
  > logs/phase_4_rerun.log 2>&1
```

Then per-cell ASR from `logs/phase_4/results.json` (Gate 1 ≥ 0.6 filter),
`cp -r logs logs/paper_run_v1`, write `docs/paper_run_v1_summary.md`.

---

## 5. File map (touched in this session)

```
docs/
  handoffs/
    orchestrator-handoff-r5-migration.md          <-- this doc
    orchestrator-handoff-auth-migration-phase4-rerun.md  (superseded)
    orchestrator-handoff-paper-run-v1.md          (superseded)
  migration/
    r5-8xlarge-scale-migration-plan.md            <-- 20-section runbook; read "r5.4xlarge" where it says "r5.8xlarge"

scripts/
  generate_compose_scale.py        <-- user is fixing 3 bugs here NOW
  scale_config.yml                  <-- replica counts, port bases
  benchmark_host.sh                 <-- host smoke-test harness
  gitlab.rb.tuned                   <-- memory-tuned gitlab config for replicas
  deploy_benchmark_proxy.sh         <-- nginx token-auth proxy installer
  proxy_ports.conf                  <-- site-to-port mapping
  restore_benchmark_archives_from_s3.sh
  wa_envctrl_patcher.py             <-- env-ctrl EXTERNAL_SITE_URL fallback patch
  patch_webarena_containers.sh      <-- applies patcher inside all webarena containers

worldsim/
  auth_tokens.py                    <-- runtime token acquisition
  llm_wrapper.py                    <-- ChatOpenAIResponses (OpenAI native)
  agent_config.py                   <-- openai vs openrouter A/B arms
  rewards.py                        <-- _apply_webarena_vendor_shims, _coerce_agent_response_strings
  seeding.py                        <-- SQL paths removed, cross-site delivery binding
  phases/
    phase_0_recon.py                <-- live verification + proxy routing
    phase_3_benign.py               <-- calls acquire_tokens_for_instances
    phase_4_adversarial.py          <-- same, plus graceful db_connection skip

logs/
  phase_0c/BENCHMARK_PROFILE_*.json <-- 6 profiles, manually patched for REST API + required fields + available_entities
  phase_2/adversarial_tasks.json    <-- 312 current paper-run tasks, auth-migration compliant
  phase_3/20260416_*/               <-- previous m5 runs, do not reuse
  phase_4/20260416_124814/          <-- previous m5 run, do not reuse
  pipeline_state.json               <-- currently step=phase_3, status=running (stale; reset before next run)

instances.json                      <-- points at m5 IP 18.117.99.179; will be superseded by instances.scale.json
.proxy_token                        <-- m5 proxy token; mint a new one for r5
.m5_instance_id                     <-- rollback marker: i-0353213d8b1d35117
```

---

## 6. What's running right now (shells / monitors / processes)

**Nothing.** Specifically:

- No local `worldsim` process — `ps aux | grep worldsim` shows only this Claude Code session
- No background `Bash` in this Claude conversation has an active job
- No `Monitor` tasks running (user asked them killed earlier: "your monitor is so aggressive")
- No Modal app running
- r5.4xlarge has 0 containers (`docker ps` would return empty)
- m5.xlarge is stopped
- Phase 3 that was running on m5 was killed by user ("yes kill it and migrate") before migration started

`logs/pipeline_state.json` still says `step=phase_3, status=running` —
stale, needs reset before next Phase 3.

---

## 7. Open anomalies / things to watch

1. **nominatim-flatnode 81.4GB vs m5's 38GB.** On r5 the extracted volume
   is 2.1× larger than on m5. Possible causes: the S3 tar included
   additional content (backup manifests?), or tar extracted duplicate
   entries. Does not block Phase 3 startup (map site should still work),
   but note for postmortem. Investigate by:
   - `du -sh /var/lib/docker/volumes/nominatim-flatnode/_data/*` on r5
   - compare file-by-file with m5 when it's restarted

2. **security group proxy ports not yet added** to `sg-08792057943b27a65`.
   Only needed for Phase 0c reruns. Phase 3/4 are direct to real ports and
   our home IP is already allowed.

3. **vCPU quota = 16** on the AWS account. Cannot spin up additional EC2
   while r5.4xlarge is running (16 vCPU used). To bring m5 back for
   comparison, stop r5 first.

4. **gpt-5.4-mini provider A/B.** Both `openai` and `openrouter` arms are
   committed. We defaulted to `openai` (native Responses) because
   OpenRouter had a 402 credit issue even with BYOK. Can switch arms via
   `--agent-provider` flag.

5. **`logs/phase_3/validated_tasks.json` was empty** earlier this session —
   was a stale symlink. Reconstructed from the dated subdirs. For Phase 4
   to proceed, this file must exist with the validated benign task subset.

---

## 8. What NOT to do

- Do not `docker compose up` on r5 until user confirms generator fixes (§2)
- Do not re-add SQL seeding (architectural decision — see `962eec7`)
- Do not "fix" the `worldsim-v5-technical-specifcation.md` typo (load-bearing)
- Do not import from `AgentLab/` (dead predecessor code, kept read-only)
- Do not manage benchmark environment lifecycles from the orchestrator
  (reset_endpoint between tasks is the one exception)
- Do not start m5 while r5 is running (16 vCPU quota)
- Do not delete the S3 tars — they are the only way to rebuild a WebArena
  instance in <2 hours
- Do not run Phase 0c just to refresh profiles — the manually patched
  profiles in `logs/phase_0c/BENCHMARK_PROFILE_*.json` contain live
  verification against the m5; a fresh run against r5 would regenerate
  everything (expensive LLM cost, voice registry churn)

---

## 9. Verification checklist before resuming

- [ ] User confirms generator fixes are in
- [ ] `git diff scripts/generate_compose_scale.py` shows the 3 fixes
- [ ] `python scripts/generate_compose_scale.py ...` produces
      `instances.scale.json` with canonical site_names, preserved auth
      blocks, and real-port `site_url`
- [ ] All 6 containers (smoke profile) or 30 (scale profile) healthy on r5
- [ ] `.proxy_token` refreshed if Phase 0c rerun is planned
- [ ] `logs/pipeline_state.json` reset to `step=phase_2, status=failed`
- [ ] `.env` exports `OPENAI_API_KEY` and `CLAUDE_CODE_OAUTH_TOKEN`
- [ ] Run Phase 3; expect ~16-way parallelism, roughly under an hour for 312 tasks

---

## 10. References

- Spec: `docs/worldsim-v5-technical-specifcation.md`
- Migration runbook: `docs/migration/r5-8xlarge-scale-migration-plan.md`
- Memory: `reference_benchmark_archives_s3.md`, `reference_webarena_aws.md`
- Proxy setup: `README.md` §"Proxy Setup (Phase 0c live verification)"
- S3 restore: `scripts/restore_benchmark_archives_from_s3.sh`
