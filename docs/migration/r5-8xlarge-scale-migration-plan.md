# WorldSim v5 Scale Migration Plan: m5.xlarge -> r5.8xlarge

Status: historical migration plan, partially implemented. Replaces the prior
2TB/$189/mo plan. The current r5 runbook is
`docs/handoffs/rigor-run-setup.md`; this file is retained for design rationale
and migration provenance.

Rewritten to absorb empirical probing of the live m5.xlarge: 30 replicas vs
24, 1TB disk vs 2TB, ~$109/mo vs $189/mo, shared-RO hydration once rather
than per replica. Do not run any step on m5.xlarge until the in-flight Phase
3 completes; pre-migration cleanup (section 18) requires an idle host.

## 1. Architecture

```
                +-----------------------------------------------------------+
                |  r5.8xlarge (us-east-2, 32 vCPU / 256GB RAM / 1TB gp3)    |
                |                                                            |
                |   Nginx benchmark-proxy (ports 11000..12099 token-auth)    |
                |             |                                              |
                |   +---------+-----------+---------+---------+---------+    |
                |   |         |           |         |         |         |    |
                |   v         v           v         v         v         v    |
                |  shop[0..3] sadmin[0..3] git[0..7] red[0..3] map[0..3] wiki[0..5]  |
                |  (4 repls) (4 repls)   (8 repls) (4 repls) (4 repls)  (6 repls)    |
                |                                                            |
                |  Shared RO docker volumes (mounted N times, one copy):     |
                |    map_tile_db (41GB)                                      |
                |    map_routing_car (6.1GB)                                 |
                |    map_routing_bike (7.4GB)                                |
                |    map_routing_foot (7.8GB)                                |
                |    map_nominatim_flatnode (38GB)                           |
                |    map_tiles (54MB)                                        |
                |    map_style (10MB)                                        |
                |    wikipedia_zim (89GB)                                    |
                |                                                            |
                |  Per-replica writable volumes (map only):                  |
                |    map_nominatim_db_<i> (37GB * 4)                         |
                |    map_website_db_<i> (56MB * 4)                           |
                |                                                            |
                |  Docker bridge network worldsim-bench: 172.20.0.0/20       |
                +-----------------------------------------------------------+
                                             ^
                                             |  instances.json (30 entries)
                                             |
                                +----------------------------+
                                | Modal + local orchestrator |
                                | worldsim.main phase 3/4    |
                                +----------------------------+
```

Totals: 30 replicas, ~470GB docker data, ~570GB used incl OS/logs/headroom.

## 2. Port allocation scheme

Convention: `site_base + 10*replica_index` for real ports. Proxy ports add
the existing `PORT_OFFSET` (default 10000) so proxy_port = real_port + 10000.
env-ctrl stays on port 8877 inside each container; host-mapped env-ctrl port
is `real_port + 1` to preserve today's pattern.

| Site | Replicas | Real port base | Real port range | env-ctrl range | Proxy port range |
|------|----------|----------------|-----------------|----------------|------------------|
| shopping | 4 | 7770 | 7770,7780,7790,7800 | 7771,7781,7791,7801 | 17770..17800 |
| shopping_admin | 4 | 7810 | 7810,7820,7830,7840 | 7811,7821,7831,7841 | 17810..17840 |
| gitlab | 8 | 8023 | 8023,8033,8043,...,8093 | 8024,8034,...,8094 | 18023..18093 |
| reddit | 4 | 9900 | 9900,9910,9920,9930 | 9901,9911,9921,9931 | 19900..19930 |
| map | 4 | 3030 | 3030,3040,3050,3060 | 3031,3041,3051,3061 | 13030..13060 |
| wikipedia | 6 | 8888 | 8888,8898,8908,...,8938 | 8889,8899,...,8939 | 18888..18938 |

shopping_admin is rebased to 7810 so its replicas don't collide with shopping
replicas at +10 stride. Security group `sg-08792057943b27a65` adds 30 new
proxy ports in contiguous ranges: 13030-13060, 17770-17840, 18023-18093,
18888-18938, 19900-19930. Real ports stay localhost-only.

## 3. Phased execution order

| Phase | What | Duration | Gate to next |
|-------|------|----------|-------------|
| A | Provision r5.8xlarge + EBS | 15 min | Instance SSH-reachable, disk mounted |
| B | Data hydration from S3 | 75 min | All shared volumes populated, sentinel present |
| C | Compose generator + config | 15 min | Generator emits valid YAML, compose config parses |
| D | Scale out (30 containers) | 10 min | All 30 env-ctrl /init returns 200 |
| E | Nginx proxy | 5 min | All 30 proxy ports listening, auth works |
| F | Orchestrator cutover | 10 min | Phase 0c smoke passes on new host |
| G | Full Phase 3 validation | 60 min | Within +/-2 tasks of m5.xlarge baseline |
| H | Stop/start CLI | 20 min | `benchmark_host.sh status` reports ready |
| I | Decommission m5.xlarge | T+2 weeks | Only after 2 clean weeks on r5 |

Gate G is the hard gate. If Phase 3 diverges by more than 2 tasks from
baseline, do not proceed to I; triage differences first.

## 4. Phase A - Provision r5.8xlarge

```
aws ec2 run-instances \
  --region us-east-2 \
  --image-id ami-0c7217cdde317cfec \
  --instance-type r5.8xlarge \
  --key-name webarena-key \
  --security-group-ids sg-08792057943b27a65 \
  --subnet-id <same as m5.xlarge> \
  --iam-instance-profile Name=worldsim-benchmark-host \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":1000,"VolumeType":"gp3","Iops":6000,"Throughput":250,"DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=worldsim-bench-r5}]'
```

Then on the host:

- `sudo apt update && sudo apt install -y docker.io docker-compose-plugin awscli unzip jq`
- `sudo usermod -aG docker ubuntu`
- Raise `fs.nr_open=1048576` and `nofile` ulimit to 65536 in `/etc/security/limits.conf`
- Bump `net.netfilter.nf_conntrack_max=262144` in `/etc/sysctl.d/99-worldsim.conf`
- Mount the 1TB gp3 at `/var/lib/docker` before starting docker, so the
  writable layers land on the big volume, not the root device.

Allocate Elastic IP and associate to the instance. Record new IP.

## 5. Phase B - Data hydration

The sharing insight from probing: six of nine map volumes and the wikipedia
ZIM are never mutated after import, so they can be mounted read-only by every
replica that needs them. Only `map_nominatim_db` and `map_website_db` need
per-replica copies.

### B.1 Restore tars from S3 (15 min)

Reuse `scripts/restore_benchmark_archives_from_s3.sh` as-is. It pulls four
files totaling ~265GB into `/home/ubuntu/downloads/`.

### B.2 Hydrate shared read-only volumes (once)

For each of the 6 shared map volumes and the wikipedia ZIM, create a single
docker volume and extract into it:

- `webarena-verified-map-tile-db_shared` <- `osm_tile_server.tar`
- `webarena-verified-map-routing-car_shared`, `..._bike`, `..._foot` <- `osrm_routing.tar` (split by subdir)
- `webarena-verified-map-nominatim-flatnode_shared` <- `nominatim_volumes.tar` (flatnode subdir)
- `webarena-verified-map-tiles_shared`, `..._style_shared` <- `osm_tile_server.tar` (respective subdirs)
- `webarena-verified-wikipedia-zim_shared` <- `wikipedia_en_all_maxi_2022-05.zim`

Each container mounts the corresponding `_shared` volume with the `:ro`
option. Docker volumes support `ro` bind; containers cannot corrupt shared
data.

Write a sentinel file `/_hydrated_<sha>` in each shared volume after
extraction so compose-generator can detect and skip re-hydration.

### B.3 Hydrate per-replica writable volumes

Only `map` has them. For each of the 4 map replicas:

- `webarena-verified-map-nominatim-db_<i>` (37GB) - extracted from
  `nominatim_volumes.tar` nominatim-db subdir
- `webarena-verified-map-website-db_<i>` (56MB) - extracted from the same tar

Total map disk: 6 shared (100GB) + 4 * 37GB writable = 248GB.

### B.4 No hydration needed

shopping, shopping_admin, gitlab, reddit: data is baked into the image.
`docker pull` once, replicas share the layer via overlayfs. Writable layers
land in `/var/lib/docker/overlay2/` on the 1TB volume.

## 6. Phase C - Docker Compose design

### C.1 Generator

New file: `scripts/generate_compose_scale.py`

Reads `scripts/scale_config.yml` plus the canonical base `instances.json` and
emits four artifacts:

- `/home/ubuntu/docker-compose.yml` (staged locally, scp'd up)
- full `instances.json` with all 30 runtime entries and preserved auth blocks
- `instances.json` fragment with all 30 entries
- `scripts/proxy_ports.conf` regenerated with 30 lines

It can also fail fast on mutable image refs via `--require-pinned-images`.
The checked-in wrapper `scripts/generate_scale_r5.sh` preserves the canonical
base `instances.json` and writes generated artifacts to
`scripts/docker-compose.scale.yml`, `scripts/proxy_ports.conf`,
`instances.scale.json`, and `instances.scale.json.fragment`.

### C.2 scale_config.yml format

```yaml
network:
  name: worldsim-bench
  subnet: 172.20.0.0/20

sites:
  shopping:
    image: shopping/magento:latest
    replicas: 4
    real_port_base: 7770
    port_step: 10
    mem_limit: 6g
  shopping_admin:
    image: shoppingadmin/magento:latest
    replicas: 4
    real_port_base: 7810
    port_step: 10
    mem_limit: 4g
  gitlab:
    image: gitlab/gitlab-ce:sameersbn
    replicas: 8
    real_port_base: 8023
    port_step: 10
    mem_limit: 16g
    shm_size: 2g
    gitlab_rb: /home/ubuntu/gitlab.rb
  reddit:
    image: postmill/postmill:latest
    replicas: 4
    real_port_base: 9900
    port_step: 10
    mem_limit: 4g
  map:
    image: webarena/openstreetmap:latest
    replicas: 4
    real_port_base: 3030
    port_step: 10
    mem_limit: 12g
    shared_ro:
      - map_tile_db
      - map_routing_car
      - map_routing_bike
      - map_routing_foot
      - map_nominatim_flatnode
      - map_tiles
      - map_style
    per_replica_writable:
      - map_nominatim_db
      - map_website_db
  wikipedia:
    image: webarena-wikipedia-amd64:latest
    replicas: 6
    real_port_base: 8888
    port_step: 10
    mem_limit: 2g
    shared_ro:
      - wikipedia_zim
```

### C.3 Sample compose output

```yaml
services:
  shopping-0:
    image: shopping/magento:latest
    container_name: webarena-verified-shopping-0
    ports:
      - "127.0.0.1:7770:80"
      - "127.0.0.1:7771:8877"
    environment:
      - BASE_URL=http://127.0.0.1:7770
    mem_limit: 6g
    networks:
      worldsim-bench:
        ipv4_address: 172.20.0.10

  map-0:
    image: webarena/openstreetmap:latest
    container_name: webarena-verified-map-0
    ports:
      - "127.0.0.1:3030:80"
      - "127.0.0.1:3031:8877"
    mem_limit: 12g
    volumes:
      - webarena-verified-map-tile-db_shared:/var/lib/postgresql/tile-db:ro
      - webarena-verified-map-routing-car_shared:/data/routing/car:ro
      - webarena-verified-map-nominatim-flatnode_shared:/nominatim/flatnode:ro
      - webarena-verified-map-nominatim-db_0:/var/lib/postgresql/nominatim-db
      - webarena-verified-map-website-db_0:/var/lib/postgresql/website-db

networks:
  worldsim-bench:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/20

volumes:
  webarena-verified-map-tile-db_shared:
    external: true
  webarena-verified-map-routing-car_shared:
    external: true
  ...
  webarena-verified-map-nominatim-db_0: {}
  webarena-verified-map-nominatim-db_1: {}
  ...
```

Service-per-replica (vs `deploy.replicas`) because we're on plain compose,
not swarm; also lets us pin static ports and per-replica volumes cleanly.

### C.4 Tuned gitlab.rb

Drop `/home/ubuntu/gitlab.rb` with:

```
puma['worker_processes'] = 2
sidekiq['concurrency'] = 10
postgresql['shared_buffers'] = "512MB"
prometheus_monitoring['enable'] = false
```

Bind-mount into each gitlab replica at `/etc/gitlab/gitlab.rb`. Memory per
gitlab drops from the default ~4GB runtime to ~2GB, saving ~16GB across 8
replicas.

## 7. Phase D - Scale out

```
cd /home/ubuntu
sudo docker compose up -d
```

Then run the modified bootstrap verification: same logic as existing
`step_verify_envctrl`, but loops over all 30 rows instead of 6. For gitlab
replicas, the same `docker exec -d ... setsid /usr/local/bin/env-ctrl serve
--port 8877` respawn trick applies (found in bootstrap_ec2.sh step 9 - the
sentinel means reconfigure won't re-run, but the env-ctrl process still
needs the detached respawn if pkill'd).

Concurrent /init across 8 gitlab replicas is safe per the probing finding
(bootstrapped sentinel, no shared state, per-container /tmp).

Gate D: all 30 /init return 200.

## 8. Phase E - Nginx proxy

Regenerate `scripts/proxy_ports.conf` with 30 lines (emitted by the
generator). Re-run:

```
./scripts/deploy_benchmark_proxy.sh --host <new-ip> --port-map scripts/proxy_ports.conf
```

`deploy_benchmark_proxy.sh` already reuses the existing token from
`.proxy_token` so the orchestrator config does not churn unless
`--new-token` is passed.

Open the 30 new proxy ports in security group `sg-08792057943b27a65`. Script
prints the exact list.

## 9. Phase F - Orchestrator cutover

Regenerate `instances.json` with 30 entries from the canonical base config.
Each entry keeps the existing schema and auth blocks, but fans out across
replicas:

- `site_name` stays canonical (`shopping`, `gitlab`, etc.)
- `site_url` points at the real web port
- `reset_endpoint` points at the real env-ctrl port (`real_web_port + 1`)
- `replica_index` / `replica_name` identify the specific replica
- `verification_proxy` remains Phase 0c-only and never replaces runtime URLs
- optional `db_connection` is rewritten per replica for post-condition reward
  checks only, never for seeding

Run Phase 0c smoke:

```
uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-verified --tiered
```

Phase 0c will auto-fanout up to 30 workers. Confirm no 403s (token mismatch)
and no 502s (container down).

## 10. Phase G - Validation

Run full Phase 3 (benign baseline, all tasks):

```
uv run python -m worldsim.main phase 3 --benchmark vendors/webarena-verified
```

Compare per-task pass/fail to the most recent m5 Phase 3 run. Pass: within
+/-2 overall, no site >1 task drift. Likely drift sources: Magento
`downloadable_domains` mis-patch on shopping replicas, per-container OOM,
DB connection pool saturation under concurrent replica load.

## 11. Phase H - Stop/start CLI

New file: `scripts/benchmark_host.sh`.

```
./scripts/benchmark_host.sh start   # ec2 start-instances + wait + health check
./scripts/benchmark_host.sh stop    # ec2 stop-instances
./scripts/benchmark_host.sh status  # print state + count of /init==200
```

Start path: `ec2 start-instances`, poll until running, SSH, wait for docker
containers to be running, then probe the real env-ctrl `/init` endpoints over
SSH until all 200 or 5 min timeout. Elastic IP stays attached while stopped
($3.60/mo line in cost model). The checked-in host config now owns the
advertised host plus bind hosts; scale compose is no longer implicitly
localhost-only.

## 12. Phase I - Decommission m5.xlarge

Only after T+2 weeks clean on r5, no unexplained regressions, final paper
run captured. Then: snapshot m5 root volume to S3, terminate instance,
release its elastic IP.

## 13. Decision gates

| Gate | What to check | Pass | Fail action |
|------|--------------|------|-------------|
| A | `docker info` shows 1TB on /var/lib/docker | proceed | resize/remount |
| B | all 8 shared volumes have `_hydrated_<sha>` sentinel | proceed | re-extract from tar |
| C | `docker compose config` exits 0 | proceed | fix generator |
| D | 30/30 env-ctrl /init == 200 | proceed | triage per-container logs |
| E | 30/30 proxy ports listening | proceed | `journalctl -u nginx` |
| F | Phase 0c smoke: 0 403, 0 502 | proceed | token or routing bug |
| G | Phase 3 diff <= 2 tasks | proceed | do not decommission m5 |

## 14. Testing strategy

Per phase:

- A: `ssh ubuntu@<new-ip> 'df -h /var/lib/docker'` shows ~1TB.
- B: `docker run --rm -v <shared_vol>:/v alpine ls /v/_hydrated_<sha>` per shared vol.
- C: `docker compose -f docker-compose.yml config > /dev/null` parses.
- D: 30 curl POSTs to 30 `/init` endpoints return 200.
- E: 30 curls with `-H "X-Worldsim-Token: ..."` return 200, 30 without return 403.
- F: Phase 0c logs show 30 workers, 0 errors.
- G: diff Phase 3 `summary.json` against baseline, per-task.

## 15. Cost model with math

Fixed (always-on):

- EBS 1TB gp3: $0.08/GB-month * 1000 = $80/mo
- Elastic IP (unattached or attached-while-stopped): $0.005/hr * 720 = $3.60/mo
- S3 Standard-IA 265GB: $0.0125/GB * 265 = $3.30/mo
- **Fixed: $86.90/mo**

Variable (r5.8xlarge compute, on-demand us-east-2):

- $2.016/hr * (3 min start + 30 min run + 5 min stop buffer) = $2.016 * 0.633 = $1.28/run
- 20 runs/mo -> $25.60
- 50 runs/mo -> $64

Totals:

- Light (5 runs/mo): ~$93/mo
- Target (20 runs/mo): ~$113/mo
- Heavy (50 runs/mo): ~$151/mo

Prior plan's $160 fixed -> $87 fixed is the main $73/mo saving.

## 16. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| Magento `downloadable_domains` points at old hostname on new host | high | generator emits a post-start `sed -i` patch step per shopping/shopping_admin replica against the new IP |
| gitlab reconfigure re-triggers on first boot (breaks sentinel) | low | bootstrapped sentinel is in the image layer, not a volume, so should persist, but verify with `docker exec git-0 cat /var/opt/gitlab/bootstrapped` in Phase D |
| Shared RO volume corrupted by buggy `:ro` enforcement | very low | docker enforces ro at the kernel mount level; caught by `_hydrated_<sha>` sentinel drift |
| Port range collision with existing listeners | low | all real ports are 127.0.0.1-bound, proxy ports are 1x-prefixed and contiguous |
| 4 concurrent Magento replicas saturate CPU | medium | mem_limit + watch load in Phase G; fallback is drop shopping to 2 replicas |
| gitlab replica count 8 saturates disk IOPS | medium | gp3 at 6000 IOPS provisioned, monitor with `iostat -x 2`; fallback is bump IOPS to 12000 ($+40/mo) |
| Docker subnet 172.20.0.0/20 collides with VPC | low | check VPC CIDR first; alt is 172.30.0.0/20 |

## 17. Rollback plan

m5 stays untouched through Phase H. If Phase G or early r5 prod use fails:
swap `instances.json` back to the m5 version (keep both committed as
`configs/instances.m5.json` / `instances.r5.json`), resume m5 runs with the
existing proxy token, triage r5 offline. Rollback cost: zero; m5 keeps
running at $140/mo until Phase I.

## 18. Pre-migration cleanup on m5.xlarge

Only after the currently in-flight Phase 3 run completes. Expected reclaim:
152GB from the orphaned volume plus 353GB from downloads if also nuked.

```
# 152GB orphaned openstreetmap-website_db-data volume (0 links, verified)
docker volume rm openstreetmap-website_db-data

# Other orphans (small, safe)
docker volume prune -f
docker builder prune -f

# If disk is still tight and S3 has the tars:
rm -rf /home/ubuntu/downloads /home/ubuntu/wiki
```

Do not run these while Phase 3 is active. Check with
`ls -t logs/*/phase3*/history.json | head -1` on the orchestrator side first.

## 19. Success criteria

Explicit pass bar:

- 30/30 containers running, 30/30 env-ctrl /init returning 200
- 30/30 proxy ports listening, token auth working
- Phase 0c discovers all 30 replicas, dispatches tasks with no 403/502
- Phase 3 against r5 produces success count within +/-2 of the m5 baseline
  for the same task set
- Total monthly cost <= $120 at 20 runs/mo

Any one failing -> do not decommission m5.

## 20. What this plan does NOT do

Scope guards, so nobody expands this into a multi-week project:

- Does not introduce swarm, kubernetes, or ECS. Plain `docker compose` only.
- Does not change WebArena task logic, the tasks JSON, or the reward functions.
- Does not change Phase 4 adversarial strategy code.
- Does not touch Modal sandbox code or the orchestrator worker pool.
- Does not add auto-scaling. Replica counts are static per `scale_config.yml`.
- Does not add monitoring/metrics infra (cloudwatch is already on by default).
- Does not re-architect envctrl or patch container images beyond the
  in-place env-ctrl patcher already in `scripts/wa_envctrl_patcher.py`.
- Does not manage benchmark environment lifecycle during evaluation (only
  the existing `reset_endpoint` per task, per the v5 threat model).

## Key files

Files introduced by this migration:

- `scripts/generate_compose_scale.py` (reads `scale_config.yml`, emits compose + proxy_ports.conf + instances.json fragment)
- `scripts/scale_config.yml` (source of truth for replica counts and port bases)
- `scripts/benchmark_host.sh` (stop/start/status CLI)
- `docs/migration/r5-8xlarge-scale-migration-plan.md` (this file)

Existing files referenced:

- `scripts/restore_benchmark_archives_from_s3.sh` - S3 fetch for Phase B.1
- `scripts/bootstrap_ec2.sh` - base pattern for verify loop; adapt to 30 rows
- `scripts/deploy_benchmark_proxy.sh` - reused as-is with new proxy_ports.conf
- `scripts/proxy_ports.conf` - regenerated from generator
- `scripts/webarena-compose-override.yml` - replaced by generator output
- `scripts/patch_webarena_containers.sh`, `scripts/wa_envctrl_patcher.py` - reused for the env-ctrl base_url Python patch across all 30 containers
- `scripts/configure_db_access.sh` - reused; loop over 30 containers for grants
