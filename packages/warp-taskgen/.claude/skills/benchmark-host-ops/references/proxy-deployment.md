# Proxy deployment and drift

## The problem

Benchmark EC2 instances run with default credentials and known-vulnerable software; opening `0.0.0.0/0` on the site ports is unsafe. But Modal sandboxes (Phase 0c verification probing) egress from dynamic IPs the EC2 security group can't allowlist. The fix: an authenticated nginx reverse proxy on offset ports that requires a shared secret header (`X-Worldsim-Token`) on every request. Proxy ports are open to `0.0.0.0/0` because they're token-gated; origin ports stay closed.

## `deploy_benchmark_proxy.sh`

Generates the nginx config from `generate_nginx_config()` (in-script, so it's versioned), copies it to the host, restarts nginx. Idempotent — safe to re-run. If a token file exists it's reused unless `--new-token` is passed.

Typical invocation:

```bash
./scripts/deploy_benchmark_proxy.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --ssh-key ~/.ssh/webarena-key.pem \
  --port-map scripts/proxy_ports.conf \
  --token-file .proxy_token
```

SSH-blocked SG case: `--via-ssm` routes through AWS Systems Manager Session Manager instead of direct SSH. Requires the EC2 instance to have the SSM agent running and the `benchmark-ec2-backup` IAM role (or equivalent SSM-permissive role) attached.

After the script runs, the operator must open the proxy ports in the SG (it prints the exact ports and a JSON snippet for `instances.json`). Per the stored-memory rule, **SG changes are always an explicit per-instance ask** — the script does not authorize ingress on its own.

## Port scheme

`scripts/proxy_ports.conf` is a flat text file, one line per site: `name:real_port:proxy_port`. If `proxy_port` is omitted it defaults to `real_port + PORT_OFFSET` (default 10000). So GitLab origin 7770 → proxy 17770, Reddit origin 9999 → proxy 19999, etc.

## `check_proxy_drift.sh`

Two layers of verification:

**Layer 1 (always runs)** — file-on-disk vs template. Regenerates the nginx config from the script + current port map + token, diffs it against `/etc/nginx/conf.d/worldsim-proxy.conf` on the host, fails if they differ.

**Layer 2 (opt-in via `--verify-runtime`)** — confirms nginx actually loaded the on-disk config:

- `nginx -t` — config is loadable.
- `systemctl is-active nginx` — daemon is up.
- Oldest worker PID start-time vs config file mtime — if workers started *before* the config was last modified, nginx has not reloaded since the edit and the file is on disk but not in memory. Non-destructive: no reload is forced.
- Grep `/var/log/nginx/error.log` for recent `[emerg]` — flags a prior failed reload.

The runtime verify does **not** introspect nginx's loaded-in-memory config directly because open-source nginx exposes no such API. `nginx -T` re-parses from disk and would give a false positive when the running process is holding stale config. The combination above approximates it.

## GitLab `external_url` drift

GitLab ships with a hardcoded `external_url` in `/etc/gitlab/gitlab.rb`. On the proxy port it would emit redirects pointing back to the origin port, breaking the Modal-sandbox-reachable path. Two choices:

1. **Change `gitlab.rb` and run `gitlab-ctl reconfigure` + service restart.** Clean but invasive.
2. **Client-side Location-rewrite via `worldsim/http_proxy.py::ProxyingHTTPAdapter`.** The adapter's `_last_proxy_port_by_host` map rewrites `Location:` headers on redirects so the client follows the proxy port. No host-side changes. This is what ships by default — simpler and survives GitLab reinstalls.

Proxy metadata is carried on each mounted `ProxyingHTTPAdapter` and read via `worldsim.http_proxy.proxy_info_from_session(session, url)`. There are no module-level globals; the editor's cross-origin relaxation is a function of whichever adapter is mounted on the session it holds. If you add a site whose redirects can't be handled client-side, go back to option 1.

## Historical cleanup — Magento directives

Older checkouts of `deploy_benchmark_proxy.sh` contained Magento-specific `proxy_buffer_size` / `proxy_redirect` directives and a `fix_magento_base_url.sh` runbook. These were removed on 2026-04-21 with the WASP-aligned scoping decision. If you see Magento code paths, they're archaeology — do not resurrect.

## What not to do

- Do not hand-edit `/etc/nginx/conf.d/worldsim-proxy.conf`. The script is the source of truth; hand edits silently revert on next deploy or instance rebuild.
- Do not authorize SG ingress without an explicit per-instance ask to the user, even scoped to `/32`. This rule exists because of prior accidents, not abundance of caution.
- Do not commit `.proxy_token`. The filename is in `.gitignore`; if you see it tracked, remove it.
- Do not change `PORT_OFFSET` without coordinating with `instances*.json` — the `reset_endpoint` URLs encode the proxy port and break when the offset shifts.
