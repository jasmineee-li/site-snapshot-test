---
name: benchmark-host-ops
description: Model-invoked router for WARP Taskgen host lifecycle and remote runs. Use for r8a host setup, host-config and topology selection, proxy deployment or drift, remote_job_start/status/tail, sync_to_host, SSH/SSM launch issues, storage_state or evaluator preflight, and choosing smoke versus scale instances. Use the Phase 4 skill for PVPO, gate routing, TP/VEA, iterator, or judge semantics.
---

# Benchmark host operations

Use the canonical operational docs; this skill contains routing only.

- Read `agent_docs/remote-runs.md` for r8a lifecycle, host-local topology, proxy, locality, remote jobs, artifact handoff, and long-run monitoring.
- Read `agent_docs/verification.md` for preflight, live-gate selection, quiet wrappers, and required evidence.
- Read `agent_docs/secrets.md` before changing credentials, tokens, or auth wiring.
- Inspect the script `--help` and selected host YAML before assuming a flag or field; generated instance, compose, and proxy files are host-local.

Current routing facts:

- `r8a` is the active paper-facing scale/smoke host; use the ignored operator config and generated topology for that host.
- Modal Phase 0c uses smoke/public or proxied instances; on-host Phase 2c and Phase 4 use scale/orchestrator instances.
- `sync_to_host.sh` deploys an accepted checkout; package source on `origin/main` remains authoritative.
- Phase 4 PVPO is runner-owned, page-surface-stable capture; no separate browser endpoint or container is part of the active path.

Completion check: the selected host config, locality-specific instances file, expected artifact source, and verification command are explicit before a remote job starts.
