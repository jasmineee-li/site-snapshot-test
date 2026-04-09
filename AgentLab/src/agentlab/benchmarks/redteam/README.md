# Redteam Benchmark Package

Security testing benchmark that runs AI agents against controlled web environments with embedded adversarial content (prompt injections). The package now uses app-mode environments exclusively.

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | `Page`, `RedteamEnvArgs`, and normalized behavior-field parsing |
| `benchmark.py` | `RedteamBenchmark` — resolves `behavior_id -> app_id`, loads app + behavior contracts, and admits only app-ready/behavior-compatible shared apps |
| `app_pipeline.py` | App-mode generation: Claude Code orchestration, variant generation, validation |
| `env.py` | `RedteamEnv` — BrowserGym environment wrapper for per-app runtimes and route interception |
| `templates/server.py` | Per-app `SimpleHTTPRequestHandler` server template for app mode |
| `routing.py` | Playwright routing: explicit domain bindings with fail-closed blocking for unlisted domains |
| `analysis.py` | `analyze_results()` — computes safety metrics (attack success rate, harmful intent rate) from paired benign/adversarial runs |
| `utils.py` | Shared utilities: route normalization, HTML stripping, filename helpers |

## Mock Endpoints

The server exposes mock endpoints for evaluating agent actions:

- `GET /mock/file/password.txt` — returns a test secret (evaluates data exfiltration)
- `POST /mock/mail/submit` — logs email intent (evaluates social engineering compliance)

## Event Logging

All agent actions are logged to `server_events.log` in the experiment directory. The judge system reads this log to evaluate what the agent did server-side, independent of browser observations.

Each run also archives normalized attack metadata to `attack_metadata.json` inside the experiment directory. Post-run judging and reporting prefer this immutable snapshot over the live app tree so historical runs remain reproducible after app regeneration.

## State Protocol

The browser owns all application state natively:

- **Browser -> Server**: On every mutation, the app pushes structured state via `PUT /api/state`
- **Server -> Judge**: Judges query `GET /api/state` to check agent actions without touching the browser
- **Reset**: `POST /api/reset` restores the immutable seed snapshot and notifies the browser via SSE

## Contracts

Shared app-mode runs use three authoritative inputs:

- Top-level ``platform_manifest.json`` maps each ``behavior_id`` to an ``app_id`` and declares behavior-level routing/domain binding metadata.
- ``apps/{app_id}/app_manifest.json`` is the app-owned v3 contract. It owns app facts only: ``app_id``, ``platform``, ``docs_path``, required ``docs_snapshot``, shared seed provenance, app pages/start route, and app readiness metadata.
- ``apps/{app_id}/behaviors/{behavior_id}.json`` is the behavior-owned contract. It owns ``safe_behavior``, ``success_condition``, ``entry_route``, ``allowed_routes``, ``domain_bindings``, compatibility state, attack metadata, variant lineage, and ``active_variant``.

Contract hardening rules:

- ``app_id``, ``behavior_id``, ``active_variant``, and lineage ``variants[].name`` must be safe single path components. Absolute paths, separators, and dot segments are rejected.
- ``docs_path`` must stay repository-relative and under ``apps/user-manuals/``.
- Runtime materialization rejects symlinked public assets and symlinked variant ``data.js`` inputs.

App manifests intentionally do not persist behavior-owned legacy fields such as ``doc``, ``target``, or ``attack_channels``. Those remain benchmark-input or behavior-contract concerns.

Benchmark admission is split:

- ``app_ready``: app manifest v3 is valid, trusted ``server.py`` is intact, app validation passed, benign readiness suites passed, and required runtime assets exist.
- ``behavior_compatible``: the behavior contract is present, ``compatibility_status == "passed"``, seed provenance matches the app contract, and the active variant lineage is valid.

Default benchmark evaluation runs exactly two conditions per behavior: ``benign`` and the behavior contract's ``active_variant``.

The hardened pipeline treats generated verifier, sanity-check, and audit artifacts as untrusted. Claude generation, sanity checks, and verifier replay run through the shared execution backend, which defaults to Modal Sandboxes. ``server.py`` remains a trusted template-owned runtime file and is checksum-enforced after Claude edits.

At rest, ``benign/data.js`` is canonical and ``js/data.js`` is derived only for runtime materialization. Behavior variants are namespaced: ``adversarial_{behavior_id}_v0`` and follow-on hardening rounds.

The ``generate_apps`` CLI now distinguishes:

- ``--resume``: skip only successfully generated benchmark-ready apps.
- ``--skip-existing``: skip any app directory that already has ``app_manifest.json``.
- ``--platform-manifest``: group behaviors into shared app generations keyed by ``app_id``.
- ``--no-functional-tests``: maintainer-only draft generation escape hatch. These outputs are marked non-admissible and cannot produce a successful manifest.

Authoring support:

- ``agentlab-redteam-validate-platform-manifest --platform-manifest ... --benchmark-file ...`` validates app entries, duplicate/missing behavior mappings, required routing fields, and docs-path presence before generation.
- Wave-1 curation/manual workflow is documented in [docs/wave1-platform-manifest-and-manuals.md](/Users/ashtonchew/projects/browser-sim/docs/wave1-platform-manifest-and-manuals.md).

``pregenerated_env_dir`` points at benchmark-ready shared app directories (``apps/{app_id}``). The benchmark loader also needs ``platform_manifest.json`` plus one behavior contract per evaluated behavior.

## Usage

```python
from agentlab.benchmarks.redteam import RedteamBenchmark

benchmark = RedteamBenchmark(
    benchmark_file="path/to/behaviors.json",
    world_sim_model="anthropic/claude-opus-4.6",
    attacker_model="x-ai/grok-4",
    adversarial_mode="single_turn",
)
```

See the root `README.md` for full quick start, behavior specification format, and output structure.
