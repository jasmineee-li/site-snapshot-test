# Eval Awareness Experiments - Setup & Progress

## What We're Doing

Building a dataset of website data (HTML, screenshots, accessibility trees) and agent trajectories from multiple benchmarks. The goal is to test whether LLMs can distinguish real websites from simulated/benchmark ones.

## Data Sources

### 1. WebArena-Infinity (DONE)

**Pages extracted:** Gmail, Gmail Accounts & Contacts, GitLab Plan & Track, PayPal My Wallet — 32 pages total with HTML + screenshot + axtree per page.

**Trajectories imported:** 20 trajectories (5 per app) from HuggingFace dataset `webarena-x/webarena-infinity-trajectories`, covering gemini/kimi/qwen agents.

```bash
# Extract pages (starts each app's server, captures via Playwright)
conda run -n worldsim python -m eval_awareness_experiments.extract_webarena_infinity

# Import trajectories from HuggingFace
conda run -n worldsim python -m eval_awareness_experiments.import_webarena_infinity_trajectories --max-per-app 5
```

More apps available: `elation-clinical-records`, `figma-slides`, `superhuman-general`, `xero-invoicing`, etc. See `extract_webarena_infinity.py` APP_PAGES dict to add them.

### 2. TheAgentCompany (IN PROGRESS)

TAC uses real Docker services (GitLab, RocketChat, OwnCloud, Plane) with pre-populated seed data. We want to capture page data directly from these services.

**Trajectories:** Already available at `external_benchmarks/tac-experiments/evaluation/1.0.0/` — 19 agent runs with screenshots and trajectories. No importer written yet for the new format.

**Pages:** Need to run the Docker services and use Playwright to capture. Must be done on a **Linux amd64 machine** — the Docker images crash on Mac ARM (MongoDB gets "Illegal instruction" under Rosetta).

#### Setting Up TAC Docker Services (Linux)

**Prerequisites:**
- Docker and Docker Compose installed
- ~10 GB disk for Docker images
- Recommended: 4+ vCPU, 16 GB RAM

**Step 1: Add hostname**
```bash
echo "127.0.0.1 the-agent-company.com" | sudo tee -a /etc/hosts
```

**Step 2: Start services**
```bash
cd external_benchmarks/TheAgentCompany/servers
GITLAB_PORT=8929 docker compose up -d
```

This starts:
| Service | URL | Login |
|---------|-----|-------|
| GitLab | http://the-agent-company.com:8929 | `root@local` / `theagentcompany` |
| RocketChat | http://the-agent-company.com:3000 | `theagentcompany` / `theagentcompany` |
| OwnCloud | http://the-agent-company.com:8092 | `theagentcompany` / `theagentcompany` |

Note: Plane requires a separate setup (see `servers/plane/README.md`). Skipping for now.

**Step 3: Wait for services to be healthy**

GitLab takes several minutes to boot. Check readiness:
```bash
# Check container status
docker ps

# Test each service
curl -s -o /dev/null -w "%{http_code}" http://the-agent-company.com:8929  # GitLab
curl -s -o /dev/null -w "%{http_code}" http://the-agent-company.com:3000  # RocketChat
curl -s -o /dev/null -w "%{http_code}" http://the-agent-company.com:8092  # OwnCloud
```

All should return 200 (or 302 redirect) when ready.

**Step 4: Set up conda env and extract pages**
```bash
conda create -n worldsim python=3.12 -y
conda run -n worldsim pip install playwright huggingface_hub
conda run -n worldsim python -m playwright install chromium
```

**Step 5: Run extraction**

TODO: Write `extract_tac.py` — similar to `extract_webarena_infinity.py` but navigates GitLab/RocketChat/OwnCloud pages. The services have real pre-populated data (repos, chat channels, users, files).

Pages to capture per service:
- **GitLab:** Projects list, project overview, issues, merge requests, CI/CD, user profile
- **RocketChat:** Channel list, general channel, direct messages, admin panel
- **OwnCloud:** File browser, shared files, settings

**Step 6: Stop services when done**
```bash
cd external_benchmarks/TheAgentCompany/servers
GITLAB_PORT=8929 docker compose down
```

### 3. Real Websites (NOT STARTED)

Scraper exists at `eval_awareness_experiments/scraper.py` with presets for GitHub, GitLab, PayPal.

```bash
conda run -n worldsim python -m eval_awareness_experiments.scraper
```

### 4. WorldSim (NOT STARTED)

Extractor exists at `eval_awareness_experiments/extract_worldsim.py`. Pulls from browser-sim `results/` directory.

```bash
conda run -n worldsim python -m eval_awareness_experiments.extract_worldsim --results-dir results/
```

## Data Format

Each source gets a directory under `eval_awareness_experiments/data/`:
```
data/
├── manifest.json                    # Index of all environments
├── webarena-infinity/
│   ├── gmail/
│   │   ├── inbox.html               # Per-page HTML
│   │   ├── inbox.png                # Per-page screenshot (1280x720)
│   │   ├── inbox_axtree.txt         # Per-page accessibility tree
│   │   ├── settings.html
│   │   └── trajectories/            # Agent interaction traces
│   │       ├── kimi_task_m16/
│   │       │   ├── history.json
│   │       │   ├── result.json
│   │       │   └── screenshots/
│   │       └── ...
│   ├── gitlab-plan-and-track/
│   └── ...
├── tac/                             # (TODO)
└── real/                            # (TODO)
```

Manifest entries:
```json
{
  "id": "webarena-infinity_gmail",
  "source": "webarena-infinity",
  "label": "synthetic",
  "path": "eval_awareness_experiments/data/webarena-infinity/gmail",
  "pages": ["inbox", "starred", "sent", "drafts", ...],
  "trajectories": [...],
  "metadata": {"benchmark": "webarena-infinity", "app": "gmail"}
}
```

## Transcript Rendering (TODO)

Trajectories are stored in raw benchmark format. Different sources need different renderers to convert to a judge-friendly transcript. Dispatch on `(source, agent)`:

| Source | Agent | Format |
|--------|-------|--------|
| webarena-infinity | kimi | `{thought, actions}` per step |
| webarena-infinity | gemini | browser-use format: `{model_output, result, state}` |
| webarena-infinity | qwen | similar to kimi |
| tac | openhands | OpenHands events: `{source, action, args}` |
| tac | muse | plain text logs |
| tac | tte | JSON trajectory |

Prompt templates for each agent's scaffolding are in `external_benchmarks/webarena-infinity/evaluation/vision_agents.py`.

## Open Questions

- How to render transcripts for the judge — assistant-only vs full conversation? System prompts give away that it's an eval (e.g., `localhost:8001`).
- Whether to include screenshots in transcripts or text-only.
- Standard transcript format across sources, or source-specific rendering at eval time.
