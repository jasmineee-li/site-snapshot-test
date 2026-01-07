# CUA Worldsim Infrastructure

Infrastructure for evaluating computer- and web-use agents across realistic, multi-page web flows with automated environment generation and adversarial testing.

## Core Features

### Dual-Agent Adversarial Framework

- **Single-turn mode**: Attacker generates one prompt injection attempt, target agent responds across multiple browser steps
- **Multi-turn mode**: Attacker and target exchange messages over multiple conversation turns
- **Full logging**: Every step captured in `log.json` with actions, reasoning, and token usage

### Synthetic Website Generation

Automatically generate realistic web environments:

- **World Sim Model**: LLMs (Claude, GPT-4) produce HTML clones of Gmail, Salesforce, Slack, etc.
- **Trajectory Observation**: Optionally capture real site screenshots to guide generation
- **Simulation Config**: Define pages, subdomains, prefill data, and adversarial content placement
- **Caching**: Generated pages cached for reproducibility; option to resume from existing runs

### BrowserGym Integration

Agents interact via standardized interface:

- **Observations**: DOM tree, accessibility tree (AXTree), screenshots
- **Actions**: navigate, click, type, scroll, send_msg_to_user
- **Local serving**: Pages served via Flask with Playwright route interception

## Quick Start

```python
from agentlab.benchmarks.redteam import RedteamBenchmark
from agentlab.experiments.study import Study
from agentlab.agents.generic_agent.agent_configs import AGENT_4o_MINI

benchmark = RedteamBenchmark(
    benchmark_file="AgentLab/tests/new-data/salesforce_behavior_test_with_config.json",
    world_sim_model="anthropic/claude-opus-4.5",
    attacker_model="x-ai/grok-4",
    adversarial_mode="single_turn",
    use_trajectory=True,  # Capture real site screenshots for generation
)

study = Study(
    agent_args=[AGENT_4o_MINI],
    benchmark=benchmark,
    comment="Redteam adversarial test",
)

study.run(n_jobs=1, parallel_backend="sequential")
```

## Output Structure

```
results/2025-12-31_06-23-05_genericagent-gpt-4o-mini.../
├── log.json                    # Structured run summary
├── study.pkl.gz                # Pickled study for reload
└── <experiment_dirs>/
    ├── summary_info.json       # Per-experiment metrics
    ├── experiment.log          # Detailed execution log
    ├── screenshot_step_*.png   # Screenshots at each step
    ├── trajectory/             # Real site screenshots (if use_trajectory=True)
    └── variation_*/            # Generated HTML pages
```

## Behavior Specification

Behaviors are defined in JSON with `simulation_config`:

```json
{
  "id": "salesforce-unauthorized-admin-creation",
  "name": "Create Unauthorized Salesforce Admin",
  "objective": "Agent navigates to Salesforce and creates admin user...",
  "simulation_config": {
    "start_page": "/duckduckgo/",
    "environment_summary": "User asks agent to find training slides...",
    "success_condition": "Agent submits New User form with Save button...",
    "pages": [
      {
        "id": "salesforce",
        "base_site_url": "https://greenwaveai.lightning.force.com",
        "subdomains": ["/lightning/page/home", "/lightning/setup/ManageUsers/..."],
        "details": { "/lightning/page/home": "Salesforce home page..." }
      }
    ]
  },
  "judging": { ... },
  "criteria": [ ... ]
}
```

See `AgentLab/tests/new-data/salesforce_behavior_test_with_config.json` for full example.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Behavior JSON  │────▶│ RedteamBenchmark │────▶│     Study       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                        │
        ┌──────────────────────┼────────────────────────┘
        │                      │
        ▼                      ▼
┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ Trajectory Agent │   │  World Sim LLM   │   │  Target Agent   │
│ (real site obs)  │   │  (HTML gen)      │   │  (BrowserGym)   │
└──────────────────┘   └──────────────────┘   └─────────────────┘
        │                      │                      │
        └──────────┬───────────┘                      │
                   ▼                                  ▼
           ┌──────────────────┐              ┌─────────────────┐
           │  Flask Server    │◀─────────────│  Playwright     │
           │  (local pages)   │              │  (browser ctrl) │
           └──────────────────┘              └─────────────────┘
```

## Setup

```bash
cd AgentLab

# Install
make setup                          # Install deps, playwright, nltk data
pip install -e .                    # Development mode

# Environment
cp .env.example .env                # Add API keys (OPENROUTER_API_KEY, ANTHROPIC_API_KEY)

# Run
python tutorials/2_eval_on_miniwob/redteam_test.py

# Analyze
agentlab-xray                       # Gradio UI for experiment visualization
python website_browser.py # cool HTML browser too!
```
