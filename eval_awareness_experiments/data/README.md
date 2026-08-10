# Data Directory

Website data for eval awareness experiments, organized by source.

## Directory Structure

```
data/
├── manifest.json          # Auto-generated index of all samples
├── worldsim/              # Extracted from browser-sim results/
│   ├── github/
│   ├── jenkins/
│   ├── jira/
│   └── ...
├── webarena/              # Manually collected from WebArena instances
│   ├── gitlab/
���   ├── reddit/
│   └── ...
├── agentcompany/          # Manually collected from AgentCompany instances
│   ├── rocketchat/
│   ├── gitlab/
│   └── ...
└── real/                  # Scraped from real websites via scraper.py
    ├── github/
    ├── gitlab/
    └── ...
```

## File Naming Convention

Each sample has up to 3 files with the same base name:
- `{name}.html` — Raw HTML source
- `{name}_axtree.txt` — Accessibility tree text (ARIA snapshot)
- `{name}.png` — Screenshot (1280x720 viewport)

## Data Collection

### WorldSim (automated)
```bash
python -m eval_awareness_experiments.extract_worldsim --results-dir results/ --output eval_awareness_experiments/data/worldsim
```

### Real Websites (automated)
```bash
python -m eval_awareness_experiments.scraper --presets github,gitlab,paypal --output eval_awareness_experiments/data/real
```

### WebArena / AgentCompany (manual)
1. Start the benchmark's hosted environment
2. Navigate to representative pages
3. Save HTML (`Ctrl+S` or browser dev tools), screenshot, and AXTree
4. Place files in the appropriate `data/{source}/{website_type}/` directory
5. Follow the naming convention above

## Rebuilding the Manifest

After adding new data, regenerate the manifest:
```bash
python -m eval_awareness_experiments.extract_worldsim --manifest eval_awareness_experiments/data/manifest.json
```
