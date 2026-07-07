# Importing Benchmark Data

Instructions for extracting website data from each benchmark source into the standardized format.

Each source produces per-environment directories containing `{page}.html`, `{page}.png`, and `{page}_axtree.txt` files, plus a manifest entry in `data/manifest.json`.

## WebArena-Infinity

Self-contained JS web apps that simulate real products (Gmail, GitLab, PayPal, etc.). Each app has its own HTTP server.

**Available apps:**
```bash
ls external_benchmarks/webarena-infinity/apps/
```

Currently configured for extraction: `gmail`, `gmail-accounts-and-contacts`, `gitlab-plan-and-track`, `paypal-my-wallet`.

**Run extraction:**
```bash
# All configured apps
conda run -n worldsim python -m eval_awareness_experiments.extract_webarena_infinity

# Specific apps
conda run -n worldsim python -m eval_awareness_experiments.extract_webarena_infinity --apps gmail,paypal-my-wallet
```

**Adding a new app:**

1. Check what views/routes the app supports by reading its `js/app.js` and `js/views.js`. Look for hash routes (`#/inbox`, `#/issues`) or sidebar nav elements (`data-view="..."`, `data-action="navigate"`).

2. Add an entry to `APP_PAGES` in `extract_webarena_infinity.py`:
   ```python
   "new-app-name": [
       ("view-name", "#/hash-route"),          # for hash-routed apps
       ("view-name", "click:[data-view='x']"), # for click-navigated apps
   ],
   ```

3. Run the extractor with `--apps new-app-name`.

**Notes:**
- Apps use SSE for state sync, so the extractor uses `wait_until="load"` instead of `"networkidle"`.
- Viewport is 1280x720, headless Chromium.
- Each app's server is started automatically on ports 9100+ and stopped after extraction.

## Real Websites

Scraped from live websites using Playwright.

```bash
# All presets (github, gitlab, paypal)
conda run -n worldsim python -m eval_awareness_experiments.scraper

# Specific presets
conda run -n worldsim python -m eval_awareness_experiments.scraper --presets github,gitlab

# Single URL
conda run -n worldsim python -m eval_awareness_experiments.scraper --url https://example.com --name example_home
```

See `PRESET_URLS` in `scraper.py` to add new presets.

## WorldSim

Extracted from browser-sim `results/` directory. These are LLM-generated web pages from the redteam pipeline.

```bash
conda run -n worldsim python -m eval_awareness_experiments.extract_worldsim --results-dir results/ --output eval_awareness_experiments/data/worldsim
```

## TheAgentCompany (TAC)

Agent trajectory data from TheAgentCompany benchmark runs. Imported via the old `eval_awareness` pipeline — see `eval_awareness/import_tac.py`.

Not yet integrated into the `eval_awareness_experiments` format.
