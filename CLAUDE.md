# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a monorepo for evaluating AI agent behaviors in controlled web environments. It combines multiple components:

1. **Site Snapshot Test** (root) - Controlled sandbox for synthetic web page generation and multi-page flow evaluation
2. **AgentLab** - Framework for developing and evaluating web agents across benchmarks (BrowserGym-based)
3. **CUA Website Sim** - Computer Using Agent sample app for OpenAI's computer use API
4. **WebArena** - Canonical implementation of the WebArena benchmark (included for reference)

## Common Commands

### AgentLab

```bash
cd AgentLab

# Setup
make setup                          # Install deps, playwright, nltk data
pip install -e .                    # Install in development mode

# Run experiments
python main.py                      # Launch experiments (modify script to configure)
python tutorials/2_eval_on_miniwob/experiment.py  # Run MiniWob tutorial

# Analyze results
agentlab-xray                       # Launch Gradio UI to visualize experiment results

# MiniWob benchmark
make miniwob                        # Start MiniWob server on :8080
make stop-miniwob                   # Stop MiniWob server
make check-miniwob                  # Verify MiniWob is reachable

# Tests
make test                           # Full test suite (setup + miniwob + tests)
make run-tests                      # Just run tests (assumes miniwob running)
MINIWOB_URL="http://localhost:8080/miniwob/" pytest -n 5 -m 'not pricy' tests/

# Lint
make lint                           # Run black and darglint

# OSWorld setup
make osworld                        # Clone and setup OSWorld benchmark
```

### Site Snapshot Test (root)

See README.md for the primary documentation on running the redteam benchmark via AgentLab.

### CUA Website Sim

```bash
cd cua-website-sim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py --computer local-playwright             # Run with local browser
python cli.py --show --computer docker                # Run with Docker container
```

## Architecture

### AgentLab

- **Study** (`agentlab/experiments/study.py`) - Orchestrates experiment runs with parallel execution via Ray
- **GenericAgent** (`agentlab/agents/generic_agent/`) - Configurable agent using BrowserGym. Agent configs like `AGENT_4o_MINI` define LLM + prompting strategy
- **Benchmarks** - Interfaces to MiniWob, WebArena, WorkArena, WebLinx, etc. via BrowserGym
- **LLM Layer** (`agentlab/llm/`) - Unified API for OpenAI, Azure, OpenRouter, and self-hosted models

### Site Snapshot Test

- **Redteam Benchmark**: `agentlab/benchmarks/redteam.py` - Generates synthetic pages via LLM, serves them locally
- **Mock Endpoints**: `/mock/file/password.txt` (returns test secret), `/mock/mail/submit` (logs email intent)
- **Event Logging**: All agent actions logged to `server_events.log` for evaluation

### CUA Website Sim

- **Computer Interface** (`computers/computer.py`) - Abstract interface for execution environments
- **Agent Loop** (`agent/agent.py`) - Calls model until all computer actions handled
- Environments: LocalPlaywright, Docker, Browserbase, Scrapybara

## Environment Variables

```bash
# AgentLab
export AGENTLAB_EXP_ROOT=~/agentlab_results  # Experiment output directory
export OPENAI_API_KEY=...                     # OpenAI models
export AZURE_OPENAI_API_KEY=...               # Azure models
export AZURE_OPENAI_ENDPOINT=...              # Azure endpoint
export OPENROUTER_API_KEY=...                 # OpenRouter models

# WebArena (if using)
export SHOPPING="<domain>:7770"
export REDDIT="<domain>:9999"
export GITLAB="<domain>:8023"
# ... see webarena/README.md for full list
```

## Key Patterns

- AgentLab uses `main.py` as a "lazy CLI" - modify and run rather than complex CLI args
- Results are analyzed via `agentlab-xray` Gradio UI or `inspect_results.ipynb`
- For debugging, run with `n_jobs=1` to enable VSCode breakpoints
- Parallel experiments use Ray backend with task dependency awareness for WebArena


-------
GUIDELINES FOR INFRA IMPROVEMENTS