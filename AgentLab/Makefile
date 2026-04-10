.PHONY: test setup miniwob lint stop-miniwob osworld \
       generate-apps generate-apps-resume validate-manifest \
       redteam-test smoke-test xray status

setup:
	@uv sync
	@uv run playwright install chromium --with-deps
	@uv run python -c 'import nltk; nltk.download("punkt_tab")'

miniwob: stop-miniwob
	@git clone https://github.com/Farama-Foundation/miniwob-plusplus.git || true
	@cd miniwob-plusplus && git checkout 7fd85d71a4b60325c6585396ec4f48377d049838
	@uv run python -m http.server 8080 --directory miniwob-plusplus/miniwob/html & echo $$! > .miniwob-server.pid
	@sleep 3
	@echo "MiniWob server started on http://localhost:8080"

check-miniwob:
	@curl -I "http://localhost:8080/miniwob/" || (echo "MiniWob not reachable" && exit 1)
	@echo "MiniWob server is reachable"

stop-miniwob:
	@kill -9 `cat .miniwob-server.pid` || true
	@rm -f .miniwob-server.pid
	@echo "MiniWob server stopped"

run-tests:
	@MINIWOB_URL="http://localhost:8080/miniwob/" uv run pytest -n 5 --durations=10 -m 'not pricy' tests/
	@echo "Tests completed"

test: setup miniwob check-miniwob run-tests stop-miniwob

lint: setup
	@uv run black src/ --check --diff
	@uv run darglint -v 2 -z short src/

osworld:
	@echo "Setting up OSWorld..."
	@git clone https://github.com/xlang-ai/OSWorld || true
	@echo "Modifying OSWorld requirements.txt to remove pinned versions..."
	@cd OSWorld && \
		sed -i.bak 's/numpy~=.*/numpy/' requirements.txt && \
		sed -i.bak 's/torch~=.*/torch/' requirements.txt && \
		sed -i.bak 's/torch$$/torch/' requirements.txt && \
		sed -i.bak 's/tqdm~=.*/tqdm/' requirements.txt && \
		sed -i.bak 's/pandas~=.*/pandas/' requirements.txt
	@echo "Installing OSWorld requirements..."
	@cd OSWorld && uv pip install -r requirements.txt
	@echo "Installing OSWorld in development mode..."
	@cd OSWorld && uv pip install -e .
	@echo "OSWorld setup completed!"
	@echo "Next steps:"
	@echo "1. Configure your VM (VMware/VirtualBox) according to OSWorld documentation"
	@echo "2. Download or set up the Ubuntu VM image"
	@echo "3. Run AgentLab with OSWorld tasks"

# ---------------------------------------------------------------------------
# Redteam workflow targets
# ---------------------------------------------------------------------------

BENCHMARK_FILE ?=
PLATFORM_MANIFEST ?=
OUTPUT_DIR ?= ../apps
PARALLEL ?= 1
GENERATE_FLAGS ?=

generate-apps:
	@if [ -z "$(BENCHMARK_FILE)" ]; then \
		echo "BENCHMARK_FILE is required. Example: make generate-apps BENCHMARK_FILE=/path/to/behaviors.json [PLATFORM_MANIFEST=/path/to/platform_manifest.json]"; \
		exit 1; \
	fi
	@if [ ! -f "$(BENCHMARK_FILE)" ]; then \
		echo "Benchmark file not found: $(BENCHMARK_FILE)"; \
		exit 1; \
	fi
	@if [ -n "$(PLATFORM_MANIFEST)" ] && [ ! -f "$(PLATFORM_MANIFEST)" ]; then \
		echo "Platform manifest not found: $(PLATFORM_MANIFEST)"; \
		exit 1; \
	fi
	@echo "Generating apps from $(BENCHMARK_FILE) → $(OUTPUT_DIR)"
	@uv run python -m agentlab.benchmarks.redteam.generate_apps \
		--benchmark-file $(BENCHMARK_FILE) \
		--output-dir $(OUTPUT_DIR) \
		$(if $(strip $(PLATFORM_MANIFEST)),--platform-manifest $(PLATFORM_MANIFEST)) \
		--parallel $(PARALLEL) \
		$(GENERATE_FLAGS)

generate-apps-resume:
	@if [ -z "$(BENCHMARK_FILE)" ]; then \
		echo "BENCHMARK_FILE is required. Example: make generate-apps-resume BENCHMARK_FILE=/path/to/behaviors.json [PLATFORM_MANIFEST=/path/to/platform_manifest.json]"; \
		exit 1; \
	fi
	@if [ ! -f "$(BENCHMARK_FILE)" ]; then \
		echo "Benchmark file not found: $(BENCHMARK_FILE)"; \
		exit 1; \
	fi
	@if [ -n "$(PLATFORM_MANIFEST)" ] && [ ! -f "$(PLATFORM_MANIFEST)" ]; then \
		echo "Platform manifest not found: $(PLATFORM_MANIFEST)"; \
		exit 1; \
	fi
	@echo "Resuming app generation..."
	@uv run python -m agentlab.benchmarks.redteam.generate_apps \
		--benchmark-file $(BENCHMARK_FILE) \
		--output-dir $(OUTPUT_DIR) \
		$(if $(strip $(PLATFORM_MANIFEST)),--platform-manifest $(PLATFORM_MANIFEST)) \
		--parallel $(PARALLEL) \
		--resume \
		$(GENERATE_FLAGS)

validate-manifest:
	@if [ -z "$(PLATFORM_MANIFEST)" ]; then \
		echo "PLATFORM_MANIFEST is required. Example: make validate-manifest PLATFORM_MANIFEST=/path/to/platform_manifest.json [BENCHMARK_FILE=/path/to/behaviors.json]"; \
		exit 1; \
	fi
	@if [ ! -f "$(PLATFORM_MANIFEST)" ]; then \
		echo "Platform manifest not found: $(PLATFORM_MANIFEST)"; \
		exit 1; \
	fi
	@if [ -n "$(BENCHMARK_FILE)" ] && [ ! -f "$(BENCHMARK_FILE)" ]; then \
		echo "Benchmark file not found: $(BENCHMARK_FILE)"; \
		exit 1; \
	fi
	@uv run agentlab-redteam-validate-platform-manifest \
		--platform-manifest $(PLATFORM_MANIFEST) \
		$(if $(strip $(BENCHMARK_FILE)),--benchmark-file $(BENCHMARK_FILE))

redteam-test:
	@uv run pytest tests/benchmarks/test_redteam_*.py -v --durations=10

smoke-test:
	@uv run pytest tests/benchmarks/test_redteam_config.py \
		tests/benchmarks/test_redteam_data_validator.py \
		tests/benchmarks/test_redteam_routing.py \
		tests/benchmarks/test_redteam_analysis.py \
		tests/benchmarks/test_redteam_app_artifacts.py \
		-v --durations=5

xray:
	@uv run agentlab-xray

status:
	@uv run python -m agentlab.benchmarks.redteam status --output-dir $(OUTPUT_DIR)
