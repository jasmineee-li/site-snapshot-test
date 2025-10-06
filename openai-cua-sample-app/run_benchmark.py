#!/usr/bin/env python3
"""
run_benchmark.py
Benchmark-driven runner for agent evaluation.

Usage:
    python run_benchmark.py benchmarks/v1.json [--case 0] [--port 8000]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from benchmarks.loader import BenchmarkLoader
from generation.page_provider import PageProvider
from generation.page_cache import PageCacheManager
from generation.synthetic_generator import SyntheticPageGenerator
from generation.prefill_analyzer import PrefillDataAnalyzer


def wait_for_server(port: int, timeout: int = 10) -> bool:
    """Wait for the server to be ready."""
    url = f"http://localhost:{port}/mode"
    for i in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except urllib.error.URLError:
            if i == 0:
                print("⏳ Waiting for server to start...")
            time.sleep(1)
    return False


def run_benchmark_case(case, port, run_dir):
    """
    Run a single benchmark case.

    Steps:
    1. Analyze behavior to determine prefill data requirements
    2. Generate prefill data (emails, calendar events, etc.)
    3. Generate/fetch all required pages WITH prefill data embedded
    4. Create dynamic flow config
    5. Start flow server
    6. Run LLM agent with the behavior as goal
    7. Collect results
    """

    print(f"\n{'='*60}")
    print(f"BENCHMARK CASE")
    print(f"{'='*60}")
    print(f"Behavior: {case.behavior}")
    print(f"Pages: {[p.domain for p in case.pages]}")
    print(f"Tools: {case.tools}")
    print(f"{'='*60}\n")

    # Ensure run directory exists
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # 1. Analyze behavior for prefill data requirements
    # print("📊 Step 1: Analyzing prefill data requirements...")
    # analyzer = PrefillDataAnalyzer(model="gpt-5")
    # prefill_spec = analyzer.analyze(case.behavior, case.pages)

    # print(f"  ✓ Prefill spec generated:")
    # print(f"    - {len(prefill_spec.entities)} entity type(s) needed")
    # print(f"    - Required tools: {prefill_spec.required_tools}")

    # # Save prefill spec for debugging
    # spec_path = Path(run_dir) / "prefill_spec.json"
    # spec_dict = prefill_spec.to_dict()
    # spec_path.write_text(json.dumps(spec_dict, indent=2, default=str), encoding="utf-8")
    # print(f"  ✓ Saved: {spec_path}")

    # # 3. Generate/fetch pages WITH prefill spec (scaffold-first)
    # print("\n🌐 Step 3: Generating/fetching pages...")
    # cache = PageCacheManager()
    # generator = SyntheticPageGenerator(model="gpt-5")
    # provider = PageProvider(cache, generator)

    # stages = {}
    # for i, page in enumerate(case.pages):
    #     # Extract stage name from domain (gmail.com -> gmail)
    #     stage_name = page.domain.split(".")[0]

    #     print(f"  Processing page {i+1}/{len(case.pages)}: {page.domain} ({page.mode})")

    #     if page.mode == "synthetic":
    #         # Ensure page exists (generate if needed) WITH prefill spec (scaffold)
    #         html = provider.get_page(page, case.behavior, prefill_data=prefill_spec)

    #         # Save the generated page to a specific location for this run
    #         page_path = Path(run_dir) / f"{stage_name}.html"
    #         page_path.write_text(html, encoding="utf-8")
    #         print(f"    ✓ Saved: {page_path}")

    #         stages[stage_name] = {"mode": "synthetic"}
    #     else:
    #         # Use snapshot bundle
    #         # Note: For snapshots, prefill data would need to be injected via rewrite
    #         bundle = provider.get_page(page, case.behavior, prefill_data=None)
    #         stages[stage_name] = {"mode": "snapshot", "bundle_dir": bundle}

    # # 4. Create dynamic flow config
    # print("\n⚙️  Step 4: Creating flow config...")
    # config = {"run_dir": run_dir, "stages": stages}
    # config_path = Path(run_dir) / "flow_config.json"
    # config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    # print(f"  ✓ Config saved: {config_path}")

    # 5. Start flow server (background)
    config_path = "runs/v1_case0_2025-10-05_15-56-00/flow_config.json"
    port = 8000
    print("\n🚀 Step 5: Starting flow server...")
    server_proc = subprocess.Popen(
        ["python", "serve_flow.py", str(config_path), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server
    if not wait_for_server(port):
        print("❌ Server failed to start")
        out, err = server_proc.communicate(timeout=2)
        print(err.decode() if isinstance(err, bytes) else err)
        server_proc.terminate()
        return {"success": False, "error": "Server failed to start"}

    print(f"  ✓ Server ready at http://localhost:{port}")

    # 6. Run LLM agent
    print("\n🤖 Step 6: Running LLM agent...")
    print(f"  Goal: {case.behavior}")
    print(f"  Tools: {', '.join(case.tools)}")

    try:
        result = subprocess.run(
            [
                "python",
                "agent/llm_executor.py",
                "--goal",
                case.behavior,
                "--base-url",
                f"http://localhost:{port}",
                "--run-dir",
                run_dir,
                "--allowed-tools",
                ",".join(case.tools),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        print("  ✓ Agent completed successfully")
        if result.stdout:
            print(f"\nAgent output:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"  ❌ Agent failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")

    finally:
        # Clean up server
        print("\n🛑 Step 7: Cleaning up...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("  ⚠️  Force killing server...")
            server_proc.kill()
            server_proc.wait()
        print("  ✓ Server stopped")

    # 7. Return results
    return {"run_dir": run_dir, "success": True}


def main():
    parser = argparse.ArgumentParser(description="Run agent benchmarks")
    parser.add_argument("benchmark", help="Path to benchmark JSON file")
    parser.add_argument(
        "--case", type=int, help="Run specific case index (default: run all)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for flow server")
    args = parser.parse_args()

    # Load benchmark
    print("📚 Loading benchmark...")
    loader = BenchmarkLoader()
    benchmark = loader.load(args.benchmark)

    print(f"✓ Loaded benchmark: {benchmark.name}")
    print(f"  Description: {benchmark.description}")
    print(f"  Cases: {len(benchmark.cases)}")

    # Determine which cases to run
    cases_to_run = (
        [benchmark.cases[args.case]] if args.case is not None else benchmark.cases
    )

    # Run each case
    for i, case in enumerate(cases_to_run):
        case_index = args.case if args.case is not None else i

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = f"runs/{benchmark.name}_case{case_index}_{now}"

        result = run_benchmark_case(case, args.port, run_dir)

        print(f"\n{'='*60}")
        print(f"✅ Case {case_index} complete")
        print(f"📁 Results: {run_dir}")
        print(f"{'='*60}\n")

        # Brief delay between cases
        if i < len(cases_to_run) - 1:
            time.sleep(2)

    print("🎉 All benchmark cases complete!")


if __name__ == "__main__":
    main()
