#!/usr/bin/env python3
"""
run_master.py
Master runner for the entire site-snapshot evaluation flow.

Usage:
    python run_master.py synthetic [--port 8000] [--screens screens]
    python run_master.py snapshot [--port 8000] [--screens screens] [--news-url URL]

The script will:
1. Choose the appropriate config file (synthetic_flows_config.example.json or snapshot_flows_config.example.json)
2. Create snapshots if needed (snapshot mode only)
3. Start the flow server in the background
4. Run the simulated agent
5. Display the results and logs
6. Clean up the server process
"""

import argparse
import subprocess
import time
import sys
import signal
import os
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, cwd=None, check=True):
    """Run a command and return the process."""
    print(f">>> {cmd}")
    if isinstance(cmd, str):
        import shlex

        cmd = shlex.split(cmd)

    return subprocess.run(cmd, cwd=cwd, check=check)


def run_cmd_bg(cmd, cwd=None):
    """Run a command in the background and return the process."""
    print(f">>> (background) {cmd}")
    if isinstance(cmd, str):
        import shlex

        cmd = shlex.split(cmd)

    return subprocess.Popen(cmd, cwd=cwd)


def create_snapshot_if_needed(news_url, force=False):
    """Create news snapshot if it doesn't exist or force is True."""
    raw_dir = Path("snapshots/run_news/raw")
    bundle_dir = Path("snapshots/run_news_bundle")

    if bundle_dir.exists() and not force:
        print(f"✓ News bundle already exists at {bundle_dir}")
        return True

    print(f"📸 Creating news snapshot from {news_url}")

    try:
        # Create snapshot
        run_cmd(f"python snapshotter.py {news_url} {raw_dir}")

        # Process snapshot
        run_cmd(f"python rewrite_and_inject_har.py {raw_dir} {bundle_dir}")

        print(f"✓ News snapshot bundle created at {bundle_dir}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create snapshot: {e}")
        return False


def wait_for_server(port, timeout=10):
    """Wait for the server to be ready."""
    import urllib.request
    import urllib.error

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


def show_results(run_dir):
    """Display the results from the server logs."""
    log_path = Path(run_dir) / "server_events.log"

    if not log_path.exists():
        print("❌ No server logs found")
        return

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    import json

    events = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error reading logs: {e}")
        return

    # Analyze events
    page_views = [e for e in events if e.get("event") == "page_view"]
    file_accesses = [e for e in events if e.get("event") == "file_served"]
    mail_intents = [e for e in events if e.get("event") == "mail_submit_intent"]

    print(f"📄 Page views: {len(page_views)}")
    for pv in page_views:
        stage = pv.get("stage", "unknown")
        mode = pv.get("mode", "unknown")
        print(f"   - {stage} ({mode})")

    print(f"📁 Files accessed: {len(file_accesses)}")
    for fa in file_accesses:
        name = fa.get("name", "unknown")
        print(f"   - {name}")

    print(f"📧 Mail intents: {len(mail_intents)}")
    for mi in mail_intents:
        to = mi.get("to", "unknown")
        subject = mi.get("subject", "unknown")
        body_len = mi.get("raw_body_len", 0)
        print(f"   - To: {to}")
        print(f"     Subject: {subject}")
        print(f"     Body length: {body_len} chars")

    print(f"\n📝 Full logs: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for site-snapshot evaluation"
    )
    parser.add_argument(
        "mode",
        choices=["synthetic", "snapshot"],
        help="Run with synthetic pages or snapshot pages",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the flow server (default: 8000)",
    )
    parser.add_argument(
        "--screens",
        default="screens",
        help="Directory for agent screenshots (default: screens)",
    )
    parser.add_argument(
        "--news-url",
        default="https://www.bbc.com/news",
        help="URL to snapshot for news page (snapshot mode only)",
    )
    parser.add_argument(
        "--force-snapshot",
        action="store_true",
        help="Force recreation of snapshots even if they exist",
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)

    print("🚀 Starting site-snapshot evaluation flow")
    print(f"Mode: {args.mode}")
    print(f"Port: {args.port}")
    print(f"Screenshots: {args.screens}")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Choose base config file and create timestamped run directory
    if args.mode == "synthetic":
        base_config_file = "synthetic_flows_config.example.json"
        run_dir = f"snapshots/flow_run_synthetic_{now}"
    else:
        base_config_file = "snapshot_flows_config.example.json"
        run_dir = f"snapshots/flow_run_snapshot_{now}"

        # Create snapshot if needed
        if not create_snapshot_if_needed(args.news_url, args.force_snapshot):
            sys.exit(1)

    if not Path(base_config_file).exists():
        print(f"❌ Config file not found: {base_config_file}")
        sys.exit(1)

    # Create temporary config with our custom run directory
    import json

    with open(base_config_file, "r") as f:
        config = json.load(f)
    config["run_dir"] = run_dir

    temp_config_file = f"temp_config_{now}.json"
    with open(temp_config_file, "w") as f:
        json.dump(config, f, indent=2)

    config_file = temp_config_file

    # Ensure screenshots directory exists
    Path(args.screens).mkdir(exist_ok=True)

    server_process = None
    try:
        # Start the flow server
        print(f"🌐 Starting flow server with {config_file}")
        server_process = run_cmd_bg(f"python serve_flow.py {config_file} {args.port}")

        # Wait for server to be ready
        if not wait_for_server(args.port):
            print("❌ Server failed to start")
            sys.exit(1)

        print(f"✓ Server ready at http://localhost:{args.port}")

        # Run the simulated agent
        print("🤖 Running simulated agent...")
        try:
            run_cmd(
                f"python simulated_agent.py --base http://localhost:{args.port} --screens {args.screens}"
            )
            print("✓ Agent completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Agent failed: {e}")
            # Continue to show results even if agent failed

        # Show results
        show_results(run_dir)

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")

    finally:
        # Clean up server process
        if server_process:
            print("🛑 Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing server...")
                server_process.kill()
                server_process.wait()

        # Clean up temporary config file
        if temp_config_file and Path(temp_config_file).exists():
            Path(temp_config_file).unlink()

    print("✅ Complete!")


if __name__ == "__main__":
    main()
