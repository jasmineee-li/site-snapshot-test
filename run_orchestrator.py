# run_orchestrator.py
# High-level orchestration: snapshot -> rewrite+inject -> serve -> instruct operator/agent to browse -> collect logs
# Usage: python run_orchestrator.py https://example.com run001 8000
import subprocess, os, signal, time, sys, shlex
from pathlib import Path

def run_cmd(cmd, env=None):
    print(">", cmd)
    p = subprocess.Popen(shlex.split(cmd))
    return p

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_orchestrator.py <url> <run_id> <port>")
        sys.exit(1)
    url = sys.argv[1]
    run_id = sys.argv[2]
    port = sys.argv[3]
    base = Path("snapshots") / run_id
    snapshot_dir = base / "raw"
    bundle_dir = base / "bundle"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1) Snapshot
    p = run_cmd(f"python snapshotter.py {url} {snapshot_dir}")
    p.wait()
    # 2) Rewrite + inject
    p = run_cmd(f"python rewrite_and_inject.py {snapshot_dir} {bundle_dir}")
    p.wait()

    # 3) Start server
    srv = run_cmd(f"python serve_snapshot.py {bundle_dir} {port}")
    print("Server starting, give it 2s...")
    time.sleep(2)

    print("\n==== ORCHESTRATION READY ====")
    print(f"Open the agent or a browser to http://localhost:{port}/")
    print("Let the agent interact with the page. The server will capture file requests and mail intents.")
    print("When done, press Ctrl+C to stop the server and exit (or kill this process).")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        # best-effort terminate server child processes
        srv.terminate()
        sys.exit(0)
