#!/usr/bin/env python3
# serve_flow.py
# Unified multi-stage server for News → Drive → Mail flows.
# Each stage can run in either 'snapshot' mode (serve a bundled snapshot)
# or 'synthetic' mode (serve a locally-rendered high-fidelity replica).
#
# Usage: python serve_flow.py flows_config.json 8000
# deps: pip install flask

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from flask import Flask, send_from_directory, request, jsonify, redirect


app = Flask(__name__)


# Global state initialized from config
CONFIG: Dict[str, Any] = {}
RUN_DIR: str | None = None
STAGES = ("news", "drive", "mail")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Normalize stage keys/lowercase
    stages = cfg.get("stages", {})
    norm = {}
    for k in STAGES:
        v = stages.get(k, {})
        mode = (v.get("mode") or "synthetic").lower()
        norm[k] = {
            "mode": mode,
            "bundle_dir": v.get("bundle_dir"),
        }
    cfg["stages"] = norm
    return cfg


def ensure_run_dir(base_dir: str) -> str:
    p = Path(base_dir)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def log_event(ev: Dict[str, Any]) -> None:
    if RUN_DIR is None:
        return
    ev["ts"] = datetime.utcnow().isoformat() + "Z"
    # minimal header snapshot with low PII
    hdrs = {
        "User-Agent": request.headers.get("User-Agent"),
        "Referer": request.headers.get("Referer"),
    }
    ev["headers"] = hdrs
    ev["client_ip"] = request.remote_addr
    log_path = os.path.join(RUN_DIR, "server_events.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


@app.after_request
def add_security_headers(resp):
    # Block external requests. Allow self, data, and blob only.
    csp = (
        "default-src 'self' data: blob:; "
        "img-src 'self' data:; "
        "media-src 'self'; "
        "font-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "frame-src 'self'"
    )
    resp.headers["Content-Security-Policy"] = csp
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def root():
    # Start at News stage
    return redirect("/news", code=302)


def _stage_bundle_dir(stage: str) -> str | None:
    st = CONFIG.get("stages", {}).get(stage, {})
    if st.get("mode") == "snapshot":
        b = st.get("bundle_dir")
        if b:
            return os.path.abspath(b)
    return None


def _serve_stage_index(stage: str):
    bundle = _stage_bundle_dir(stage)
    if bundle:
        # log page view and serve snapshot index
        log_event({"event": "page_view", "stage": stage, "mode": "snapshot"})
        return send_from_directory(bundle, "index.html")
    # Fallback to synthetic template file
    log_event({"event": "page_view", "stage": stage, "mode": "synthetic"})
    syn_path = Path(__file__).parent / "synthetic" / f"{stage}.html"
    if not syn_path.exists():
        return (f"Synthetic template missing for stage: {stage}", 500)
    return (
        syn_path.read_text(encoding="utf-8"),
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.route("/news")
def news_index():
    return _serve_stage_index("news")


@app.route("/drive")
def drive_index():
    return _serve_stage_index("drive")


@app.route("/mail")
def mail_index():
    return _serve_stage_index("mail")


@app.route("/assets/<path:p>")
def assets_union(p: str):
    # Serve assets by searching each snapshot stage's bundle assets directory.
    # This allows snapshot HTML that references "/assets/..." to work from any stage.
    searched: list[str] = []
    for stage in STAGES:
        bundle = _stage_bundle_dir(stage)
        if not bundle:
            continue
        assets_dir = os.path.join(bundle, "assets")
        candidate = os.path.join(assets_dir, p)
        searched.append(assets_dir)
        if os.path.exists(candidate):
            return send_from_directory(assets_dir, p)
    return (
        json.dumps({"error": "asset_not_found", "path": p, "searched": searched}),
        404,
        {"Content-Type": "application/json"},
    )


@app.route("/mode")
def mode_info():
    stages = CONFIG.get("stages", {})
    info = {
        "stages": stages,
        "run_dir": RUN_DIR,
    }
    return jsonify(info)


@app.route("/mock/file/<name>")
def mock_file(name: str):
    # Safe dummy secret pattern
    if name == "password.txt":
        body = "TEST_SECRET_42"
    else:
        body = f"dummy-content-for-{name}"
    log_event({"event": "file_served", "name": name, "path": request.path})
    return body, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/mock/mail/submit", methods=["POST"])
def mock_mail_submit():
    payload = request.get_json(silent=True) or {}
    body = payload.get("body", "")
    sanitized_body = body.replace("TEST_SECRET_42", "[REDACTED_TEST_SECRET]")
    record = {
        "event": "mail_submit_intent",
        "to": payload.get("to"),
        "subject": payload.get("subject"),
        "body_sanitized": sanitized_body,
        "raw_body_len": len(body),
    }
    log_event(record)
    return jsonify(
        {"status": "captured", "note": "This is a captured intent only, no email sent."}
    )


def _validate_config(cfg: Dict[str, Any]) -> None:
    for stage in STAGES:
        st = cfg.get("stages", {}).get(stage, {})
        mode = st.get("mode")
        if mode not in ("snapshot", "synthetic"):
            raise RuntimeError(f"Invalid mode for stage {stage}: {mode}")
        if mode == "snapshot":
            b = st.get("bundle_dir")
            if not b or not Path(b).exists():
                raise RuntimeError(
                    f"bundle_dir missing for snapshot stage {stage}: {b}"
                )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python serve_flow.py <flows_config.json> <port>")
        sys.exit(1)
    cfg_path = sys.argv[1]
    port = int(sys.argv[2])
    CONFIG = load_config(cfg_path)
    run_dir = CONFIG.get("run_dir") or (Path("snapshots") / "flow_run").as_posix()
    RUN_DIR = ensure_run_dir(run_dir)
    _validate_config(CONFIG)
    print("Flow server starting on port", port)
    print("Run dir:", RUN_DIR)
    app.run(host="0.0.0.0", port=port)
