#!/usr/bin/env python3
"""
Website Browser Server for AgentLab Redteam Experiments

This server allows browsing generated HTML websites from redteam experiments
via a simple web interface. Run this script to start the server, then visit
http://localhost:8000 in your browser.

Usage:
    python -m agentlab.analyze.website_browser
"""

import http.server
import socketserver
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from agentlab.experiments.exp_utils import RESULTS_DIR
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8000


class WebsiteBrowserHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for browsing experiment websites."""

    def __init__(self, *args, **kwargs):
        # Set the directory to serve from (will be overridden per request)
        super().__init__(*args, directory=str(RESULTS_DIR), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/" or path == "/index":
            self.serve_index()
        elif path.startswith("/website/"):
            self.serve_website_detail(path)
        elif path.startswith("/html/"):
            self.serve_html_file(path)
        elif path == "/api/websites":
            self.serve_website_list()
        else:
            self.send_error(404, "File not found")

    def serve_index(self):
        """Serve the main index page with website browser (grouped by domain)."""
        websites = self.get_websites_grouped()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentLab Website Browser</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .header p {{
            margin: 0;
            color: #666;
        }}
        .website-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        .website-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
            text-decoration: none;
            color: inherit;
            display: block;
        }}
        .website-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .website-domain {{
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #007acc;
        }}
        .website-meta {{
            font-size: 14px;
            color: #666;
            margin-bottom: 12px;
        }}
        .website-meta-item {{
            margin-bottom: 4px;
        }}
        .website-meta-label {{
            font-weight: 500;
            color: #444;
        }}
        .route-count {{
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 13px;
            margin-top: 8px;
        }}
        .run-count {{
            display: inline-block;
            background: #e8f5e9;
            color: #388e3c;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 13px;
            margin-left: 8px;
        }}
        .no-websites {{
            text-align: center;
            padding: 40px;
            color: #666;
            background: white;
            border-radius: 8px;
        }}
        .local-path {{
            font-size: 12px;
            color: #888;
            word-break: break-all;
            background: #f9f9f9;
            padding: 6px 10px;
            border-radius: 4px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AgentLab Website Browser</h1>
        <p>Browse generated websites from redteam experiments. Found {len(websites)} unique website domains.</p>
    </div>

    <div class="website-grid">
"""

        if not websites:
            html += """
        <div class="no-websites">
            <h2>No Websites Found</h2>
            <p>No redteam experiments with generated HTML files were found in the results directory.</p>
            <p>Make sure you've run some redteam experiments and that they generated HTML files.</p>
        </div>
"""
        else:
            for domain, info in sorted(websites.items()):
                routes_count = len(info['routes'])
                runs_count = len(info['runs'])
                sample_path = info['sample_local_path']

                html += f"""
        <a href="/website/{domain}" class="website-card">
            <div class="website-domain">{domain}</div>
            <div class="website-meta">
                <div class="website-meta-item">
                    <span class="website-meta-label">Routes:</span> {routes_count} page(s)
                </div>
            </div>
            <span class="route-count">{routes_count} routes</span>
            <span class="run-count">{runs_count} run(s)</span>
            <div class="local-path">{sample_path}</div>
        </a>
"""

        html += """
    </div>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_website_detail(self, path):
        """Serve detail page for a specific website domain."""
        # Extract domain from path: /website/jira -> jira
        parts = path.strip("/").split("/", 1)
        if len(parts) < 2:
            self.send_error(404, "Domain not specified")
            return

        domain = unquote(parts[1])
        websites = self.get_websites_grouped()

        if domain not in websites:
            self.send_error(404, f"Website domain '{domain}' not found")
            return

        info = websites[domain]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{domain} - AgentLab Website Browser</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .back-link {{
            color: #007acc;
            text-decoration: none;
            font-size: 14px;
        }}
        .back-link:hover {{
            text-decoration: underline;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            font-size: 18px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .route-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .route-item {{
            display: flex;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .route-item:last-child {{
            border-bottom: none;
        }}
        .route-path {{
            flex: 1;
            font-family: monospace;
            font-size: 14px;
            color: #333;
        }}
        .route-actions {{
            display: flex;
            gap: 10px;
        }}
        .btn {{
            display: inline-block;
            padding: 6px 14px;
            background: #007acc;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 13px;
        }}
        .btn:hover {{
            background: #005aa3;
        }}
        .btn-secondary {{
            background: #666;
        }}
        .btn-secondary:hover {{
            background: #444;
        }}
        .run-item {{
            padding: 12px;
            background: #f9f9f9;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        .run-item:last-child {{
            margin-bottom: 0;
        }}
        .run-title {{
            font-weight: 500;
            color: #333;
            margin-bottom: 6px;
        }}
        .run-meta {{
            font-size: 13px;
            color: #666;
        }}
        .local-path {{
            font-size: 12px;
            color: #888;
            word-break: break-all;
            background: #f0f0f0;
            padding: 8px 12px;
            border-radius: 4px;
            margin-top: 12px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <a href="/" class="back-link">&larr; Back to all websites</a>
        <h1>{domain}</h1>
        <p>Generated website with {len(info['routes'])} routes, used in {len(info['runs'])} experiment run(s).</p>
    </div>

    <div class="section">
        <h2>Routes / Pages</h2>
        <ul class="route-list">
"""

        # Sort routes and display them
        for route in sorted(info['routes'].keys()):
            route_info = info['routes'][route]
            display_route = route if route else "/"
            html_path = route_info['html_path']

            html += f"""
            <li class="route-item">
                <span class="route-path">/{display_route}</span>
                <div class="route-actions">
                    <a href="/html/{html_path}" class="btn" target="_blank">View HTML</a>
                </div>
            </li>
"""

        html += """
        </ul>
    </div>

    <div class="section">
        <h2>Experiment Runs</h2>
"""

        for run in info['runs']:
            html += f"""
        <div class="run-item">
            <div class="run-title">{run['task']}</div>
            <div class="run-meta">
                Agent: {run['agent']} | Date: {run['date']}
            </div>
            <div class="local-path">{run['path']}</div>
        </div>
"""

        html += """
    </div>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_html_file(self, path):
        """Serve an HTML file from the results directory."""
        # Path format: /html/relative/path/to/file.html
        parts = path.strip("/").split("/", 1)
        if len(parts) < 2:
            self.send_error(404, "File path not specified")
            return

        relative_path = unquote(parts[1])
        full_path = RESULTS_DIR / relative_path

        if not full_path.exists():
            self.send_error(404, f"File not found: {relative_path}")
            return

        if not full_path.suffix.lower() == '.html':
            self.send_error(403, "Only HTML files can be served")
            return

        # Security check: make sure path is within RESULTS_DIR
        try:
            full_path.resolve().relative_to(RESULTS_DIR.resolve())
        except ValueError:
            self.send_error(403, "Access denied")
            return

        try:
            content = full_path.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def serve_website_list(self):
        """Serve JSON list of websites."""
        websites = self.get_websites_grouped()
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(websites, default=str).encode())

    def get_websites_grouped(self):
        """
        Find all websites grouped by domain.
        Returns dict with structure:
        {
            "domain": {
                "routes": {
                    "route/path": {
                        "html_path": "relative/path/to/file.html",
                        "mode": "synthetic"
                    }
                },
                "runs": [
                    {"task": "...", "agent": "...", "date": "...", "path": "..."}
                ],
                "sample_local_path": "..."
            }
        }
        """
        websites = defaultdict(lambda: {
            'routes': {},
            'runs': [],
            'sample_local_path': ''
        })

        def scan_directory(dir_path, relative_path=""):
            for item in dir_path.iterdir():
                if item.is_dir():
                    # Check for flow_config.json (indicates experiment with websites)
                    flow_config = item / "flow_config.json"
                    if flow_config.exists():
                        try:
                            config = json.loads(flow_config.read_text())
                            stages = config.get("stages", {})
                            run_dir = config.get("run_dir", "")

                            # Extract experiment metadata
                            exp_info = self._get_experiment_info(item)

                            # Group by domain from stage keys
                            for stage_key, stage_info in stages.items():
                                # Extract domain from stage key (first part before /)
                                parts = stage_key.split("/", 1)
                                domain = parts[0]
                                route = parts[1] if len(parts) > 1 else ""

                                if not domain:
                                    continue

                                # Find the HTML file for this route
                                html_filename = self._stage_to_html_filename(stage_key)
                                variation_dir = item / "variation_0"
                                html_file = variation_dir / html_filename

                                if html_file.exists():
                                    html_relative_path = str(html_file.relative_to(RESULTS_DIR))

                                    # Add route if not already present
                                    if route not in websites[domain]['routes']:
                                        websites[domain]['routes'][route] = {
                                            'html_path': html_relative_path,
                                            'mode': stage_info.get('mode', 'unknown')
                                        }

                                    if not websites[domain]['sample_local_path']:
                                        websites[domain]['sample_local_path'] = str(variation_dir)

                            # Add this run to all domains found
                            domains_in_run = set()
                            for stage_key in stages.keys():
                                domain = stage_key.split("/")[0]
                                if domain:
                                    domains_in_run.add(domain)

                            for domain in domains_in_run:
                                # Avoid duplicate runs
                                run_info = {
                                    'task': exp_info.get('task', 'Unknown'),
                                    'agent': exp_info.get('agent', 'Unknown'),
                                    'date': exp_info.get('date', 'Unknown'),
                                    'path': str(item)
                                }
                                if run_info not in websites[domain]['runs']:
                                    websites[domain]['runs'].append(run_info)

                        except Exception as e:
                            logger.debug(f"Error processing {flow_config}: {e}")
                    else:
                        # Recursively scan subdirectories
                        scan_directory(item, str(Path(relative_path) / item.name))

        scan_directory(RESULTS_DIR)
        return dict(websites)

    def _stage_to_html_filename(self, stage_key):
        """Convert a stage key to the expected HTML filename."""
        # Replace / with _ (preserving consecutive slashes as consecutive underscores)
        filename = stage_key.replace("/", "_")
        # Strip leading/trailing underscores only
        filename = filename.strip('_')
        return f"{filename}.html"

    def _get_experiment_info(self, exp_dir):
        """Extract experiment metadata from directory."""
        info = {
            'date': 'Unknown',
            'agent': 'Unknown',
            'task': 'Unknown'
        }

        # Try to extract date from directory name
        dir_name = exp_dir.name
        date_match = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', dir_name)
        if date_match:
            info['date'] = date_match.group(1).replace('_', ' ')

        # Try to load exp_args.pkl for more metadata
        try:
            import pickle
            exp_args_file = exp_dir / "exp_args.pkl"
            if exp_args_file.exists():
                with open(exp_args_file, "rb") as f:
                    exp_args = pickle.load(f)
                    if hasattr(exp_args, 'agent_args'):
                        info['agent'] = getattr(exp_args.agent_args, 'agent_name', 'Unknown')
                    if hasattr(exp_args, 'env_args'):
                        info['task'] = getattr(exp_args.env_args, 'task_name', 'Unknown')
        except Exception as e:
            logger.debug(f"Could not load metadata for {exp_dir.name}: {e}")

        return info


def main():
    """Start the website browser server."""
    with socketserver.TCPServer(("", PORT), WebsiteBrowserHandler) as httpd:
        logger.info(f"Website Browser Server started at http://localhost:{PORT}")
        logger.info(f"Serving experiments from: {RESULTS_DIR}")
        logger.info("Press Ctrl+C to stop the server")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")


if __name__ == "__main__":
    main()
