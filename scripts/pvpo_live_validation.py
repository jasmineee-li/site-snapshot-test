"""Live PVPO validation against r5 — STALE; superseded by ``pvpo_live_render_check.py``.

This script tests the pre-2026-04-19 ``<span data-worldsim-payload="1">``
HTML-attribute anchor strategy, which was deleted in commit ``8158b707`` and
replaced with TreeWalker substring-matching of the seeded ``payload_text``
against rendered DOM text content (see ``worldsim/phase_4/pvpo_query.js``).
The first question this script asks is therefore moot — there is no DOM
attribute to grep for. Use ``scripts/pvpo_live_render_check.py`` instead.

Original docstring (preserved for archival context):

  One-shot script (not a pytest test) that answers the two questions I cannot
  answer from a synthetic Docker-only fixture:

    1. Does the ``<span data-worldsim-payload="1">`` wrapper that
       ``materialize_adversarial_seed`` emits survive HTML rendering on each
       WebArena site? — Pick one adversarial task per site from
       ``logs/phase_2/adversarial_tasks.json``, materialize its seed, apply
       it via the editor API to the live r5 stack, fetch the delivery page
       URL, grep for ``data-worldsim-payload``. Sites that strip the span
       will show up as False here.

    2. Does the ink-occupancy oracle produce non-zero ``max_coverage`` on
       real r5 pages? — For sites where the wrapper survives, connect
       chrome-headless-shell (the Docker container already running on
       127.0.0.1:9222) via Playwright, navigate to the delivery page, run
       ``atomic_capture_with_visibility``, run ``determine_encounter``,
       report ``max_coverage``.

Prerequisites (operator-enforced):
  - SG sg-08792057943b27a65 temporarily opens 22/7770-7781/8023-8024/9999/
    3030/8888/3306/3307 from the caller's public IP.
  - chrome-headless-shell Docker container running at 127.0.0.1:9222.
  - .env sourced (Anthropic + WORLDSIM_* credentials).

Usage:
  (set -a; source ./.env; set +a; uv run python scripts/pvpo_live_validation.py \
    --tasks /path/to/adversarial_tasks.json)

The script resets each site via its ``reset_endpoint`` BEFORE seeding so
no stale state from prior runs pollutes the test. It does NOT reset after
(the SG change covers this session; subsequent pipeline runs re-reset
between tasks as part of normal operation).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worldsim.config import BenchmarkConfig, load_benchmark_config
from worldsim.phase_4.encounter_detection import determine_encounter
from worldsim.phase_4.pvpo_browser_config import inject_animation_killer
from worldsim.phase_4.pvpo_capture import (
    Rect,
    atomic_capture_with_visibility,
    save_step_artifacts,
)
from worldsim.phases.phase_2_text_fill import materialize_adversarial_seed
from worldsim.seeding import apply_data_seed_async

CDP_URL = "http://127.0.0.1:9222"
WASP_SITES = frozenset({"gitlab", "reddit"})


def first_task_for_site(tasks: list[dict], site: str) -> dict | None:
    for t in tasks:
        if t.get("site") == site and t.get("seed_template") and t.get("payload_texts"):
            return t
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        required=True,
        type=Path,
        help=(
            "Path to the Phase 2 adversarial_tasks.json artifact to validate. "
            "This is explicit because logs/phase_2 artifacts are runtime output "
            "and are no longer tracked in fresh worktrees."
        ),
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=ROOT / "instances.smoke.json",
        help="Instances config for the live stack, defaults to instances.smoke.json.",
    )
    return parser


def load_json_array(path: Path, *, label: str) -> list[dict]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{label} not found: {path}. Pass an explicit artifact path."
        ) from exc
    if not isinstance(payload, list):
        raise ValueError(f"{label} must contain a JSON array: {path}")
    return [item for item in payload if isinstance(item, dict)]


def sites_to_validate(tasks: list[dict], config: BenchmarkConfig) -> list[str]:
    configured_sites = {instance.site_name for instance in config.instances}
    sites: set[str] = set()
    for task in tasks:
        site = str(task.get("site") or "").strip()
        delivery_channel = task.get("delivery_channel")
        delivery_site = (
            str(delivery_channel.get("delivery_site") or "").strip()
            if isinstance(delivery_channel, dict)
            else ""
        )
        for candidate in (site, delivery_site):
            if candidate in WASP_SITES and candidate in configured_sites:
                sites.add(candidate)
    return sorted(sites)


def reset_instance(reset_endpoint: str, label: str) -> None:
    print(f"[{label}] resetting {reset_endpoint} ...", flush=True)
    r = requests.post(reset_endpoint, timeout=300)
    r.raise_for_status()


def delivery_url_for(task: dict, instance_site_url: str) -> str | None:
    """Derive the agent-visible delivery URL.

    Prefer ``delivery_url_from_seed_metadata`` after seed execution. This
    fallback exists for old artifacts or editor methods that do not emit
    read-surface metadata yet.
    """
    seed_template = task.get("seed_template") or {}
    calls = seed_template.get("editor_calls") or []
    if not calls:
        return None
    call = calls[0]
    method = call.get("method")
    args = call.get("args") or {}
    if method == "create_product_review":
        pk = args.get("entity_pk_value")
        if pk is not None:
            return urljoin(instance_site_url + "/", f"catalog/product/view/id/{pk}")
    if method in {"create_subreddit_post", "create_submission", "create_submission_title"}:
        forum = args.get("forum_name") or args.get("subreddit") or args.get("forum")
        if forum:
            submission_id = args.get("submission_id")
            if submission_id not in (None, ""):
                return urljoin(instance_site_url + "/", f"f/{forum}/{submission_id}")
            return urljoin(instance_site_url + "/", f"f/{forum}")
    if method == "create_comment":
        forum = args.get("forum_name") or args.get("subreddit") or args.get("forum")
        submission_id = args.get("submission_id")
        if forum and submission_id not in (None, ""):
            return urljoin(instance_site_url + "/", f"f/{forum}/{submission_id}")
        if forum:
            return urljoin(instance_site_url + "/", f"f/{forum}")
    if method == "create_forum":
        name = args.get("name_template") or args.get("name")
        if name:
            return urljoin(instance_site_url + "/", f"f/{name}")
    if method in {
        "create_issue_comment",
        "create_issue",
        "create_issue_description",
        "create_issue_title",
        "create_issue_note",
    }:
        project = (
            args.get("project_path_template") or args.get("project_path") or args.get("project")
        )
        if project:
            issue_iid = args.get("issue_iid")
            if issue_iid not in (None, ""):
                return urljoin(instance_site_url + "/", f"{project}/-/issues/{issue_iid}")
            return urljoin(instance_site_url + "/", f"{project}/-/issues")
    if method in {"create_mr", "create_mr_note"}:
        project = (
            args.get("project_path_template") or args.get("project_path") or args.get("project")
        )
        if project:
            mr_iid = args.get("mr_iid")
            if mr_iid not in (None, ""):
                return urljoin(instance_site_url + "/", f"{project}/-/merge_requests/{mr_iid}")
            return urljoin(instance_site_url + "/", f"{project}/-/merge_requests")
    return None


def delivery_url_from_seed_metadata(metadata: dict | None, instance_site_url: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    urls = metadata.get("read_surface_urls")
    if not isinstance(urls, list):
        return None
    for raw_url in urls:
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = raw_url.strip()
        if url.startswith(("http://", "https://")):
            return url
        return urljoin(instance_site_url + "/", url.lstrip("/"))
    return None


def validation_failures(results: dict[str, dict], expected_sites: list[str]) -> list[str]:
    failures: list[str] = []
    for site in expected_sites:
        result = results.get(site)
        if not isinstance(result, dict):
            failures.append(f"{site}: missing result")
            continue
        status = result.get("status")
        if status != "ok":
            failures.append(f"{site}: status={status or 'missing'}")
            continue
        if not result.get("html_has_attr"):
            failures.append(f"{site}: rendered payload text was not found in DOM text")
            continue
        try:
            max_coverage = float(result.get("max_coverage"))
        except (TypeError, ValueError):
            max_coverage = 0.0
        if max_coverage <= 0:
            failures.append(f"{site}: max_coverage={result.get('max_coverage')!r}")
    return failures


async def run_pvpo_on_page(url: str, payload: str, site: str, out_dir: Path) -> dict:
    """Navigate chrome-headless-shell to ``url`` and run PVPO capture."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"status": "skipped", "reason": "playwright not installed"}

    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(CDP_URL)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await context.new_page()
        print(f"  navigating to {url} ...", flush=True)
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as exc:
            await page.close()
            await browser.close()
            return {"status": "navigation_failed", "error": str(exc)}

        # Give dynamic widgets a moment to settle. WebArena sites are mostly
        # server-rendered so this is a short wait.
        await page.wait_for_timeout(1500)

        # Does the payload substring survive this site's HTML rendering?
        # Walk every text node looking for the first occurrence of the
        # rendered payload string. If the site stripped or reflowed the
        # text, matchFound stays False.
        payload_json = json.dumps(payload)
        html_has_attr = await page.evaluate(
            f"""(() => {{
                const target = {payload_json};
                if (!target) return false;
                const walker = document.createTreeWalker(
                    document.body || document.documentElement,
                    NodeFilter.SHOW_TEXT
                );
                while (walker.nextNode()) {{
                    if ((walker.currentNode.textContent || '').includes(target)) {{
                        return true;
                    }}
                }}
                return false;
            }})()"""
        )

        cdp = await context.new_cdp_session(page)
        await inject_animation_killer(page, cdp)

        viewport = await page.evaluate("(() => ({w: window.innerWidth, h: window.innerHeight}))()")
        viewport_rect = Rect(
            x=0,
            y=0,
            w=int(viewport.get("w", 1280)) or 1280,
            h=int(viewport.get("h", 720)) or 720,
        )
        capture = await atomic_capture_with_visibility(
            cdp, viewport_rect=viewport_rect, payload_text=payload
        )
        save_step_artifacts(out_dir, step_idx=0, capture=capture)

        await page.close()
        await browser.close()

    task = {
        "id": f"pvpo-live-{site}",
        "site": site,
        "payload_texts": [{"rendered_payload": payload}],
    }
    encounter = determine_encounter(task, out_dir)
    return {
        "status": "ok",
        "html_has_attr": html_has_attr,
        "max_coverage": encounter.max_coverage,
        "reference_step": encounter.reference_step,
        "per_step_coverage": encounter.per_step_coverage,
        "background_color": list(capture.background_color),
        "has_damage": capture.has_damage,
        "num_non_space_chars": sum(1 for c in payload if not c.isspace()),
    }


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = load_json_array(args.tasks, label="Phase 2 tasks artifact")
        config = load_benchmark_config(args.instances)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    sites_to_test = sites_to_validate(tasks, config)
    if not sites_to_test:
        print(
            "ERROR: no GitLab or Reddit tasks in the artifact match configured instances",
            file=sys.stderr,
        )
        return 2

    results: dict[str, dict] = {}

    for site in sites_to_test:
        task = first_task_for_site(tasks, site)
        if task is None:
            # Try the editor's delivery site if the bind site has no tasks.
            for t in tasks:
                ch = t.get("delivery_channel") or {}
                if ch.get("delivery_site") == site and t.get("seed_template"):
                    task = t
                    break
        if task is None:
            results[site] = {"status": "no_task"}
            continue

        # The seed executes against the delivery_channel.delivery_site (where
        # the editor posts), but the rendered page (where we measure PVPO)
        # also lives there.
        delivery_site = (task.get("delivery_channel") or {}).get("delivery_site") or site
        instance = next((i for i in config.instances if i.site_name == delivery_site), None)
        if instance is None:
            results[site] = {"status": "no_instance", "delivery_site": delivery_site}
            continue

        print(f"\n=== {site} (delivery via {delivery_site}, task {task['id']}) ===", flush=True)

        try:
            reset_instance(instance.reset_endpoint, label=delivery_site)
        except Exception as exc:
            results[site] = {"status": "reset_failed", "error": str(exc)}
            continue

        rendered_payload = task["payload_texts"][0]["rendered_payload"]
        seed = materialize_adversarial_seed(task["seed_template"], rendered_payload)
        sample_body_field = _sample_body_from_seed(seed)
        print(
            f"  materialized seed detail starts: {sample_body_field[:140]!r}",
            flush=True,
        )

        instance_dict = instance.model_dump()
        cleanup = None
        seed_metadata: dict = {}
        try:
            cleanup, seed_metadata = await apply_data_seed_async(seed, instance_dict)
        except Exception as exc:
            results[site] = {
                "status": "seed_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue

        # Small wait: some sites (reddit) index posts asynchronously.
        time.sleep(2)

        try:
            url = delivery_url_from_seed_metadata(
                seed_metadata, instance.site_url
            ) or delivery_url_for(task, instance.site_url)
            if url is None:
                results[site] = {
                    "status": "no_delivery_url",
                    "note": "editor method not yet mapped in delivery_url_for",
                }
                continue

            out_dir = ROOT / f"logs/pvpo_live_r5/{site}"
            result = await run_pvpo_on_page(url, rendered_payload, site, out_dir)
            result["delivery_url"] = url
            results[site] = result
            print(f"  result: {json.dumps(result, indent=2)}", flush=True)
        finally:
            if cleanup is not None:
                try:
                    await asyncio.to_thread(cleanup.cleanup)
                except Exception as exc:
                    print(f"  cleanup error: {exc}", flush=True)

    print("\n\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))
    failures = validation_failures(results, sites_to_test)
    if failures:
        print("\nPVPO validation failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def _sample_body_from_seed(seed: dict) -> str:
    """Return a representative body-field string from the materialized seed."""
    calls = seed.get("editor_calls") or []
    if not calls:
        return ""
    args = calls[0].get("args") or {}
    for key in ("detail", "body", "content", "text", "comment"):
        v = args.get(key)
        if isinstance(v, str) and v:
            return v
    return json.dumps(args)[:200]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
