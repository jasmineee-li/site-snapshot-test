"""Live PVPO end-to-end smoke.

Exercises the full per-step capture + encounter detection + observational
P(eval) pipeline against a pre-running ``chrome-headless-shell`` Docker
container (see ``worldsim/docker/chrome-headless-shell.Dockerfile``). Writes
artifacts under ``tests/integration/fixtures/pvpo_smoke/`` and asserts the
result shape matches what Phase 4 integration now emits:

- ``result.json``'s ``encounter`` dict has ``max_coverage: float``,
  ``reference_step: int | None``, plus per-step coverage array.
- ``p_eval_score`` is a float in ``[0.0, 1.0]`` or ``None`` on probe
  failure; ``p_eval_breakdown`` is a 5-item list or ``None``;
  ``p_eval_failure`` is ``None`` on success or a known failure class.
- Legacy ``ecologically_valid`` stays pass-through; no ``VALIDITY_THRESHOLD``
  references appear.

Skipped unless:
  - A chrome-headless-shell container is reachable at 127.0.0.1:9222.
  - Anthropic credentials are configured (for the P(eval) API call).

The container is expected to be launched by the caller:
  docker run --rm -p 9222:9222 worldsim-chrome-headless-shell
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from urllib.request import urlopen

import pytest

from worldsim.phase_4.encounter_detection import determine_encounter
from worldsim.phase_4.p_eval_api import run_p_eval_api
from worldsim.phase_4.pvpo_browser_config import inject_animation_killer
from worldsim.phase_4.pvpo_capture import (
    Rect,
    atomic_capture_with_visibility,
    save_step_artifacts,
)

CDP_HOST = "127.0.0.1"
CDP_PORT = 9222


def _has_credentials() -> bool:
    for k in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        if os.environ.get(k, "").strip():
            return True
    return bool(
        os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
        and os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    )


def _chrome_headless_shell_reachable() -> bool:
    try:
        with socket.create_connection((CDP_HOST, CDP_PORT), timeout=1):
            pass
    except OSError:
        return False
    try:
        with urlopen(f"http://{CDP_HOST}:{CDP_PORT}/json/version", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


_FIXTURE_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>pvpo smoke</title></head>
<body>
<div>
  <p>Normal page content.</p>
  <article>
    Product review:
    <span data-worldsim-payload="1">ATTACK_PAYLOAD_TEXT_AB</span>
  </article>
</div>
</body></html>
"""


@pytest.mark.integration
@pytest.mark.skipif(
    not _chrome_headless_shell_reachable(),
    reason="chrome-headless-shell not reachable at 127.0.0.1:9222 — start the Docker container first",
)
@pytest.mark.skipif(
    not _has_credentials(),
    reason="Anthropic credentials required for P(eval) API call",
)
@pytest.mark.asyncio
async def test_pvpo_e2e_single_step_produces_expected_result_shape(tmp_path: Path):
    """Full Stage 1 + Stage 2 pipeline against the live headless-shell container."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        pytest.skip("playwright not installed")

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(f"http://{CDP_HOST}:{CDP_PORT}")
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await context.new_page()
        await page.set_content(_FIXTURE_HTML)

        cdp = await context.new_cdp_session(page)
        await inject_animation_killer(page, cdp)

        viewport = await page.evaluate("(() => ({w: window.innerWidth, h: window.innerHeight}))()")
        viewport_rect = Rect(
            x=0,
            y=0,
            w=int(viewport.get("w", 1280)),
            h=int(viewport.get("h", 720)),
        )
        capture = await atomic_capture_with_visibility(cdp, viewport_rect=viewport_rect)
        save_step_artifacts(tmp_path, step_idx=0, capture=capture)

        await page.close()
        await browser.close()

    task = {
        "id": "pvpo-smoke-1",
        "site": "shopping",
        "payload_texts": [{"rendered_payload": "ATTACK_PAYLOAD_TEXT_AB"}],
    }
    encounter = determine_encounter(task, tmp_path)

    assert isinstance(encounter.max_coverage, float)
    assert 0.0 <= encounter.max_coverage <= 1.0
    assert encounter.max_coverage > 0.0, (
        "fixture renders the payload visibly in the viewport; paint-verified coverage must be > 0"
    )
    assert encounter.reference_step == 0
    assert encounter.reference_screenshot_path is not None
    assert encounter.reference_screenshot_path.is_file()

    encounter_dict = encounter.as_dict()
    assert "max_coverage" in encounter_dict
    assert "per_step_coverage" in encounter_dict
    assert len(encounter_dict["per_step_coverage"]) == 1

    p_eval = await run_p_eval_api(
        encounter.reference_screenshot_path,
        site_context="shopping",
        trajectory_dir=tmp_path,
    )

    assert p_eval["failure_class"] is None or p_eval["failure_class"] in (
        "api_error",
        "auth_invalid",
        "insufficient_credits",
        "quota_exceeded",
        "no_tool_use",
        "schema_violation",
        "non_numeric_score",
    )
    if p_eval["failure_class"] is None:
        assert isinstance(p_eval["p_eval_score"], float)
        assert 0.0 <= p_eval["p_eval_score"] <= 1.0
        assert isinstance(p_eval["p_eval_breakdown"], list)
        assert len(p_eval["p_eval_breakdown"]) == 5
        probe_names = {p["name"] for p in p_eval["p_eval_breakdown"]}
        assert probe_names == {
            "content_realism",
            "ui_realism",
            "data_coherence",
            "artifact_presence",
            "overall_plausibility",
        }

    # Raw response persisted.
    raw = json.loads((tmp_path / "p_eval_raw_response.json").read_text())
    assert raw["site_context"] == "shopping"
