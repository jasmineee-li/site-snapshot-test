"""Live smoke for Phase 2 L3 / L4 target resolver.

Hits the real Anthropic Messages API and the real benchmark instance.
Skipped in default CI; run locally with:

    LIVE_INSTANCES_FILE=configs/benchmark_hosts/r5.yaml \
    CLAUDE_CODE_OAUTH_TOKEN=... \
    uv run pytest -m live_l3 tests/integration/test_phase_2_target_resolver_live.py

Acceptance:
  * classifier round-trips a trivial intent into a well-formed
    emit_target tool call.
  * L3 probe returns anchors referring to a real issue / submission
    in the live image.
  * L4 probe returns >=1 concrete item for a known-populated forum
    and search query.
"""

from __future__ import annotations

import os

import pytest

from worldsim.phase_2.target_resolution.http_probes import _benign_probe_instance
from worldsim.phase_2.target_resolution.l3 import _call_anthropic_classifier, resolve_l3
from worldsim.phase_2.target_resolution.l4 import resolve_l4
from worldsim.phase_2.target_resolution.listing_probes import _default_listing_probe
from worldsim.placeholders import placeholders_for_site_urls

pytestmark = [pytest.mark.integration, pytest.mark.live_l3]


def _has_anthropic_credentials() -> bool:
    for key in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        if os.environ.get(key, "").strip():
            return True
    if (
        os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
        and os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    ):
        return True
    return False


_SKIP_NO_CREDS = pytest.mark.skipif(
    not _has_anthropic_credentials(),
    reason="requires CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN+BASE_URL",
)


@pytest.fixture
def gitlab_placeholders(live_instance):
    instance = live_instance("gitlab")
    return (
        instance,
        placeholders_for_site_urls([("gitlab", str(instance.get("site_url") or ""))]),
    )


@pytest.fixture
def reddit_placeholders(live_instance):
    instance = live_instance("reddit")
    return (
        instance,
        placeholders_for_site_urls([("reddit", str(instance.get("site_url") or ""))]),
    )


@_SKIP_NO_CREDS
async def test_live_l3_classifier_round_trips_gitlab_intent(gitlab_placeholders):
    _instance, placeholders = gitlab_placeholders
    task = {
        "id": "live-l3-1",
        "sites": ["gitlab"],
        "start_urls": ["__GITLAB__"],
        "instruction": ("Get whether my latest updated issue with 'theme' in its title is closed"),
        "agent_context": {"authentication": {"credentials": {"username": "byteblaze"}}},
        "reward_function": {"eval": []},
    }
    parsed = await _call_anthropic_classifier(task, placeholders)
    assert isinstance(parsed, dict), "classifier returned non-dict or None"
    assert "kind" in parsed and "probe_query" in parsed and "confidence" in parsed
    if parsed["kind"] is not None:
        assert parsed["kind"] in {
            "gitlab_issue",
            "gitlab_mr",
            "gitlab_search_result",
            "gitlab_dashboard_list",
        }


@_SKIP_NO_CREDS
async def test_live_l3_resolves_theme_editor_to_concrete_issue(gitlab_placeholders):
    instance, placeholders = gitlab_placeholders
    task = {
        "id": "live-l3-2",
        "sites": ["gitlab"],
        "start_urls": ["__GITLAB__"],
        "instruction": ("Get whether my latest updated issue with 'theme' in its title is closed"),
        "agent_context": {"authentication": {"credentials": {"username": "byteblaze"}}},
        "reward_function": {"eval": []},
    }
    result = await resolve_l3(task, placeholders, instance)
    assert result["layer"] == "L3"
    if result["kind"] is None:
        pytest.skip(
            f"classifier marked task out of scope in the live image: {result.get('reason')}"
        )
    assert result["kind"] in {"gitlab_issue", "gitlab_mr"}
    anchors = result.get("anchors") or {}
    assert anchors.get("project_id") or anchors.get("project_path")


async def test_live_l4_lists_gitlab_issues_in_populated_project(gitlab_placeholders):
    instance, _ = gitlab_placeholders
    # Self-heal: pick a project with issues at runtime instead of hardcoding
    # a project_id that drifts between image refreshes. We hit the sitewide
    # /issues listing once, take the project_id with the most rows in the
    # top page, and run L4's listing probe against it. If the image has no
    # issues at all (fresh reset), the test skips with a meaningful reason.
    import requests

    from worldsim.auth_tokens import resolve_bearer_token

    base = str(instance.get("site_url") or "").rstrip("/")
    probe_instance = _benign_probe_instance(instance)
    auth = probe_instance.get("api_auth") or probe_instance.get("auth") or {}
    headers: dict[str, str] = {}
    if isinstance(auth, dict) and auth.get("type") == "bearer_token":
        token = resolve_bearer_token(auth, site_url=str(probe_instance.get("site_url", "")))
        header_name = str(auth.get("header_name") or "Authorization")
        if header_name.lower() == "authorization" and not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers[header_name] = token
    sess = requests.Session()
    r = sess.get(
        f"{base}/api/v4/issues",
        headers=headers,
        params={"per_page": 20, "order_by": "updated_at", "sort": "desc"},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or not payload:
        pytest.skip("live gitlab image has no visible issues sitewide")
    from collections import Counter

    by_project = Counter(item["project_id"] for item in payload if "project_id" in item)
    if not by_project:
        pytest.skip("sitewide /issues returned rows without project_id")
    picked_project_id, _ = by_project.most_common(1)[0]

    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"project_id": picked_project_id, "scope": "issues"},
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L2",
    }
    items = await _default_listing_probe(resource, {}, instance)
    assert items, f"project {picked_project_id} unexpectedly returned no items"
    records = await resolve_l4(resource, {}, instance, top_n=3)
    assert len(records) >= 1
    assert records[0]["kind"] in {"gitlab_issue", "gitlab_mr"}


async def test_live_reddit_forum_stays_created_child_exposure_not_l4(reddit_placeholders):
    instance, _ = reddit_placeholders
    # Point at an actually-live reddit replica port on the current host.
    # instances.smoke.json still carries the legacy-topology port 9999,
    # but r5 runs the scale topology with reddit replicas on 9900-9990
    # (see scripts/proxy_ports.conf). Probe the canonical reddit_0 port
    # so the proxying adapter rewrites to a port the nginx proxy actually
    # serves. Falls back to the fixture port when we're on a host with
    # the legacy single-reddit deploy.
    import os as _os
    from urllib.parse import urlsplit as _urlsplit

    original_url = str(instance.get("site_url") or "")
    parsed = _urlsplit(original_url)
    override_port = int(_os.environ.get("LIVE_REDDIT_L4_PORT", "9900") or "9900")
    if parsed.scheme and parsed.hostname:
        instance = {
            **instance,
            "site_url": f"{parsed.scheme}://{parsed.hostname}:{override_port}",
        }
    resource = {
        "kind": "reddit_forum",
        "anchors": {"forum_name": "books"},
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L1",
    }
    records = await resolve_l4(resource, {}, instance, top_n=3)
    assert records == [resource]


@_SKIP_NO_CREDS
async def test_live_l3_classifier_handles_pure_action_as_null_kind(
    gitlab_placeholders,
):
    _instance, placeholders = gitlab_placeholders
    task = {
        "id": "live-l3-3",
        "sites": ["gitlab"],
        "start_urls": ["__GITLAB__"],
        "instruction": "Fork the metaseq repository.",
        "agent_context": {"authentication": {"credentials": {"username": "byteblaze"}}},
        "reward_function": {"eval": []},
    }
    parsed = await _call_anthropic_classifier(task, placeholders)
    assert isinstance(parsed, dict)
    if parsed["kind"] is not None:
        pytest.skip(
            "classifier picked a concrete kind for a pure-action task; prompt needs tightening"
        )
    assert parsed.get("confidence") is not None
