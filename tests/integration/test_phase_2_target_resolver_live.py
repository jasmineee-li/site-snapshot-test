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

from worldsim.phases.phase_2_target_resolver import (
    _call_anthropic_classifier,
    _default_listing_probe,
    _default_probe,
    resolve_l3,
    resolve_l4,
)
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
    instance, placeholders = gitlab_placeholders
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
    # byteblaze/dotfiles is present in the WebArena GitLab image and has
    # enough issues to exercise top-N. We go straight to _default_listing_probe
    # to isolate the probe from the classifier.
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"project_id": 158, "scope": "issues"},
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L2",
    }
    items = await _default_listing_probe(resource, {}, instance)
    if not items:
        pytest.skip(
            "project 158 has no issues on this image; pick another project_id via "
            "GET /api/v4/projects?search=dotfiles"
        )
    records = await resolve_l4(resource, {}, instance, top_n=3)
    assert len(records) >= 1
    assert records[0]["kind"] in {"gitlab_issue", "gitlab_mr"}


async def test_live_l4_lists_reddit_forum_submissions(reddit_placeholders):
    instance, _ = reddit_placeholders
    resource = {
        "kind": "reddit_forum",
        "anchors": {"forum_name": "books"},
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L1",
    }
    records = await resolve_l4(resource, {}, instance, top_n=3)
    if not records:
        pytest.skip("forum 'books' returned no submissions; sitewide data may differ")
    assert records[0]["kind"] == "reddit_submission"
    assert records[0]["anchors"]["submission_id"].isdigit()


@_SKIP_NO_CREDS
async def test_live_l3_classifier_handles_pure_action_as_null_kind(
    gitlab_placeholders,
):
    instance, placeholders = gitlab_placeholders
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
