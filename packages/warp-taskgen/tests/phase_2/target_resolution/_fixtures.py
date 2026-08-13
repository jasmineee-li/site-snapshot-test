# ruff: noqa
"""L1/L2 fixtures for :mod:`warp_taskgen.phase_2.target_resolution`.

Every fixture builds a minimal benign task record inline — no dependency
on ``logs/phase_1/benign_tasks.json``. Covers every ResourceKind plus
edge shapes lifted from the real dataset (regex-anchored eval URLs,
``.json`` suffix, array-of-URLs, intent-only bare ``__GITLAB__``).
"""

from __future__ import annotations

from typing import Any

import pytest

from warp_taskgen.phase_2.target_resolution import listing_probes
from warp_taskgen.phase_2.target_resolution import runner as resolver
from warp_taskgen.phase_2.target_resolution.constants import VIEWPORT_BUDGET_CHARS
from warp_taskgen.phase_2.target_resolution.encounter import _assert_anchor_contract_conformance
from warp_taskgen.phase_2.target_resolution.http_probes import (
    _benign_probe_instance,
    _postmill_submission_comment_count_from_html,
)
from warp_taskgen.phase_2.target_resolution.l3 import resolve_l3
from warp_taskgen.phase_2.target_resolution.l4 import resolve_l4
from warp_taskgen.phase_2.target_resolution.runner import resolve_tasks
from warp_taskgen.phase_2.target_resolution.resolver import derive_benign_target_resource
from warp_taskgen.phase_2.target_resolution.types import ResolverContractDriftError
from warp_taskgen.phase_2.target_resolution.url_matching import _literalize_regex_value
from warp_taskgen.placeholders import placeholders_for_site_urls

PLACEHOLDERS = placeholders_for_site_urls(
    [
        ("gitlab", "https://gitlab.local"),
        ("reddit", "https://reddit.local"),
    ]
)


def _gitlab_task(
    *,
    task_id: str = "t",
    eval_url: Any | None = None,
    start_urls: list[str] | None = None,
    username: str = "byteblaze",
    evaluator: str = "NetworkEventEvaluator",
    instruction: str = "",
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "sites": ["gitlab"],
        "start_urls": start_urls if start_urls is not None else ["__GITLAB__"],
        "instruction": instruction,
        "agent_context": {"authentication": {"credentials": {"username": username}}},
        "reward_function": {"eval": []},
    }
    if eval_url is not None:
        task["reward_function"]["eval"] = [{"evaluator": evaluator, "expected": {"url": eval_url}}]
    return task


def _reddit_task(
    *,
    task_id: str = "t",
    eval_url: Any | None = None,
    start_urls: list[str] | None = None,
    username: str = "MarvelsGrantMan136",
    evaluator: str = "NetworkEventEvaluator",
    instruction: str = "",
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "sites": ["reddit"],
        "start_urls": start_urls if start_urls is not None else ["__REDDIT__"],
        "instruction": instruction,
        "agent_context": {"authentication": {"credentials": {"username": username}}},
        "reward_function": {"eval": []},
    }
    if eval_url is not None:
        task["reward_function"]["eval"] = [{"evaluator": evaluator, "expected": {"url": eval_url}}]
    return task


# --- gitlab_issue --------------------------------------------------------

# --- gitlab_mr -----------------------------------------------------------

# --- gitlab_search_result ------------------------------------------------

# --- gitlab_dashboard_list ----------------------------------------------

# --- reddit_submission --------------------------------------------------

# --- reddit_forum -------------------------------------------------------

# --- reddit_dashboard_list ----------------------------------------------

# --- fallthrough / out-of-scope ------------------------------------------

# --- L3 resolver (stubbed classifier + probe) ----------------------------

import asyncio  # noqa: E402


def _make_classifier(parsed):
    async def _stub(task, placeholders):
        return parsed

    return _stub


def _make_probe(anchors):
    async def _stub(probe_query, task, instance, placeholders):
        return anchors

    return _stub


# --- L4 dynamic listing expansion ----------------------------------------


def _make_listing_probe(items):
    async def _stub(resource, task, instance):
        return items

    return _stub


# Real live tests live in tests/integration/test_phase_2_target_resolver_live.py
# behind pytest.mark.live_l3 (skipped by default; run with -m live_l3).

# ---------------------------------------------------------------------
# Anchor / contract conformance self-check (commit 3 of the registry work)
# ---------------------------------------------------------------------

# -----------------------------------------------------------------------
# start_url_resolved reconstruction (Bug A) — Phase 2c anchor-vs-probe
# alignment. The probe must navigate to the concrete entity where the
# seed lives, not whatever project root the benign task's raw
# start_urls[0] happens to carry.
# -----------------------------------------------------------------------

# --- _canonicalize_project_path -----------------------------------------
#
# The L3 LLM sometimes emits project_path values like
# 'localhost:8023/a11yproject/a11yproject.com' because the API probe's
# `web_url` carries the localhost authority. The GitLab project-by-path
# endpoint requires a host-stripped path that we then percent-encode.
# These tests pin the host-stripping behavior so encoding bugs surface
# at unit-test time, not at probe time.

# --- L3 out_of_scope_for_option_a kind -----------------------------------
#
# Bucket C of the GitLab attrition (18 unique tasks) was the L3 LLM
# being forced by `tool_choice: "tool"` to pick a kind even on commit-
# count / blob-view / fork-action tasks. Adding `out_of_scope_for_option_a`
# to the kind enum gives the LLM a clean abstain branch. The resolver
# must treat it as terminal: kind=None, layer=L3, no L4 retry.

# --- L3 probe-kind coherence check --------------------------------------
#
# The L3 LLM sometimes emits a (kind, probe_query.api) pair where the
# probe's result shape can't fill the kind's anchor schema. We catch the
# mismatch before running the probe so the failure is diagnostic rather
# than a silent "no anchors" log line.

# --- L3 classifier failure includes class name --------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
