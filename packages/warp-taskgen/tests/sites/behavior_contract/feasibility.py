"""Assertions for the deterministic Feasibility owner seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def assert_feasibility_behavior(
    policy: Any,
    task: Mapping[str, object],
    *,
    instance_site_url: str,
    expected_probe_url: str,
    expected_parent_id: str,
) -> None:
    """Check exact source probe planning and fail-closed classification."""

    probes = policy.probe_targets(dict(task), instance_site_url)
    assert len(probes) == 1
    probe = probes[0]
    assert probe["url"] == expected_probe_url
    assert probe["thread_id"] == expected_parent_id
    assert policy.auth_self_test_path() is None
    assert policy.requires_authenticated_preflight() is False
    assert policy.classify_probe(status_code=200, url=expected_probe_url) == "verified"
    assert policy.decide_source_data(classification="verified") == "admissible"
    assert policy.counts_toward_run_bailout("verified") is False
    assert policy.should_bailout_source_data_run(classification="verified") is False
    assert policy.restore_drop_on_run_bailout({}) is False

    wrong_site = dict(task)
    foreign_site = f"not_{getattr(policy, 'site', 'site')}"
    wrong_site["site"] = foreign_site
    wrong_site["sites"] = [foreign_site]
    assert policy.probe_targets(wrong_site, instance_site_url) == []
    assert policy.classify_probe(status_code=403, url=expected_probe_url) == "unsupported"
    assert policy.decide_source_data(classification="unsupported") == "ineligible"
