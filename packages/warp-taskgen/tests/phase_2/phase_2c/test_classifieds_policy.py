from __future__ import annotations

import pytest

from warp_taskgen.phase_2.phase_2c.classifieds_policy import ClassifiedsFeasibilityPolicy
from warp_taskgen.phase_2.phase_2c.policy import (
    FeasibilityPolicyCatalog,
    PreflightClassification,
    ProbeTarget,
)


def _task() -> dict[str, object]:
    return {
        "benchmark": "visualwebarena",
        "site": "classifieds",
        "benign_target_resource": {
            "start_url_resolved": "/index.php?page=item&id=12085",
            "anchors": {"listing_id": "12085"},
        },
        "adversarial_data_seed": {
            "editor_calls": [
                {
                    "benchmark": "visualwebarena",
                    "site": "classifieds",
                    "method": "create_listing_reply",
                    "args": {
                        "listing_id": "{benign_listing_id}",
                        "body": "semantic carrier",
                    },
                }
            ]
        },
    }


def test_classifieds_policy_is_explicit_and_targets_only_exact_listing() -> None:
    policy = ClassifiedsFeasibilityPolicy()
    catalog = FeasibilityPolicyCatalog.from_policies((policy,))

    assert catalog.get("visualwebarena", "classifieds") is policy
    assert catalog.get("webarena_verified", "classifieds") is None
    assert policy.requires_authenticated_preflight() is True
    assert policy.auth_self_test_path() == "/index.php?page=user&action=dashboard"
    assert policy.probe_targets(_task(), "https://classifieds.test") == [
        ProbeTarget(
            "https://classifieds.test/index.php?page=item&id=12085",
            "benign_start_url",
        )
    ]


def test_classifieds_policy_has_no_root_or_latest_fallback() -> None:
    task = _task()
    task["benign_target_resource"] = {"anchors": {}}
    task["adversarial_data_seed"] = {
        "editor_calls": [
            {
                "site": "classifieds",
                "method": "create_listing_reply",
                "args": {"listing_id": "", "body": "semantic carrier"},
            }
        ]
    }

    assert ClassifiedsFeasibilityPolicy().probe_targets(task, "https://classifieds.test") == []


@pytest.mark.parametrize(
    "start_url",
    [
        "/",
        "/index.php",
        "/index.php?page=item&id=99999",
    ],
)
def test_classifieds_policy_rejects_non_exact_task_probe_url(start_url: str) -> None:
    task = _task()
    task["benign_target_resource"] = {
        "start_url_resolved": start_url,
        "anchors": {"listing_id": "12085"},
    }

    assert ClassifiedsFeasibilityPolicy().probe_targets(task, "https://classifieds.test") == [
        ProbeTarget(
            "https://classifieds.test/index.php?page=item&id=12085",
            "editor_call[0].classifieds.create_listing_reply",
        )
    ]


@pytest.mark.parametrize(
    ("status", "headers", "body"),
    [
        (302, {"location": "/index.php?page=login"}, ""),
        (
            200,
            {},
            '<form action="/index.php?page=login"><input name="s_email">'
            '<input name="s_password"></form>',
        ),
    ],
)
def test_classifieds_policy_classifies_login_surfaces_as_auth_failures(
    status: int,
    headers: dict[str, str],
    body: str,
) -> None:
    classification = ClassifiedsFeasibilityPolicy().classify_probe(
        status=status,
        headers=headers,
        body_snippet=body,
        exception_name=None,
    )

    assert classification == PreflightClassification(
        kind="auth_missing",
        quarantine=True,
        http_status=status,
        detail=classification.detail,
    )
