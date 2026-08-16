from __future__ import annotations

import asyncio

import pytest

from warp_taskgen.phase_2.phase_2c.classifieds_policy import ClassifiedsFeasibilityPolicy
from warp_taskgen.phase_2.phase_2c.policy import (
    FeasibilityPolicyCatalog,
    PreflightClassification,
    ProbeTarget,
)
from warp_taskgen.phases.phase_2c_preflight import preflight_benign_targets


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
    assert policy.requires_authenticated_preflight() is False
    assert policy.auth_self_test_path() is None
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
    assert "anonymous-reader contract" in classification.detail


def test_classifieds_source_preflight_probes_with_anonymous_context() -> None:
    calls: list[str] = []
    context_options: list[dict[str, object]] = []

    class _Response:
        def __init__(self) -> None:
            self.status = 200
            self.headers: dict[str, str] = {}

        async def text(self) -> str:
            return "<html>public listing</html>"

        async def dispose(self) -> None:
            return None

    class _Context:
        async def get(self, url: str, **_kwargs: object) -> _Response:
            calls.append(url)
            return _Response()

    async def factory(options: dict[str, object]) -> _Context:
        context_options.append(dict(options))
        return _Context()

    catalog = FeasibilityPolicyCatalog.from_policies((ClassifiedsFeasibilityPolicy(),))
    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [_task()],
            instances_by_site={
                "classifieds": [
                    {
                        "site_name": "classifieds",
                        "benchmark": "visualwebarena",
                        "site_url": "https://classifieds.test",
                        "preflight_request_context": {},
                    }
                ]
            },
            request_context_factory=factory,
            feasibility_policy_catalog=catalog,
        )
    )

    assert keep and not dropped
    assert len(context_options) == 1
    assert not any(
        key in context_options[0]
        for key in ("storage_state", "extra_http_headers", "http_credentials")
    )
    assert calls == ["https://classifieds.test/index.php?page=item&id=12085"]
