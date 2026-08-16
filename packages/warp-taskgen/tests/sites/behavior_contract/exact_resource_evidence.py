"""Assertions for Exact Resource Evidence and its fail-closed identities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from warp_taskgen.seeding.site_contracts import EditorSeedResult
from warp_taskgen.sites import BoundSite, ReadbackDecision, ReadbackObservation
from warp_taskgen.sites.read_surface import ReadSurfacePlanFailure


@dataclass(frozen=True)
class ExactResourceEvidenceCase:
    """Immutable expected identities for one created child resource."""

    benchmark: str
    site: str
    origin: str
    parent_kind: str
    parent_id: str
    parent_path: str
    resource_kind: str
    resource_id: str
    actor: str
    body: str
    signature: str
    action_kind: str
    action_path: str


def assert_exact_resource_evidence(
    bound_site: BoundSite,
    case: ExactResourceEvidenceCase,
    seed_result: EditorSeedResult,
    *,
    foreign_read_surface_url: str,
) -> None:
    """Check mutation/readback/visibility identity and reject near misses."""

    body_sha256 = hashlib.sha256(case.body.encode("utf-8")).hexdigest()
    assert dict(seed_result.write_tokens) == {
        "actor": case.actor,
        "body_sha256": body_sha256,
        "comment_id": case.resource_id,
        "thread_id": case.parent_id,
    }
    assert len(seed_result.created_resources) == 1
    created = seed_result.created_resources[0]
    assert (created.kind, created.id, created.parent_url) == (
        case.resource_kind,
        case.resource_id,
        case.parent_path,
    )
    assert seed_result.read_surface_urls == (case.parent_path,)
    plan = bound_site.read_surface_plan(seed_result=seed_result, signature=case.signature)
    assert getattr(plan, "site", None) == case.site
    assert getattr(plan, "persist_readback_identity_tokens", False) is True
    assert tuple(getattr(plan, "urls", ())) == (f"{case.origin}{case.parent_path}",)
    assert dict(getattr(plan, "identity_tokens", {})) == {
        "actor": case.actor,
        "body_sha256": body_sha256,
        "comment_id": case.resource_id,
        "thread_id": case.parent_id,
    }

    foreign_plan = bound_site.read_surface_plan(
        seed_result=replace(seed_result, read_surface_urls=(foreign_read_surface_url,)),
        signature=case.signature,
    )
    assert isinstance(foreign_plan, ReadSurfacePlanFailure)
    assert foreign_plan.reason == "foreign_read_surface"

    payload = {
        "site": case.site,
        "origin": case.origin,
        "parent_kind": case.parent_kind,
        "parent_id": case.parent_id,
        "thread_id": case.parent_id,
        "resource_kind": case.resource_kind,
        "comment_id": case.resource_id,
        "actor": case.actor,
        "body": case.body,
        "body_sha256": body_sha256,
        "signature": case.signature,
        "visible": True,
        "match_count": 1,
    }
    observation = ReadbackObservation(
        "comment_visibility",
        plan.identity_tokens,
        payload,
        signature=case.signature,
    )
    decision = bound_site.interpret_readback(observation)
    assert isinstance(decision, ReadbackDecision)
    assert decision.verified is True
    assert decision.reason == "exact_comment_visible"

    for field, value in (
        ("site", "different-site"),
        ("origin", "https://foreign.invalid"),
        ("parent_id", "999"),
        ("thread_id", "999"),
        ("parent_kind", "submission"),
        ("resource_kind", "thread"),
        ("actor", "different-actor"),
        ("comment_id", "stale-foreign-id"),
        ("body", "different body"),
        ("body", f"different prefix {case.signature}"),
        ("body_sha256", "0" * 64),
        ("match_count", 2),
        ("reader_context", "regular_participant_writer"),
        ("writer_cookie_names", ("session",)),
        ("visible", False),
    ):
        negative_payload = dict(payload)
        negative_payload[field] = value
        negative = bound_site.interpret_readback(
            ReadbackObservation(
                "comment_visibility",
                plan.identity_tokens,
                negative_payload,
                signature=case.signature,
            )
        )
        assert isinstance(negative, ReadbackDecision)
        assert negative.verified is False, field
