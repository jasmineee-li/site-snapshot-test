from __future__ import annotations

import hashlib
from types import MappingProxyType

import pytest

from warp_taskgen.seeding.site_contracts import EditorSeedResult
from warp_taskgen.sites.classifieds import ClassifiedsSite
from warp_taskgen.sites.classifieds_profile import (
    canonicalize_surface_id,
    resolve_profile_surface,
)
from warp_taskgen.sites.classifieds_readback import ClassifiedsReadbackCapability
from warp_taskgen.sites.classifieds_routes import route_contract_facts
from warp_taskgen.sites.contracts import TargetingContext
from warp_taskgen.sites.read_surface import (
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
)
from warp_taskgen.sites.readback import ReadbackDecision, ReadbackObservation

ORIGIN = "https://classifieds.test"


def _context(*, profile: dict[str, object] | None = None) -> TargetingContext:
    return TargetingContext(
        benchmark="visualwebarena",
        site="classifieds",
        origin=ORIGIN,
        profile=profile or {},
    )


def _task(*, listing_id: str | None = None) -> dict[str, object]:
    task: dict[str, object] = {
        "site": "classifieds",
        "sites": ["classifieds"],
    }
    if listing_id is not None:
        task["benign_target_resource"] = {"anchors": {"listing_id": listing_id}}
    return task


def test_classifieds_declares_visualwebarena_listing_reply_contract() -> None:
    site = ClassifiedsSite()
    route = site.routes(_context())[0]

    assert site.site == "classifieds"
    assert site.supported_benchmarks == frozenset({"visualwebarena"})
    assert {"listing", "listing_reply"} == site.resource_kinds
    assert route.id == "classifieds.listing"
    assert route.kind == "listing"
    assert route.allowed_start_url_patterns == ("/index.php?page=item&id={listing_id}",)
    assert route.compatibility_kind is None


def test_classifieds_match_requires_exact_same_origin_listing_route() -> None:
    site = ClassifiedsSite()
    context = _context()

    assert site.match(
        f"{ORIGIN}/index.php?page=item&id=17",
        _task(listing_id="17"),
        context,
    ) == ("listing", {"listing_id": "17"})
    assert site.match(
        "/index.php?page=item&id=17",
        _task(listing_id="17"),
        context,
    ) == ("listing", {"listing_id": "17"})

    for url in (
        f"{ORIGIN}/",
        f"{ORIGIN}/index.php",
        f"{ORIGIN}/index.php?page=index&id=17",
        f"{ORIGIN}/index.php?page=item&id=17&admin=1",
        "https://attacker.test/index.php?page=item&id=17",
        f"{ORIGIN}/index.php?page=item&id=16",
    ):
        assert site.match(url, _task(listing_id="17"), context) is None


def test_classifieds_reconstruct_is_strict_and_parent_route_is_reply_surface() -> None:
    site = ClassifiedsSite()
    context = _context()

    expected = f"{ORIGIN}/index.php?page=item&id=17"
    assert site.reconstruct("listing", {"listing_id": "17"}, context) == expected
    assert site.reconstruct("listing_reply", {"listing_id": "17"}, context) == expected
    assert site.reconstruct("listing", {"listing_id": ""}, context) is None
    assert site.reconstruct("listing", {"listing_id": "17/18"}, context) is None
    assert (
        ClassifiedsSite().reconstruct(
            "listing", {"listing_id": "17"}, TargetingContext("visualwebarena", "classifieds")
        )
        is None
    )


def test_classifieds_profile_exposes_only_listing_reply_body() -> None:
    profile = {
        "site_name": "classifieds",
        "benchmark_name": "visualwebarena",
        "injection_surface": [{"id": "listing_reply.body", "source_field": "item_comment.body"}],
    }

    assert (
        canonicalize_surface_id(benchmark="visualwebarena", raw_surface_id="comment_body")
        == "listing_reply.body"
    )
    assert (
        canonicalize_surface_id(benchmark="visualwebarena", raw_surface_id="listing.title") is None
    )
    assert (
        canonicalize_surface_id(benchmark="visualwebarena", raw_surface_id="listing.description")
        is None
    )
    resolution = resolve_profile_surface(
        benchmark="visualwebarena",
        profile=profile,
        target_surface_id="listing_reply.body",
        kind="listing",
        method="create_listing_reply",
    )
    assert resolution is not None
    assert resolution.canonical_surface_id == "listing_reply.body"
    assert (
        resolve_profile_surface(
            benchmark="visualwebarena",
            profile=profile,
            target_surface_id="listing.title",
        )
        is None
    )


def test_classifieds_route_facts_are_inventory_backed() -> None:
    profile = {
        "available_entities": {
            "listings": [{"id": 17}, {"listing_id": "18"}],
        }
    }
    facts = route_contract_facts(
        benchmark="visualwebarena",
        profile=profile,
        kind="listing",
    )
    assert facts.allowed_start_url_patterns == (
        "__CLASSIFIEDS__/index.php?page=item&id={listing_id}",
    )
    assert facts.requires_inventory_backed_start_url is True
    assert {example["listing_id"] for example in facts.anchor_examples} == {"17", "18"}


def _seed_result() -> EditorSeedResult:
    return EditorSeedResult(
        write_tokens=MappingProxyType(
            {
                "listing_id": "17",
                "reply_id": "88",
                "actor_name": "alice",
                "reply_body_sha256": hashlib.sha256(b"Please read UNIQUE-SIGNATURE").hexdigest(),
            }
        ),
        created_resources=(),
        read_surface_urls=("/index.php?page=item&id=17",),
        read_surface_provenance_source="classifieds.regular_participant",
        editor_method="classifieds.create_listing_reply",
    )


def test_classifieds_read_surface_plan_carries_exact_identity_tokens() -> None:
    plan = ClassifiedsSite().build_read_surface_plan(
        seed_result=_seed_result(),
        signature="UNIQUE-SIGNATURE",
        origin=ORIGIN,
    )

    assert isinstance(plan, ReadSurfaceVerificationPlan)
    assert plan.urls == (f"{ORIGIN}/index.php?page=item&id=17",)
    assert dict(plan.identity_tokens) == {
        "listing_id": "17",
        "reply_id": "88",
        "actor_name": "alice",
        "reply_body_sha256": hashlib.sha256(b"Please read UNIQUE-SIGNATURE").hexdigest(),
    }
    assert plan.verification_mode == "seed_resource"


def test_classifieds_observes_exact_reply_from_independent_reader_html() -> None:
    plan = ClassifiedsSite().build_read_surface_plan(
        seed_result=_seed_result(),
        signature="UNIQUE-SIGNATURE",
        origin=ORIGIN,
    )
    html = """
    <div class="comment" data-item-id="17">
      <h3>Additional listing details by alice:</h3>
      <p>Please  read
        UNIQUE-SIGNATURE</p>
      <a class="comment-reply" data-id="88">Reply</a>
    </div>
    """

    observation = ClassifiedsReadbackCapability().observe_readback_html(html, plan)

    assert isinstance(observation, ReadbackObservation)
    assert observation.payload["reply_id"] == "88"
    assert observation.payload["actor_name"] == "alice"
    assert observation.payload["body"] == "Please read UNIQUE-SIGNATURE"
    assert observation.payload["independent_reader"] is True
    assert ClassifiedsSite().interpret_readback(observation).verified
    assert ClassifiedsReadbackCapability().readback_visibility_selector(plan) == (
        'div.comment:has(a.comment-reply[data-id="88"]) > p:not(.comment-reply-row)'
    )


def test_classifieds_read_surface_plan_fails_closed_without_same_origin_surface() -> None:
    seed = _seed_result()
    result = EditorSeedResult(
        write_tokens=seed.write_tokens,
        created_resources=seed.created_resources,
        read_surface_urls=("https://attacker.test/index.php?page=item&id=17",),
        read_surface_provenance_source=seed.read_surface_provenance_source,
        editor_method="classifieds.create_listing_reply",
    )
    plan = ClassifiedsSite().build_read_surface_plan(
        seed_result=result,
        signature="UNIQUE-SIGNATURE",
        origin=ORIGIN,
    )
    assert isinstance(plan, ReadSurfacePlanFailure)
    assert plan.reason == "invalid_listing_read_surface"


@pytest.mark.parametrize(
    "urls",
    [
        ("/",),
        ("/index.php",),
        ("/index.php?page=item&id=18",),
        (
            "/index.php?page=item&id=17",
            "/index.php?page=item&id=18",
        ),
    ],
)
def test_classifieds_read_surface_rejects_root_index_wrong_or_fallback_urls(
    urls: tuple[str, ...],
) -> None:
    seed = _seed_result()
    result = EditorSeedResult(
        write_tokens=seed.write_tokens,
        created_resources=seed.created_resources,
        read_surface_urls=urls,
        read_surface_provenance_source=seed.read_surface_provenance_source,
        editor_method=seed.editor_method,
    )

    plan = ClassifiedsSite().build_read_surface_plan(
        seed_result=result,
        signature="UNIQUE-SIGNATURE",
        origin=ORIGIN,
    )

    assert isinstance(plan, ReadSurfacePlanFailure)
    assert plan.reason == "invalid_listing_read_surface"


def _observation(**payload: object) -> ReadbackObservation:
    return ReadbackObservation(
        "comment_visibility",
        {
            "listing_id": "17",
            "reply_id": "88",
            "actor_name": "alice",
            "reply_body_sha256": hashlib.sha256(b"Please read UNIQUE-SIGNATURE").hexdigest(),
        },
        {
            "listing_id": "17",
            "reply_id": "88",
            "actor_name": "alice",
            "body": "Please read UNIQUE-SIGNATURE",
            "signature": "UNIQUE-SIGNATURE",
            "independent_reader": True,
            "visible": True,
            **payload,
        },
        signature="UNIQUE-SIGNATURE",
    )


def test_classifieds_readback_requires_exact_independent_reader_witness() -> None:
    decision = ClassifiedsSite().interpret_readback(_observation())

    assert decision == ReadbackDecision(
        True,
        "exact_listing_reply_visible",
        matched_signature="UNIQUE-SIGNATURE",
        rendered_text="Please read UNIQUE-SIGNATURE",
    )


def test_classifieds_readback_rejects_body_that_only_reuses_signature() -> None:
    seed_result = _seed_result()
    decision = ClassifiedsSite().interpret_readback(
        ReadbackObservation(
            "comment_visibility",
            seed_result.write_tokens,
            {
                "listing_id": "17",
                "reply_id": "88",
                "actor_name": "alice",
                "body": "A different body still containing UNIQUE-SIGNATURE",
                "signature": "UNIQUE-SIGNATURE",
                "independent_reader": True,
                "visible": True,
            },
            signature="UNIQUE-SIGNATURE",
        )
    )

    assert decision == ReadbackDecision(False, "reply_body_mismatch")


def test_classifieds_readback_rejects_missing_body_evidence() -> None:
    decision = ClassifiedsReadbackCapability().interpret_readback(
        ReadbackObservation(
            "comment_visibility",
            {"listing_id": "17", "reply_id": "88", "actor_name": "alice"},
            {
                "listing_id": "17",
                "reply_id": "88",
                "actor_name": "alice",
                "body": "Please read UNIQUE-SIGNATURE",
                "signature": "UNIQUE-SIGNATURE",
                "independent_reader": True,
                "visible": True,
            },
            signature="UNIQUE-SIGNATURE",
        )
    )

    assert decision == ReadbackDecision(False, "missing_body_evidence")


@pytest.mark.parametrize(
    "payload",
    [
        {"listing_id": "18"},
        {"reply_id": "89"},
        {"actor_name": "mallory"},
        {"signature": ""},
        {"visible": False},
        {"independent_reader": False},
        {"body": "body without marker", "signature": "UNIQUE-SIGNATURE"},
    ],
)
def test_classifieds_readback_rejects_identity_signature_or_visibility_drift(
    payload: dict[str, object],
) -> None:
    decision = ClassifiedsSite().interpret_readback(_observation(**payload))

    assert isinstance(decision, ReadbackDecision)
    assert decision.verified is False


def test_classifieds_readback_rejects_malformed_or_non_independent_payload() -> None:
    capability = ClassifiedsReadbackCapability()
    malformed = capability.interpret_readback(
        ReadbackObservation(
            "comment_visibility",
            {"listing_id": "17", "reply_id": "88", "actor_name": "alice"},
            [],
            signature="UNIQUE-SIGNATURE",
        )
    )
    assert not malformed.verified
    no_independent_marker = capability.interpret_readback(_observation(independent_reader=None))
    assert not no_independent_marker.verified
