from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from worldsim.seeding.site_contracts import EditorSeedResult
from worldsim.sites import (
    CanonicalRoute,
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
    SiteCatalog,
    TargetingContext,
)
from worldsim.sites.read_surface import build_read_surface_plan


def _seed_result(
    *urls: object,
    editor_method: str = "create_message",
    **tokens: object,
) -> EditorSeedResult:
    return EditorSeedResult.from_mapping(
        {
            "read_surface_urls": list(urls),
            "read_surface_provenance_source": "editor_result",
            **tokens,
        },
        editor_method=editor_method,
    )


class FakeReadSurfaceSite:
    site = "fake"
    supported_benchmarks = frozenset({"webarena_verified"})

    def validate(self) -> None:
        return None

    def validate_task(self, task: Mapping[str, Any]) -> tuple[str, str] | None:
        del task
        return None

    def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
        return (
            CanonicalRoute(
                id="fake.message",
                site=context.site,
                kind="message",
                allowed_start_url_patterns=("/messages/{comment_id}",),
                anchor_examples=({"comment_id": "17"},),
            ),
        )

    def match(
        self,
        url: str,
        task: Mapping[str, Any],
        context: TargetingContext,
    ) -> tuple[str, dict[str, Any]] | None:
        del url, task, context
        return None

    def reconstruct(
        self,
        kind: str,
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> str | None:
        del kind, anchors, context
        return None

    def is_listing(self, kind: str) -> bool:
        del kind
        return False

    def listing_start_url(
        self,
        kind: str,
        resolved_url: str,
        fallback_url: str | None,
    ) -> str | None:
        del kind, resolved_url
        return fallback_url

    def build_read_surface_plan(
        self,
        *,
        seed_result: EditorSeedResult,
        signature: str,
        origin: str,
    ) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure:
        return build_read_surface_plan(
            site=self.site,
            seed_result=seed_result,
            signature=signature,
            origin=origin,
            identity_keys=("comment_id",),
        )


def test_active_sites_build_safe_immutable_plans() -> None:
    cases = (
        (
            "gitlab",
            "https://gitlab.local",
            _seed_result(
                "https://external.invalid/project/-/issues/7",
                "/project/-/issues/7",
                editor_method="create_issue_note",
                note_id="31",
                issue_iid="7",
            ),
            "note_id",
        ),
        (
            "reddit",
            "https://reddit.local",
            _seed_result(
                "/f/books/9",
                editor_method="create_comment",
                comment_id="44",
                submission_id="9",
            ),
            "comment_id",
        ),
    )

    for site, origin, seed_result, identity_key in cases:
        plan = (
            SiteCatalog()
            .bind(site=site, origin=origin)
            .read_surface_plan(
                seed_result=seed_result,
                signature="unique payload body",
            )
        )
        assert isinstance(plan, ReadSurfaceVerificationPlan)
        assert plan.urls[0].startswith(f"{origin}/")
        assert all("external.invalid" not in url for url in plan.urls)
        assert plan.verification_mode == "seed_resource"
        assert plan.identity_tokens[identity_key]
        with pytest.raises(TypeError):
            plan.identity_tokens[identity_key] = "changed"  # type: ignore[index]


def test_read_surface_plan_rejects_unsafe_or_missing_evidence() -> None:
    bound = SiteCatalog().bind(site="gitlab", origin="https://gitlab.local")

    for seed_result, reason in (
        (_seed_result("https://attacker.invalid/project/1"), "foreign_read_surface"),
        (_seed_result("http://gitlab.local/project/1"), "foreign_read_surface"),
        (_seed_result("not-a-local-path", 7), "missing_read_surface"),
    ):
        failure = bound.read_surface_plan(
            seed_result=seed_result,
            signature="payload body",
        )
        assert isinstance(failure, ReadSurfacePlanFailure)
        assert failure.reason == reason

    missing_signature = bound.read_surface_plan(
        seed_result=_seed_result("/project/1"),
        signature="",
    )
    assert isinstance(missing_signature, ReadSurfacePlanFailure)
    assert missing_signature.reason == "invalid_read_surface_plan"


def test_plan_snapshots_surface_sequence() -> None:
    mutable_surfaces = list(_seed_result("/messages/17").read_surfaces)
    plan = ReadSurfaceVerificationPlan(
        site="fake",
        surfaces=mutable_surfaces,  # type: ignore[arg-type]
        signature="fake payload",
        verification_mode="body_text",
        identity_tokens={},
    )

    mutable_surfaces.clear()
    assert len(plan.surfaces) == 1
    assert isinstance(plan.surfaces, tuple)


def test_injected_fake_site_plans_without_global_registration() -> None:
    catalog = SiteCatalog([FakeReadSurfaceSite()])
    plan = catalog.bind(site="fake", origin="https://fake.local").read_surface_plan(
        seed_result=_seed_result("/messages/17", comment_id="17"),
        signature="fake payload",
    )

    assert isinstance(plan, ReadSurfaceVerificationPlan)
    assert plan.site == "fake"
    assert plan.urls == ("https://fake.local/messages/17",)
    assert plan.identity_tokens == {"comment_id": "17"}
    assert SiteCatalog().sites == ("gitlab", "reddit")


def test_bound_site_rejects_foreign_surface_returned_by_adapter() -> None:
    class UnsafeSite(FakeReadSurfaceSite):
        def build_read_surface_plan(
            self,
            *,
            seed_result: EditorSeedResult,
            signature: str,
            origin: str,
        ) -> ReadSurfaceVerificationPlan:
            del seed_result, origin
            safe_shape = _seed_result("https://attacker.invalid/messages/17")
            return ReadSurfaceVerificationPlan(
                site=self.site,
                surfaces=safe_shape.read_surfaces,
                signature=signature,
                verification_mode="body_text",
                identity_tokens={},
            )

    failure = (
        SiteCatalog([UnsafeSite()])
        .bind(
            site="fake",
            origin="https://fake.local",
        )
        .read_surface_plan(
            seed_result=_seed_result("/messages/17"),
            signature="fake payload",
        )
    )

    assert isinstance(failure, ReadSurfacePlanFailure)
    assert failure.reason == "foreign_read_surface"
