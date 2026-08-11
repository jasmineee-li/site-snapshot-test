"""Reddit/Postmill read-surface verification planning."""

from __future__ import annotations

from worldsim.seeding.site_contracts import EditorSeedResult
from worldsim.sites.read_surface import (
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
    build_read_surface_plan,
)


class RedditReadSurfaceCapability:
    def build_read_surface_plan(
        self,
        *,
        seed_result: EditorSeedResult,
        signature: str,
        origin: str,
    ) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure:
        return build_read_surface_plan(
            site="reddit",
            seed_result=seed_result,
            signature=signature,
            origin=origin,
            identity_keys=("comment_id", "submission_id"),
        )


__all__ = ["RedditReadSurfaceCapability"]
