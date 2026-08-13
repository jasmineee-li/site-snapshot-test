"""GitLab read-surface verification planning."""

from __future__ import annotations

from warp_taskgen.seeding.site_contracts import EditorSeedResult
from warp_taskgen.sites.read_surface import (
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
    build_read_surface_plan,
)


class GitLabReadSurfaceCapability:
    def build_read_surface_plan(
        self,
        *,
        seed_result: EditorSeedResult,
        signature: str,
        origin: str,
    ) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure:
        return build_read_surface_plan(
            site="gitlab",
            seed_result=seed_result,
            signature=signature,
            origin=origin,
            identity_keys=("note_id", "issue_iid", "project_id"),
        )


__all__ = ["GitLabReadSurfaceCapability"]
