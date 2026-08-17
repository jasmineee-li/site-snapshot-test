"""Aggregate pure Site Composition declarations."""

from __future__ import annotations

from warp_taskgen.site_composition_contracts import SiteComposition
from warp_taskgen.site_compositions.classifieds import classifieds_site_composition
from warp_taskgen.site_compositions.gitlab import gitlab_site_composition
from warp_taskgen.site_compositions.reddit import reddit_site_composition


def default_site_compositions() -> tuple[SiteComposition, ...]:
    """Return the explicit GitLab, Reddit, and Classifieds projections."""

    return (
        gitlab_site_composition(),
        reddit_site_composition(),
        classifieds_site_composition(),
    )


__all__ = ["default_site_compositions"]
