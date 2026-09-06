"""Phase 0c: host-side enrichment of a profile with observed handles and forums.

Orchestrates the GitLab handle, GitLab project, and Reddit forum enrichment leaf
helpers against the host-side runtime host for one Benchmark Instance.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.placeholders import normalize_site_name

logger = logging.getLogger(__name__)


def _enrich_agent_context_with_handles(
    *,
    site_name: str,
    agent_context: dict[str, Any],
    instance: BenchmarkInstance | None,
) -> dict[str, Any]:
    """For gitlab sites, enumerate user/group handles via the live API.

    Returns ``agent_context`` unchanged for non-gitlab sites or when the
    instance is unavailable / unauthenticated. Enrichment failures are
    logged at warning level — Phase 0c does not abort on a transient
    handle-enumeration outage. Phase 2's resolver categorizes downstream
    drops cleanly when the lists are absent.
    """
    if normalize_site_name(site_name) != "gitlab":
        return agent_context
    if instance is None:
        logger.info(
            "Phase 0c: site %r has no instance config; skipping handle enrichment", site_name
        )
        return agent_context
    auth_config = instance.api_auth or instance.auth
    if not auth_config:
        logger.info(
            "Phase 0c: site %r instance has no api_auth/auth; skipping handle enrichment",
            site_name,
        )
        return agent_context

    from warp_taskgen.phases.phase_0c_handle_enrichment import (
        HandleEnrichmentError,
        enrich_gitlab_handles,
        merge_into_agent_context,
    )

    try:
        handles = enrich_gitlab_handles(
            instance.site_url,
            auth_config,
            runtime_web_host=_host_side_runtime_host(),
        )
    except HandleEnrichmentError as exc:
        logger.warning(
            "Phase 0c: gitlab handle enrichment for site %r failed: %s",
            site_name,
            exc,
        )
        return agent_context

    logger.info(
        "Phase 0c: site %r enriched with %d user_handles and %d group_handles",
        site_name,
        len(handles.get("user_handles", [])),
        len(handles.get("group_handles", [])),
    )
    return merge_into_agent_context(agent_context, handles)


def _enrich_gitlab_profile_with_projects(
    *,
    site_name: str,
    profile: dict[str, Any],
    instance: BenchmarkInstance | None,
) -> dict[str, Any]:
    """For gitlab sites, attach namespace-qualified project inventory."""
    if normalize_site_name(site_name) != "gitlab":
        return profile
    if instance is None:
        logger.info(
            "Phase 0c: site %r has no instance config; skipping gitlab project enrichment",
            site_name,
        )
        return profile
    auth_config = instance.api_auth or instance.auth
    if not auth_config:
        logger.info(
            "Phase 0c: site %r instance has no api_auth/auth; skipping gitlab project enrichment",
            site_name,
        )
        return profile

    from warp_taskgen.phases.phase_0c_handle_enrichment import (
        HandleEnrichmentError,
        enrich_gitlab_projects,
        merge_gitlab_project_inventory_into_profile,
    )

    try:
        inventory = enrich_gitlab_projects(
            instance.site_url,
            auth_config,
            runtime_web_host=_host_side_runtime_host(),
        )
    except HandleEnrichmentError as exc:
        logger.warning(
            "Phase 0c: gitlab project enrichment for site %r failed: %s",
            site_name,
            exc,
        )
        return profile

    projects = inventory.get("projects", [])
    if not projects:
        logger.warning(
            "Phase 0c: gitlab project enrichment for site %r found no projects",
            site_name,
        )
        return profile
    logger.info(
        "Phase 0c: site %r enriched with %d gitlab projects",
        site_name,
        len(projects),
    )
    return merge_gitlab_project_inventory_into_profile(profile, inventory)


def _enrich_reddit_profile_with_forums(
    *,
    site_name: str,
    profile: dict[str, Any],
    instance: BenchmarkInstance | None,
    instances: list[BenchmarkInstance] | None = None,
) -> dict[str, Any]:
    """For reddit sites, attach live-reachable forum inventory to the profile."""
    if normalize_site_name(site_name) != "reddit":
        return profile
    inventory_instances = list(instances or ([instance] if instance is not None else []))
    inventory_instances = [
        item for item in inventory_instances if item is not None and item.db_connection
    ]
    if not inventory_instances:
        logger.info(
            "Phase 0c: site %r has no instance config; skipping reddit forum enrichment",
            site_name,
        )
        return profile

    from warp_taskgen.phases.phase_0c_reddit_enrichment import (
        RedditInventoryEnrichmentError,
        common_reddit_forum_inventory,
        enrich_reddit_forums,
        merge_reddit_inventory_into_profile,
    )

    try:
        inventories = [
            enrich_reddit_forums(
                item.site_url,
                item.db_connection,
                runtime_db_host=_host_side_runtime_host(),
            )
            for item in inventory_instances
        ]
    except RedditInventoryEnrichmentError as exc:
        logger.warning(
            "Phase 0c: reddit forum enrichment for site %r failed: %s",
            site_name,
            exc,
        )
        return profile

    inventory = common_reddit_forum_inventory(inventories)
    forums = inventory.get("forums", [])
    if not forums:
        logger.warning(
            "Phase 0c: reddit forum enrichment for site %r found no forums common to %d replica(s)",
            site_name,
            len(inventory_instances),
        )
        return profile
    logger.info(
        "Phase 0c: site %r enriched with %d reachable reddit forums common to %d replica(s)",
        site_name,
        len(forums),
        len(inventory_instances),
    )
    return merge_reddit_inventory_into_profile(profile, inventory)


def _host_side_runtime_host() -> str | None:
    """Return an optional host-local hostname for Phase 0c enrichment.

    Modal receives public/proxied web URLs for Phase 0c live browsing, but the
    enrichment hooks run in the orchestrator process. Registered r5 jobs
    export ``WORLDSIM_ORCHESTRATOR_HOST`` so host-side DB/API queries can use
    the same local network view as Phase 2c/4 instead of trying to hairpin
    through the public EC2 address.
    """

    for name in ("WORLDSIM_ORCHESTRATOR_HOST", "WORLDSIM_REMOTE_ORCHESTRATOR_HOST"):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
