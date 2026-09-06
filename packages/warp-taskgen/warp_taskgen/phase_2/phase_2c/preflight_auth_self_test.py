"""Phase 2c preflight auth self-test: does the current request context still
hold live authenticated browser state for this site?

Split out of :mod:`source_data_preflight`. The source-data preflight quarantines
tasks whose benign surface is deterministically broken; that verdict is only
trustworthy while the benign agent's storage state is still accepted. This
module answers that narrower question against the site policy's cheap
authenticated endpoint, so a stale cookie is diagnosed as auth staleness rather
than mass-quarantined as broken source data.
"""

from __future__ import annotations

from typing import Any

from warp_taskgen.phase_2.phase_2c.policy import (
    FeasibilityPolicyCatalog,
    PreflightClassification,
    resolve_feasibility_policy,
)
from warp_taskgen.phase_2.phase_2c.source_data_preflight import (
    DEFAULT_PREFLIGHT_TIMEOUT_S,
    _probe_one,
)
from warp_taskgen.phases.phase_2_reachability import resolve_start_url


def auth_self_test_path(
    site: str,
    *,
    benchmark: str = "webarena_verified",
    feasibility_policy_catalog: FeasibilityPolicyCatalog,
) -> str | None:
    """Return a cheap authenticated endpoint path for sites that need one."""
    policy = resolve_feasibility_policy(
        benchmark,
        str(site or "").strip().lower(),
        feasibility_policy_catalog=feasibility_policy_catalog,
    )
    return policy.auth_self_test_path() if policy is not None else None


async def self_test_preflight_auth(
    *,
    request_context: Any,
    site: str,
    site_url: str,
    benchmark: str = "webarena_verified",
    timeout_s: float = DEFAULT_PREFLIGHT_TIMEOUT_S,
    feasibility_policy_catalog: FeasibilityPolicyCatalog,
) -> PreflightClassification | None:
    """Probe whether the current request context has live browser auth.

    Returns ``None`` for sites whose source-data preflight does not depend on
    authenticated browser state. For GitLab, ``reachable`` means the storage
    state is accepted; ``login_redirect``/``auth_missing`` means it is stale.
    """
    path = auth_self_test_path(
        site,
        benchmark=benchmark,
        feasibility_policy_catalog=feasibility_policy_catalog,
    )
    if path is None:
        return None
    base = str(site_url or "").strip()
    if not base:
        return PreflightClassification(
            kind="host_unreachable",
            quarantine=False,
            http_status=None,
            detail="auth self-test has no site_url",
        )
    return await _probe_one(
        request_context=request_context,
        url=resolve_start_url(path, base),
        timeout_s=timeout_s,
        policy=resolve_feasibility_policy(
            benchmark,
            str(site or "").strip().lower(),
            feasibility_policy_catalog=feasibility_policy_catalog,
        ),
    )
