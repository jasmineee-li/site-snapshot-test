"""Compatibility facade for the canonical WebArena Phase 2c policy."""

from __future__ import annotations

from worldsim.phase_2.phase_2c import webarena_policy as _canonical
from worldsim.phase_2.phase_2c.policy import default_feasibility_policy_catalog

DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO = _canonical.DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO
WebArenaFeasibilityPolicy = _canonical.WebArenaFeasibilityPolicy
classify_webarena_probe = _canonical.classify_webarena_probe
clean_gitlab_project_path = _canonical.clean_gitlab_project_path
dedupe_targets = _canonical.dedupe_targets
editor_surface_path = _canonical.editor_surface_path
first_value = _canonical.first_value
location_is_login = _canonical.location_is_login
looks_like_login_stub = _canonical.looks_like_login_stub
render_anchor_tokens = _canonical.render_anchor_tokens
safe_redirect_detail = _canonical.safe_redirect_detail
task_probe_url = _canonical.task_probe_url


def register_webarena_policies() -> None:
    """Explicitly repopulate the legacy policy registry."""
    from worldsim.phase_2c.policy import register_feasibility_policy

    for policy in default_feasibility_policy_catalog().policies.values():
        register_feasibility_policy(policy)


__all__ = [
    "DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO",
    "WebArenaFeasibilityPolicy",
    "classify_webarena_probe",
    "clean_gitlab_project_path",
    "dedupe_targets",
    "editor_surface_path",
    "first_value",
    "location_is_login",
    "looks_like_login_stub",
    "register_webarena_policies",
    "render_anchor_tokens",
    "safe_redirect_detail",
    "task_probe_url",
]
