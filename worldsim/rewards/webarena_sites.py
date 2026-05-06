from __future__ import annotations

from collections.abc import Mapping

from worldsim.placeholders import placeholder_for_site

SUPPORTED_WEBARENA_PLACEHOLDER_SITES: Mapping[str, str] = {
    "__GITLAB__": "gitlab",
    "__REDDIT__": "reddit",
    "__POSTMILL__": "reddit",
}


def build_supported_webarena_environments(
    url_placeholders: Mapping[str, str] | None,
    *,
    site_name: str = "",
    site_url: str = "",
) -> dict[str, list[str]]:
    """Build evaluator environments for the WebArena sites WorldSim supports."""
    explicit = dict(url_placeholders or {})
    normalized_site_name = site_name.strip().lower()
    if normalized_site_name == "postmill" and "__REDDIT__" not in explicit and site_url:
        explicit["__REDDIT__"] = site_url

    primary_placeholder = placeholder_for_site(normalized_site_name)
    if (
        primary_placeholder in SUPPORTED_WEBARENA_PLACEHOLDER_SITES
        and primary_placeholder not in explicit
        and site_url
    ):
        explicit[primary_placeholder] = site_url

    environments: dict[str, list[str]] = {}
    for placeholder, evaluator_site in SUPPORTED_WEBARENA_PLACEHOLDER_SITES.items():
        url = explicit.get(placeholder)
        if url:
            environments.setdefault(evaluator_site, []).append(url)
    return environments
