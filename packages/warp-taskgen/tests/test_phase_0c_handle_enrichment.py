from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from warp_taskgen.phases.phase_0c_handle_enrichment import (
    HandleEnrichmentError,
    _site_url_candidates,
    enrich_gitlab_handles,
    enrich_gitlab_projects,
    merge_gitlab_project_inventory_into_profile,
    merge_into_agent_context,
)


def _mock_response(payload: list[dict[str, Any]], *, next_page: int | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.headers = {"X-Next-Page": str(next_page)} if next_page else {}
    return resp


@pytest.fixture
def auth_config() -> dict[str, Any]:
    return {
        "type": "bearer_token",
        "header_name": "PRIVATE-TOKEN",
        "token_generator": "gitlab_pat",
        "credentials": {"username": "byteblaze", "password": "hello1234"},
    }


def test_enrich_collects_user_and_group_handles(auth_config: dict[str, Any]) -> None:
    users = [{"username": "root"}, {"username": "byteblaze"}, {"username": "primer"}]
    groups = [{"full_path": "a11yproject"}, {"full_path": "design"}]

    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            return_value="glpat-deadbeef",
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [
            _mock_response(users),
            _mock_response(groups),
        ]
        result = enrich_gitlab_handles("http://gitlab.local", auth_config)

    assert result == {
        "user_handles": ["byteblaze", "primer", "root"],
        "group_handles": ["a11yproject", "design"],
    }


def test_enrich_collects_namespace_qualified_projects(auth_config: dict[str, Any]) -> None:
    projects = [
        {
            "id": 179,
            "name": "a11y-webring.club",
            "path": "a11y-webring.club",
            "path_with_namespace": "a11yproject/a11y-webring.club",
            "namespace": {"path": "a11yproject", "full_path": "a11yproject"},
        },
        {
            "id": 180,
            "name": "bare",
            "path": "bare",
            "path_with_namespace": "bare",
        },
    ]

    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            return_value="glpat-deadbeef",
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.return_value = _mock_response(projects)
        result = enrich_gitlab_projects("http://gitlab.local", auth_config)

    assert result == {
        "projects": [
            {
                "id": "179",
                "name": "a11y-webring.club",
                "path": "a11y-webring.club",
                "path_with_namespace": "a11yproject/a11y-webring.club",
                "full_path": "a11yproject/a11y-webring.club",
                "namespace": "a11yproject",
                "namespace_full_path": "a11yproject",
            }
        ]
    }
    assert mock_get.call_args.args[0] == "http://gitlab.local/api/v4/projects"


def test_enrich_projects_prefers_runtime_web_host(auth_config: dict[str, Any]) -> None:
    projects = [
        {
            "id": 179,
            "path_with_namespace": "a11yproject/a11y-webring.club",
        }
    ]
    acquired_bases: list[str] = []

    def fake_acquire_token(config: dict[str, Any], site_url: str) -> str:
        acquired_bases.append(site_url)
        return "glpat-deadbeef"

    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            side_effect=fake_acquire_token,
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.return_value = _mock_response(projects)
        result = enrich_gitlab_projects(
            "http://3.12.221.9:8023",
            auth_config,
            runtime_web_host="172.17.0.1",
        )

    assert acquired_bases == ["http://172.17.0.1:8023"]
    assert mock_get.call_args.args[0] == "http://172.17.0.1:8023/api/v4/projects"
    assert result["projects"][0]["path_with_namespace"] == "a11yproject/a11y-webring.club"


def test_enrich_prefers_runtime_web_host_for_host_side_api(auth_config: dict[str, Any]) -> None:
    users = [{"username": "root"}]
    groups = [{"full_path": "a11yproject"}]
    acquired_bases: list[str] = []

    def fake_acquire_token(config: dict[str, Any], site_url: str) -> str:
        acquired_bases.append(site_url)
        return "glpat-deadbeef"

    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            side_effect=fake_acquire_token,
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [_mock_response(users), _mock_response(groups)]
        result = enrich_gitlab_handles(
            "http://3.12.221.9:8023",
            auth_config,
            runtime_web_host="172.17.0.1",
        )

    assert acquired_bases == ["http://172.17.0.1:8023"]
    assert [call.args[0] for call in mock_get.call_args_list] == [
        "http://172.17.0.1:8023/api/v4/users",
        "http://172.17.0.1:8023/api/v4/groups",
    ]
    assert result == {"user_handles": ["root"], "group_handles": ["a11yproject"]}


def test_enrich_falls_back_to_original_url_after_runtime_host_failure(
    auth_config: dict[str, Any],
) -> None:
    users = [{"username": "root"}]
    groups = [{"full_path": "a11yproject"}]
    acquired_bases: list[str] = []

    def fake_acquire_token(config: dict[str, Any], site_url: str) -> str:
        acquired_bases.append(site_url)
        if site_url == "http://172.17.0.1:8023":
            raise RuntimeError("host-local transient failure")
        return "glpat-deadbeef"

    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            side_effect=fake_acquire_token,
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [_mock_response(users), _mock_response(groups)]
        result = enrich_gitlab_handles(
            "http://3.12.221.9:8023",
            auth_config,
            runtime_web_host="172.17.0.1",
        )

    assert acquired_bases == ["http://172.17.0.1:8023", "http://3.12.221.9:8023"]
    assert result == {"user_handles": ["root"], "group_handles": ["a11yproject"]}


def test_site_url_candidates_dedupes_matching_runtime_host() -> None:
    assert _site_url_candidates("http://172.17.0.1:8023/", "172.17.0.1") == [
        "http://172.17.0.1:8023"
    ]


def test_enrich_paginates_users(auth_config: dict[str, Any]) -> None:
    page1 = [{"username": f"u{i}"} for i in range(100)]
    page2 = [{"username": "u100"}]
    groups = [{"full_path": "team"}]

    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            return_value="glpat-deadbeef",
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [
            _mock_response(page1, next_page=2),
            _mock_response(page2),
            _mock_response(groups),
        ]
        result = enrich_gitlab_handles("http://gitlab.local", auth_config)

    assert "u100" in result["user_handles"]
    assert len(result["user_handles"]) == 101


def test_enrich_filters_subgroup_full_paths(auth_config: dict[str, Any]) -> None:
    """Top-level segments only — paths with '/' are subgroups, not addressable via /<segment>."""
    users: list[dict[str, Any]] = []
    groups = [
        {"full_path": "design"},
        {"full_path": "design/subteam"},
        {"full_path": "design/subteam/leaf"},
        {"full_path": "primer"},
    ]
    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            return_value="glpat-deadbeef",
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [_mock_response(users), _mock_response(groups)]
        result = enrich_gitlab_handles("http://gitlab.local", auth_config)

    assert result["group_handles"] == ["design", "primer"]


def test_enrich_dedupes_and_sorts(auth_config: dict[str, Any]) -> None:
    users = [{"username": "byteblaze"}, {"username": "byteblaze"}, {"username": "root"}]
    groups: list[dict[str, Any]] = []
    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            return_value="glpat-deadbeef",
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [_mock_response(users), _mock_response(groups)]
        result = enrich_gitlab_handles("http://gitlab.local", auth_config)
    assert result["user_handles"] == ["byteblaze", "root"]


def test_enrich_raises_on_missing_auth() -> None:
    with pytest.raises(HandleEnrichmentError, match="auth_config is required"):
        enrich_gitlab_handles("http://gitlab.local", None)


def test_enrich_raises_on_missing_url(auth_config: dict[str, Any]) -> None:
    with pytest.raises(HandleEnrichmentError, match="site_url is required"):
        enrich_gitlab_handles("", auth_config)


def test_enrich_raises_on_token_acquisition_failure(auth_config: dict[str, Any]) -> None:
    with patch(
        "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(HandleEnrichmentError, match="could not acquire gitlab token"):
            enrich_gitlab_handles("http://gitlab.local", auth_config)


def test_enrich_raises_on_4xx(auth_config: dict[str, Any]) -> None:
    resp = MagicMock()
    resp.status_code = 403
    resp.headers = {}
    with (
        patch(
            "warp_taskgen.phases.phase_0c_handle_enrichment.acquire_token",
            return_value="glpat-deadbeef",
        ),
        patch("warp_taskgen.phases.phase_0c_handle_enrichment.requests.get", return_value=resp),
    ):
        with pytest.raises(HandleEnrichmentError, match="HTTP 403"):
            enrich_gitlab_handles("http://gitlab.local", auth_config)


def test_merge_into_agent_context_creates_gitlab_block() -> None:
    ctx = {
        "response_format": {"requires_structured_output": True, "description": "x"},
        "authentication": {"pre_authenticated": True, "description": "x"},
        "site_context": {"platform_name": "GitLab", "description": "x"},
    }
    result = merge_into_agent_context(
        ctx,
        {"user_handles": ["root"], "group_handles": ["design"]},
    )
    assert result["gitlab"] == {
        "user_handles": ["root"],
        "group_handles": ["design"],
    }
    assert result["response_format"] == ctx["response_format"]


def test_merge_preserves_existing_gitlab_block() -> None:
    ctx = {"gitlab": {"existing_field": "kept"}}
    result = merge_into_agent_context(
        ctx,
        {"user_handles": ["a"], "group_handles": ["b"]},
    )
    assert result["gitlab"]["existing_field"] == "kept"
    assert result["gitlab"]["user_handles"] == ["a"]
    assert result["gitlab"]["group_handles"] == ["b"]


def test_merge_gitlab_project_inventory_into_profile_preserves_existing_entities() -> None:
    profile = {"available_entities": {"users": [{"username": "root"}]}}
    result = merge_gitlab_project_inventory_into_profile(
        profile,
        {"projects": [{"id": "179", "path_with_namespace": "a11yproject/a11y-webring.club"}]},
    )

    assert result["available_entities"]["users"] == [{"username": "root"}]
    assert result["available_entities"]["projects"] == [
        {"id": "179", "path_with_namespace": "a11yproject/a11y-webring.club"}
    ]
