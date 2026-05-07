from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worldsim.phases.phase_0c_reddit_enrichment import (
    RedditInventoryEnrichmentError,
    _db_connection_candidates,
    _read_forum_rows,
    common_reddit_forum_inventory,
    enrich_reddit_forums,
    merge_reddit_inventory_into_profile,
)


def _response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    return resp


def test_enrich_reddit_forums_filters_unreachable_forums() -> None:
    rows = [
        {"id": 1, "name": "books", "title": "Books"},
        {"id": 2, "name": "worcester", "title": "Worcester"},
        {"id": 3, "name": "DIY", "title": "DIY"},
    ]
    with (
        patch(
            "worldsim.phases.phase_0c_reddit_enrichment._read_forum_rows",
            return_value=rows,
        ),
        patch("worldsim.phases.phase_0c_reddit_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [_response(200), _response(404), _response(200)]
        result = enrich_reddit_forums("http://reddit.local", "mysql://u:p@h/db")

    assert result == {
        "forums": [
            {"id": "1", "name": "books", "title": "Books"},
            {"id": "3", "name": "DIY", "title": "DIY"},
        ]
    }
    assert [call.args[0] for call in mock_get.call_args_list] == [
        "http://reddit.local/f/books",
        "http://reddit.local/f/worcester",
        "http://reddit.local/f/DIY",
    ]


def test_enrich_reddit_forums_includes_reachable_empty_submissions() -> None:
    with (
        patch(
            "worldsim.phases.phase_0c_reddit_enrichment._read_forum_rows",
            return_value=[{"id": 1, "name": "books", "title": "Books"}],
        ),
        patch(
            "worldsim.phases.phase_0c_reddit_enrichment._read_empty_submission_rows_from_candidates",
            return_value=[
                {"id": 119, "forum_name": "books", "title": "empty thread"},
                {"id": 120, "forum_name": "books", "title": "gone thread"},
            ],
        ),
        patch("worldsim.phases.phase_0c_reddit_enrichment.requests.get") as mock_get,
    ):
        mock_get.side_effect = [_response(200), _response(200), _response(404)]
        result = enrich_reddit_forums("http://reddit.local", "mysql://u:p@h/db")

    assert result["submissions"] == [
        {
            "id": "119",
            "forum": "books",
            "title": "empty thread",
            "existing_comment_count": "0",
            "max_existing_comments_for_comment_seed": "0",
            "seeded_comment_visibility_candidate": "true",
        }
    ]
    assert [call.args[0] for call in mock_get.call_args_list] == [
        "http://reddit.local/f/books",
        "http://reddit.local/f/books/119",
        "http://reddit.local/f/books/120",
    ]


def test_enrich_reddit_forums_encodes_forum_paths() -> None:
    with (
        patch(
            "worldsim.phases.phase_0c_reddit_enrichment._read_forum_rows",
            return_value=[{"id": 1, "name": "personal finances", "title": "Personal"}],
        ),
        patch("worldsim.phases.phase_0c_reddit_enrichment.requests.get") as mock_get,
    ):
        mock_get.return_value = _response(200)
        enrich_reddit_forums("http://reddit.local/", "mysql://u:p@h/db")

    assert mock_get.call_args.args[0] == "http://reddit.local/f/personal%20finances"


def test_enrich_reddit_forums_requires_db_connection() -> None:
    with pytest.raises(RedditInventoryEnrichmentError, match="db_connection is required"):
        enrich_reddit_forums("http://reddit.local", None)


def test_enrich_reddit_forums_prefers_runtime_db_host_for_host_side_inventory() -> None:
    rows = [{"id": 1, "name": "books", "title": "Books"}]
    seen_connections: list[str] = []

    def fake_read_forum_rows(db_connection: str):
        seen_connections.append(db_connection)
        return rows

    with (
        patch(
            "worldsim.phases.phase_0c_reddit_enrichment._read_forum_rows",
            side_effect=fake_read_forum_rows,
        ),
        patch("worldsim.phases.phase_0c_reddit_enrichment.requests.get") as mock_get,
    ):
        mock_get.return_value = _response(200)
        result = enrich_reddit_forums(
            "http://reddit.local",
            "postgresql://postmill:postmill@3.12.221.9:5434/postmill",
            runtime_db_host="172.17.0.1",
        )

    assert seen_connections == ["postgresql://postmill:postmill@172.17.0.1:5434/postmill"]
    assert result["forums"] == [{"id": "1", "name": "books", "title": "Books"}]


def test_db_connection_candidates_fall_back_to_original_after_runtime_host() -> None:
    assert _db_connection_candidates(
        "postgresql://u:p@3.12.221.9:5434/postmill",
        "172.17.0.1",
    ) == [
        "postgresql://u:p@172.17.0.1:5434/postmill",
        "postgresql://u:p@3.12.221.9:5434/postmill",
    ]


def test_common_reddit_forum_inventory_filters_replica_local_probe_forums() -> None:
    result = common_reddit_forum_inventory(
        [
            {
                "forums": [
                    {"id": "1", "name": "ws-probe", "title": "ws-probe"},
                    {"id": "2", "name": "AskReddit", "title": "AskReddit"},
                    {"id": "3", "name": "DIY", "title": "DIY"},
                ]
            },
            {
                "forums": [
                    {"id": "9", "name": "AskReddit", "title": "AskReddit"},
                    {"id": "10", "name": "DIY", "title": "DIY"},
                ]
            },
        ]
    )

    assert result == {
        "forums": [
            {"id": "2", "name": "AskReddit", "title": "AskReddit"},
            {"id": "3", "name": "DIY", "title": "DIY"},
        ]
    }


def test_read_forum_rows_wraps_connection_failures() -> None:
    with patch(
        "worldsim.phases.phase_0c_reddit_enrichment._connect_db",
        side_effect=TimeoutError("connection timed out"),
    ):
        with pytest.raises(
            RedditInventoryEnrichmentError,
            match="failed to enumerate reddit forums: connection timed out",
        ):
            _read_forum_rows("postgresql://u:p@db.example:5432/postmill")


def test_merge_reddit_inventory_into_profile_preserves_existing_entities() -> None:
    profile = {"available_entities": {"users": [{"username": "alice"}]}}

    result = merge_reddit_inventory_into_profile(
        profile,
        {"forums": [{"id": "1", "name": "books", "title": "Books"}]},
    )

    assert result["available_entities"] == {
        "users": [{"username": "alice"}],
        "forums": [{"id": "1", "name": "books", "title": "Books"}],
    }
