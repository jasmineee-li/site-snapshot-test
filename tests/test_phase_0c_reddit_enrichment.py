from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worldsim.phases.phase_0c_reddit_enrichment import (
    RedditInventoryEnrichmentError,
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
