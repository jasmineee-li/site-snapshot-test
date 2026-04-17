from __future__ import annotations

import pytest

from worldsim.seed_resolvers import reddit
from worldsim.seed_resolvers.types import ResolverError


def test_create_submission_routes_to_forum_submit_path_in_dry_run():
    resolved = reddit.create(
        {
            "method": "POST",
            "resource_type": "submission",
            "create": {
                "forum": {"name_template": "books"},
                "submission": {"title_template": "Books thread"},
            },
        },
        {"site_url": "http://reddit.test", "seed_resolver_dry_run": True},
        {},
    )

    assert resolved.url == "http://reddit.test/submit/books"
    assert resolved.context_additions["forum_name"] == "books"


def test_create_forum_routes_to_create_forum_path():
    resolved = reddit.create(
        {
            "method": "POST",
            "resource_type": "forum",
            "create": {"forum": {"name_template": "books"}},
        },
        {"site_url": "http://reddit.test"},
        {},
    )

    assert resolved.url == "http://reddit.test/create_forum"
    assert resolved.context_additions["forum_name"] == "books"


def test_create_submission_prefers_explicit_forum_over_seed_context():
    resolved = reddit.create(
        {
            "method": "POST",
            "resource_type": "submission",
            "create": {
                "forum": {"name_template": "explicit-forum"},
                "submission": {"title_template": "Books thread"},
            },
        },
        {"site_url": "http://reddit.test", "seed_resolver_dry_run": True},
        {"forum_name": "stale-context"},
    )

    assert resolved.url == "http://reddit.test/submit/explicit-forum"
    assert resolved.context_additions["forum_name"] == "explicit-forum"


def test_create_comment_uses_preview_submission_in_dry_run():
    resolved = reddit.create(
        {
            "method": "POST",
            "resource_type": "comment",
            "create": {
                "forum": {"name_template": "books"},
                "submission": {"title_template": "Books thread"},
            },
        },
        {"site_url": "http://reddit.test", "seed_resolver_dry_run": True},
        {},
    )

    assert resolved.url == "http://reddit.test/f/books/preview-submission/-/comment"
    assert resolved.context_additions["submission_id"] == "preview-submission"


def test_create_comment_reuses_existing_submission_lookup(monkeypatch):
    monkeypatch.setattr(reddit, "_find_existing_submission_id", lambda instance, forum_name, title: "4242")
    monkeypatch.setattr(reddit, "_ensure_forum", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not create forum")))

    resolved = reddit.create(
        {
            "method": "POST",
            "resource_type": "comment",
            "create": {
                "forum": {"name_template": "books"},
                "submission": {"title_template": "Books thread"},
            },
        },
        {"site_url": "http://reddit.test"},
        {},
    )

    assert resolved.url == "http://reddit.test/f/books/4242/-/comment"
    assert resolved.context_additions["submission_id"] == "4242"


def test_update_is_unsupported():
    with pytest.raises(ResolverError, match="unsupported_resource_type"):
        reddit.update({}, {"site_url": "http://reddit.test"}, {})


def test_update_user_bio_routes_to_profile_edit():
    resolved = reddit.update(
        {"resource_type": "user_bio"},
        {
            "site_url": "http://reddit.test",
            "auth": {"credentials": {"username": "Marvels/Grant Man136"}},
        },
        {},
    )

    assert resolved.method == "POST"
    assert resolved.url == "http://reddit.test/user/Marvels%2FGrant%20Man136/edit_biography"


def test_update_user_bio_rejects_seed_context_username_mismatch():
    with pytest.raises(ResolverError, match="username_mismatch"):
        reddit.update(
            {"resource_type": "user_bio"},
            {
                "site_url": "http://reddit.test",
                "auth": {"credentials": {"username": "auth-user"}},
            },
            {"username": "other-user"},
        )
