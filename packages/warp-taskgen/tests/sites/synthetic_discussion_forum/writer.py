"""Fake regular-participant editor with no network or credentials."""

from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Any

from tests.sites.synthetic_discussion_forum.site import BENCHMARK, ORIGIN, SITE
from warp_taskgen.editors._method_spec import FreeText, Token
from warp_taskgen.editors._registry import EditorMethodSpec
from warp_taskgen.seeding.site_contracts import (
    CreatedResourceFact,
    EditorSeedResult,
    SeedSiteRegistration,
    SeedSiteRegistry,
)


class SyntheticDiscussionForumEditor:
    def __init__(self, *, fail_cleanup: bool = False) -> None:
        self.cleaned = False
        self.fail_cleanup = fail_cleanup

    def create_comment(self, *, thread_id: str, actor: str, body: str) -> EditorSeedResult:
        if actor != "participant":
            raise ValueError("synthetic writer requires a regular participant actor")
        body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return EditorSeedResult(
            write_tokens=MappingProxyType(
                {
                    "thread_id": thread_id,
                    "comment_id": "42",
                    "actor": actor,
                    "body_sha256": body_sha256,
                }
            ),
            created_resources=(
                CreatedResourceFact(
                    url=f"{ORIGIN}/threads/{thread_id}#comment-42",
                    kind="comment",
                    id="42",
                    parent_url=f"/threads/{thread_id}",
                    editor_method="create_comment",
                ),
            ),
            read_surface_urls=(f"/threads/{thread_id}",),
            read_surface_provenance_source="regular_participant_writer",
            editor_method="create_comment",
        )

    def cleanup(self) -> None:
        if self.fail_cleanup:
            raise RuntimeError("synthetic cleanup failure")
        self.cleaned = True


failing_editor = SyntheticDiscussionForumEditor(fail_cleanup=True)


def _editor_factory(
    instance: dict[str, Any],
    session: Any,
) -> SyntheticDiscussionForumEditor:
    del instance, session
    return SyntheticDiscussionForumEditor()


seed_registration = SeedSiteRegistration(BENCHMARK, SITE, _editor_factory)
seed_registry = SeedSiteRegistry.from_registrations((seed_registration,))

editor_spec = EditorMethodSpec(
    benchmark=BENCHMARK,
    site=SITE,
    method="create_comment",
    kinds=frozenset({"thread"}),
    http=("POST", "/threads/{thread_id}/replies"),
    bindings={
        "thread_id": Token("{benign_thread_id}"),
        "actor": Token("{benign_user_handle}"),
        "body": FreeText(),
    },
    surface_id_per_kind={"thread": "comment.body"},
    required_editor_args=("thread_id", "actor", "body"),
)
