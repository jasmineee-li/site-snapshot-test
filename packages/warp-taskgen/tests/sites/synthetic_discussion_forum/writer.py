"""Fake regular-participant editor with no network or credentials."""

from __future__ import annotations

from types import MappingProxyType

from tests.sites.synthetic_discussion_forum.site import ORIGIN
from warp_taskgen.editors._registry import EditorMethodSpec
from warp_taskgen.seeding.site_contracts import CreatedResourceFact, EditorSeedResult


class SyntheticDiscussionForumEditor:
    def __init__(self, *, fail_cleanup: bool = False) -> None:
        self.cleaned = False
        self.fail_cleanup = fail_cleanup

    def create_comment(self, *, thread_id: str, actor: str, body: str) -> EditorSeedResult:
        del body
        return EditorSeedResult(
            write_tokens=MappingProxyType(
                {"thread_id": thread_id, "comment_id": "42", "actor": actor}
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


editor = SyntheticDiscussionForumEditor()
failing_editor = SyntheticDiscussionForumEditor(fail_cleanup=True)

editor_spec = EditorMethodSpec(
    benchmark="webarena_verified",
    site="synthetic_discussion_forum",
    method="create_comment",
    kinds=frozenset({"thread"}),
    http=("POST", "/threads/{thread_id}/replies"),
    bindings={},
    surface_id_per_kind={"thread": "comment.body"},
    required_editor_args=(),
)
