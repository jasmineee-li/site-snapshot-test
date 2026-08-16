"""Fake action-card owner for the test-only discussion forum Site."""

from __future__ import annotations

from tests.sites.synthetic_discussion_forum.site import SITE
from warp_taskgen.adversarial_actions.capability_adapters import CapabilityTaskAdapter


class SyntheticDiscussionForumActionCard(CapabilityTaskAdapter):
    carrier_surface = "comment.body"

    def validate(self) -> None:
        if self.benchmark_family != "webarena_verified":
            raise ValueError("action card benchmark mismatch")
        if self.site != SITE:
            raise ValueError("action card Site mismatch")
        if self.action_kind != "submit_comment":
            raise ValueError("action card action mismatch")
        if not self.route_ids or len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("action card routes must be unique")
        if self.route_ids != (f"{SITE}.comment_body.thread.create_comment",):
            raise ValueError("action card route mismatch")


action_card = SyntheticDiscussionForumActionCard(
    id="synthetic_discussion_forum_public_reply",
    benchmark_family="webarena_verified",
    site=SITE,
    action_kind="submit_comment",
    route_ids=(f"{SITE}.comment_body.thread.create_comment",),
    archetype_id="synthetic_discussion_forum_public_reply",
    benign_task_family_id="submission_discussion_followup",
)
