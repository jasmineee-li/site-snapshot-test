"""Immutable test inputs for the discussion forum Site Behavior Contract."""

from __future__ import annotations

from tests.sites.behavior_contract.exact_resource_evidence import ExactResourceEvidenceCase

evidence_case = ExactResourceEvidenceCase(
    benchmark="webarena_verified",
    site="synthetic_discussion_forum",
    origin="https://forum.test",
    parent_kind="thread",
    parent_id="17",
    parent_path="/threads/17",
    resource_kind="comment",
    resource_id="42",
    actor="participant",
    body="A regular participant reply. TRACE-SIGNATURE",
    signature="TRACE-SIGNATURE",
    action_kind="submit_comment",
    action_path="/threads/17/replies",
)
