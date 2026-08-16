"""Assertions for the Regular Participant Writer owner seam."""

from __future__ import annotations

from typing import Any

import pytest

from warp_taskgen.editors._registry import EditorMethodSpec
from warp_taskgen.seeding.site_contracts import EditorSeedResult


def assert_regular_participant_writer_behavior(
    editor: Any,
    *,
    thread_id: str,
    actor: str,
    body: str,
    expected_comment_id: str,
    expected_resource_kind: str,
    expected_parent_path: str,
    expected_read_surface_provenance: str,
    expected_editor_method: str,
    editor_spec: EditorMethodSpec,
    expected_surface_id: str,
    cleanup_failure_editor: Any | None = None,
) -> None:
    """Check ordinary-user creation returns exact, secret-free write evidence."""

    assert editor_spec.method == expected_editor_method
    assert editor_spec.kinds == frozenset({"thread"})
    assert editor_spec.surface_id_per_kind == {"thread": expected_surface_id}
    assert editor_spec.http == ("POST", "/threads/{thread_id}/replies")

    result = editor.create_comment(thread_id=thread_id, actor=actor, body=body)
    assert isinstance(result, EditorSeedResult)
    assert dict(result.write_tokens) == {
        "actor": actor,
        "comment_id": expected_comment_id,
        "thread_id": thread_id,
    }
    assert result.editor_method == expected_editor_method
    assert len(result.created_resources) == 1
    created = result.created_resources[0]
    assert created.kind == expected_resource_kind
    assert created.id == expected_comment_id
    assert created.parent_url == expected_parent_path
    assert result.read_surface_urls == (expected_parent_path,)
    assert result.read_surface_provenance_source == expected_read_surface_provenance
    assert actor != "admin"

    editor.cleanup()
    editor.cleanup()
    assert editor.cleaned is True

    if cleanup_failure_editor is not None:
        with pytest.raises(RuntimeError, match="cleanup"):
            cleanup_failure_editor.cleanup()
