from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, ClassVar

import pytest

from worldsim import seeding
from worldsim.editors import EDITOR_REGISTRY, GitlabEditor, RedditEditor
from worldsim.seeding.site_contracts import (
    CreatedResourceFact,
    EditorSeedResult,
    ReadSurfaceFact,
    SeedSiteRegistration,
    SeedSiteRegistry,
)


class _IsolatedSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _IsolatedEditor:
    site_name = "test_site"
    supported_methods = frozenset({"create_resource", "fail_after_create"})
    instances: ClassVar[list[_IsolatedEditor]] = []
    fail_cleanup: ClassVar[bool] = False

    def __init__(self, instance: dict[str, Any], session: _IsolatedSession) -> None:
        self.instance = instance
        self.session = session
        self.cleaned = False
        _IsolatedEditor.instances.append(self)

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        return None

    def create_resource(self, *, body: str) -> dict[str, Any]:
        return {
            "resource_id": "resource-1",
            "created_resource": {
                "role": "seed_render_surface",
                "kind": "test_resource",
                "id": 1,
                "url": "http://test.invalid/resource-1",
            },
            "read_surface_urls": ["/resource-1"],
            "read_surface_provenance_source": "editor_constructed",
        }

    def fail_after_create(self, *, body: str) -> dict[str, Any]:
        raise RuntimeError("seed failed")

    def cleanup(self) -> None:
        self.cleaned = True
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")


def _isolated_registry() -> SeedSiteRegistry:
    return SeedSiteRegistry.from_registrations(
        [SeedSiteRegistration("WebArena Verified", "test_site", _IsolatedEditor)]
    )


def test_seed_registry_is_immutable_and_normalizes_registration_keys() -> None:
    registry = _isolated_registry()

    assert registry.get("webarena_verified", "TEST_SITE") is not None
    with pytest.raises(TypeError):
        registry.registrations["bad", "site"] = registry.registrations[
            "webarena_verified", "test_site"
        ]
    with pytest.raises(FrozenInstanceError):
        registry.registrations["webarena_verified", "test_site"].site = "other"


def test_explicit_seed_registry_runs_test_site_without_global_registration(monkeypatch) -> None:
    _IsolatedEditor.instances.clear()
    monkeypatch.setattr(_IsolatedEditor, "fail_cleanup", False)
    session = _IsolatedSession()
    monkeypatch.setattr(seeding.requests, "Session", lambda: session)
    production_keys = set(EDITOR_REGISTRY)

    handle, metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "site": "test_site",
                    "method": "create_resource",
                    "args": {"body": "payload"},
                }
            ],
        },
        {"site_name": "test_site", "site_url": "http://test.invalid"},
        seed_registry=_isolated_registry(),
    )

    assert handle is not None
    assert metadata["created_resource"]["url"] == "http://test.invalid/resource-1"
    assert metadata["read_surface_urls"] == ["/resource-1"]
    assert _IsolatedEditor.instances[-1].cleaned is False
    handle.cleanup()
    assert _IsolatedEditor.instances[-1].cleaned is True
    assert session.closed is True
    assert set(EDITOR_REGISTRY) == production_keys
    assert ("webarena_verified", "test_site") not in EDITOR_REGISTRY


def test_removed_test_site_fails_closed_while_active_site_snapshot_stays_available() -> None:
    production = SeedSiteRegistry.from_editor_registry(EDITOR_REGISTRY)

    assert production.get("webarena_verified", "gitlab") is not None
    assert production.get("webarena_verified", "reddit") is not None
    assert production.get("webarena_verified", "test_site") is None

    with pytest.raises(seeding.EditorError) as raised:
        seeding.apply_data_seed(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "site": "test_site",
                        "method": "create_resource",
                        "args": {"body": "payload"},
                    }
                ],
            },
            {"site_name": "test_site", "site_url": "http://test.invalid"},
            seed_registry=production,
        )

    assert raised.value.kind == "unsupported_site"


def test_seed_registry_rejects_duplicate_normalized_registrations() -> None:
    first = SeedSiteRegistration("WebArena Verified", "TEST_SITE", _IsolatedEditor)
    second = SeedSiteRegistration("webarena_verified", "test_site", _IsolatedEditor)

    with pytest.raises(ValueError, match="duplicate seed registry registration"):
        SeedSiteRegistry.from_registrations([first, second])

    with pytest.raises(ValueError, match="duplicate seed registry registration"):
        SeedSiteRegistry.from_editor_registry(
            {
                ("WebArena Verified", "TEST_SITE"): _IsolatedEditor,
                ("webarena_verified", "test_site"): _IsolatedEditor,
            }
        )

    with pytest.raises(ValueError, match="invalid editor registry key"):
        SeedSiteRegistry.from_editor_registry({(None, "test_site"): _IsolatedEditor})


def test_cleanup_handle_is_lifo_and_idempotent() -> None:
    events: list[str] = []

    class _CleanupEditor:
        def __init__(self, name: str) -> None:
            self.name = name

        def cleanup(self) -> None:
            events.append(self.name)

    class _CleanupSession:
        def close(self) -> None:
            events.append("session")

    handle = seeding.SeedCleanupHandle(
        session=_CleanupSession(),
        editor_instances={
            ("webarena_verified", "first"): _CleanupEditor("first"),
            ("webarena_verified", "second"): _CleanupEditor("second"),
        },
    )

    handle.cleanup()
    handle.cleanup()

    assert events == ["second", "first", "session"]


def test_preflight_cleanup_attempts_every_editor_in_lifo_order(monkeypatch) -> None:
    events: list[str] = []

    class _PreflightSession:
        def close(self) -> None:
            events.append("session")

    def _editor_factory(label: str, *, fail_cleanup: bool = False):
        class _PreflightEditor:
            site_name = "test_site"
            supported_methods = frozenset({"create_resource"})

            def __init__(self, instance: dict[str, Any], session: Any) -> None:
                pass

            def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
                pass

            def preview_context(self, method_name: str, args: dict[str, Any]) -> dict[str, Any]:
                return {}

            def cleanup(self) -> None:
                events.append(label)
                if fail_cleanup:
                    raise RuntimeError(f"cleanup failed for {label}")

        return _PreflightEditor

    registry = SeedSiteRegistry.from_registrations(
        [
            SeedSiteRegistration(
                "webarena_verified",
                "test_site",
                _editor_factory("first"),
            ),
            SeedSiteRegistration(
                "stwebagentbench",
                "test_site",
                _editor_factory("second", fail_cleanup=True),
            ),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", _PreflightSession)

    with pytest.raises(RuntimeError, match="cleanup failed for second"):
        seeding.preflight_editor_seed_calls(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "test_site",
                        "method": "create_resource",
                        "args": {"body": "first"},
                    },
                    {
                        "benchmark": "stwebagentbench",
                        "site": "test_site",
                        "method": "create_resource",
                        "args": {"body": "second"},
                    },
                ],
            },
            {"site_name": "test_site", "site_url": "http://test.invalid"},
            seed_registry=registry,
        )

    assert events == ["second", "first", "session"]


def test_partial_seed_failure_cleans_all_and_preserves_primary_error(monkeypatch) -> None:
    _IsolatedEditor.instances.clear()
    monkeypatch.setattr(_IsolatedEditor, "fail_cleanup", True)
    session = _IsolatedSession()
    monkeypatch.setattr(seeding.requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="seed failed") as raised:
        seeding.apply_data_seed(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "site": "test_site",
                        "method": "create_resource",
                        "args": {"body": "payload"},
                    },
                    {
                        "site": "test_site",
                        "method": "fail_after_create",
                        "args": {"body": "payload"},
                    },
                ],
            },
            {"site_name": "test_site", "site_url": "http://test.invalid"},
            seed_registry=_isolated_registry(),
        )

    assert str(raised.value) == "seed failed"
    assert "cleanup also failed" in "\n".join(raised.value.__notes__)
    assert _IsolatedEditor.instances[-1].cleaned is True
    assert session.closed is True


def test_typed_editor_result_normalizes_legacy_resource_and_surface_fields() -> None:
    result = EditorSeedResult.from_mapping(
        {
            "submission_id": 7,
            "created_resource": {
                "kind": "submission",
                "id": 7,
                "url": " http://test.invalid/submission/7 ",
            },
            "read_surface_urls": [" /submission/7 ", ""],
            "read_surface_provenance_source": "editor_api_response",
        },
        editor_method="test_site.create_resource",
    )

    assert result.write_tokens["submission_id"] == 7
    assert result.read_surface_urls == ("/submission/7",)
    assert result.read_surfaces == (
        ReadSurfaceFact(
            url="/submission/7",
            provenance_source="editor_api_response",
            editor_method="test_site.create_resource",
        ),
    )
    assert result.created_resources == (
        CreatedResourceFact(
            role="created_resource",
            kind="submission",
            id="7",
            url="http://test.invalid/submission/7",
            editor_method="test_site.create_resource",
        ),
    )
    assert result.created_resources[0].as_mapping()["editor_method"] == "test_site.create_resource"


def test_legacy_registry_adapter_preserves_active_editor_classes() -> None:
    registry = SeedSiteRegistry.from_editor_registry(EDITOR_REGISTRY)

    assert registry.get("webarena_verified", "gitlab").editor_factory is GitlabEditor
    assert registry.get("webarena_verified", "reddit").editor_factory is RedditEditor


def test_patchable_seeding_facade_exposes_seed_contract_types() -> None:
    assert seeding.SeedSiteRegistry is SeedSiteRegistry
    assert seeding.SeedSiteRegistration is SeedSiteRegistration
    assert seeding.EditorSeedResult is EditorSeedResult
    assert seeding.CreatedResourceFact is CreatedResourceFact
    assert seeding.ReadSurfaceFact is ReadSurfaceFact


def test_default_path_preserves_legacy_apply_hook_signature(monkeypatch) -> None:
    calls: list[tuple[int | None, str]] = []
    session = _IsolatedSession()

    def _legacy_apply_hook(
        session: Any,
        call: dict[str, Any],
        instance: dict[str, Any],
        *,
        call_index: int | None = None,
        seed_context: dict[str, Any],
        editor_instances: dict[tuple[str, str], Any],
        read_surface_accumulator: list[str] | None = None,
        read_surface_provenance: dict[str, Any] | None = None,
        created_resource_accumulator: list[dict[str, Any]] | None = None,
        editor_call_result_accumulator: list[dict[str, Any]] | None = None,
    ) -> None:
        calls.append((call_index, str(call["method"])))

    monkeypatch.setattr(seeding.requests, "Session", lambda: session)
    monkeypatch.setattr(seeding, "_apply_editor_seed_call", _legacy_apply_hook)

    handle, metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                }
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert handle is None
    assert metadata == {}
    assert calls == [(0, "create_submission")]
    assert session.closed is True


def test_default_path_snapshots_legacy_registry_once_per_seed(monkeypatch) -> None:
    class _MutatingEditor(_IsolatedEditor):
        site_name = "test_site"

        def create_resource(self, *, body: str) -> dict[str, Any]:
            seeding.EDITOR_REGISTRY[("stwebagentbench", "test_site")] = _IsolatedEditor
            return {"resource_id": "first"}

    monkeypatch.setitem(
        seeding.EDITOR_REGISTRY,
        ("webarena_verified", "test_site"),
        _MutatingEditor,
    )
    monkeypatch.delitem(
        seeding.EDITOR_REGISTRY,
        ("stwebagentbench", "test_site"),
        raising=False,
    )
    monkeypatch.setattr(seeding.requests, "Session", _IsolatedSession)

    try:
        with pytest.raises(seeding.EditorError) as raised:
            seeding.apply_data_seed(
                {
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "test_site",
                            "method": "create_resource",
                            "args": {"body": "first"},
                        },
                        {
                            "benchmark": "stwebagentbench",
                            "site": "test_site",
                            "method": "create_resource",
                            "args": {"body": "second"},
                        },
                    ],
                },
                {"site_name": "test_site", "site_url": "http://test.invalid"},
            )
    finally:
        seeding.EDITOR_REGISTRY.pop(("stwebagentbench", "test_site"), None)

    assert raised.value.kind == "unsupported_site"
