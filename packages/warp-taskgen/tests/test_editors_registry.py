"""Unit tests for the editor-method contract registry primitives.

Commit 1 (R1) — covers the decorator, the registry dataclasses, and
:func:`register_editor`. No production editor classes are touched here;
these tests use minimal in-test fixture classes so the registry stays
empty in production until commit 2 wires up the real editors.
"""

from __future__ import annotations

import json

import pytest

from warp_taskgen.editors import _registry
from warp_taskgen.editors._method_spec import (
    FreeText,
    SelectorGroup,
    Token,
    editor_method,
)
from warp_taskgen.editors._registry import (
    EditorMethodSpec,
    KindContract,
    RegistryError,
    attach_surfaces_for_kind,
    available_tokens_for_kind,
    iter_specs,
    kind_contract,
    method_spec,
    register_editor,
    serialize_registry,
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a fresh empty ``_REGISTRY`` and cleared caches."""
    monkeypatch.setattr(_registry, "_REGISTRY", {})
    _registry._clear_caches()
    yield
    _registry._clear_caches()


def _make_fixture_editor(site: str, supported: tuple[str, ...]):
    class _FixtureEditor:
        site_name = site
        supported_methods = frozenset(supported)

    return _FixtureEditor


class TestBindingFactories:
    def test_token_defaults(self) -> None:
        binding = Token("{benign_issue_iid}")
        assert binding.kind == "token"
        assert binding.tokens == frozenset({"{benign_issue_iid}"})
        assert binding.selector_group is None
        assert binding.required is True

    def test_token_multiple(self) -> None:
        binding = Token("{a}", "{b}")
        assert binding.tokens == frozenset({"{a}", "{b}"})

    def test_selector_group_defaults(self) -> None:
        binding = SelectorGroup("project", "{benign_project_id}", "{benign_project_path}")
        assert binding.kind == "selector"
        assert binding.selector_group == "project"
        assert binding.tokens == frozenset({"{benign_project_id}", "{benign_project_path}"})
        assert binding.required is True

    def test_selector_group_empty_tokens(self) -> None:
        binding = SelectorGroup("project")
        assert binding.kind == "selector"
        assert binding.tokens == frozenset()
        assert binding.selector_group == "project"

    def test_free_text_defaults(self) -> None:
        binding = FreeText()
        assert binding.kind == "free_text"
        assert binding.tokens == frozenset()
        assert binding.selector_group is None
        assert binding.required is True

    def test_binding_spec_is_hashable(self) -> None:
        # frozen=True should give hashability; needed for frozenset usage
        binding = Token("{x}")
        assert hash(binding) == hash(Token("{x}"))


class TestEditorMethodDecorator:
    def test_preserves_function_identity(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/issues/{id}/notes"),
            bindings={"body": FreeText()},
        )
        def original_fn(self, *, body: str) -> str:
            """The original docstring."""
            return body

        assert original_fn.__name__ == "original_fn"
        assert original_fn.__doc__ == "The original docstring."
        assert hasattr(original_fn, "_editor_method_spec")

    def test_attaches_spec_metadata(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/issues/{id}/notes"),
            bindings={
                "issue_iid": Token("{benign_issue_iid}"),
                "body": FreeText(),
            },
            surface_id_per_kind={"gitlab_issue": "note_on_issue"},
        )
        def fn(self, *, issue_iid: str, body: str) -> None:
            pass

        spec = fn._editor_method_spec
        assert spec["kinds"] == frozenset({"gitlab_issue"})
        assert spec["http"] == ("POST", "/issues/{id}/notes")
        assert set(spec["bindings"].keys()) == {"issue_iid", "body"}
        assert spec["surface_id_per_kind"] == {"gitlab_issue": "note_on_issue"}

    def test_required_editor_args_derived_from_bindings(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/x"),
            bindings={
                "project_id": Token("{benign_project_id}", required=True),
                "issue_iid": Token("{benign_issue_iid}", required=True),
                "body": FreeText(required=True),
                "opt_field": FreeText(required=False),
            },
        )
        def fn(self, **kwargs) -> None:
            pass

        # Derived tuple preserves declaration order for required entries only
        assert fn._editor_method_spec["required_editor_args"] == (
            "project_id",
            "issue_iid",
            "body",
        )

    def test_required_editor_args_explicit_override(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/x"),
            bindings={
                "note_body": FreeText(),
                "issue_iid": Token("{benign_issue_iid}"),
                "project_id": SelectorGroup("project", "{benign_project_id}"),
            },
            # Resolver's legacy naming (LLM-facing "body", ordered)
            required_editor_args=("project_id", "issue_iid", "body"),
        )
        def fn(self, **kwargs) -> None:
            pass

        assert fn._editor_method_spec["required_editor_args"] == (
            "project_id",
            "issue_iid",
            "body",
        )

    def test_decorator_is_callable_passthrough(self) -> None:
        @editor_method(
            kinds=frozenset(),
            http=("POST", "/x"),
            bindings={},
        )
        def fn() -> int:
            return 42

        assert fn() == 42


class TestRegisterEditor:
    def test_registers_single_decorated_method(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/issues/{id}/notes"),
            bindings={"body": FreeText()},
            surface_id_per_kind={"gitlab_issue": "note_on_issue"},
        )
        def create_issue_note(self) -> None:
            pass

        cls = type(
            "_E",
            (),
            {
                "site_name": "gitlab",
                "supported_methods": frozenset({"create_issue_note"}),
                "create_issue_note": create_issue_note,
            },
        )
        register_editor(cls, "gitlab")

        spec = method_spec("gitlab", "create_issue_note")
        assert isinstance(spec, EditorMethodSpec)
        assert spec.site == "gitlab"
        assert spec.method == "create_issue_note"
        assert spec.kinds == frozenset({"gitlab_issue"})

    def test_missing_decorator_raises(self) -> None:
        def plain_method(self) -> None:
            pass

        cls = type(
            "_E",
            (),
            {
                "site_name": "gitlab",
                "supported_methods": frozenset({"plain_method"}),
                "plain_method": plain_method,
            },
        )
        with pytest.raises(RegistryError, match="has no @editor_method"):
            register_editor(cls, "gitlab")

    def test_empty_supported_methods_raises(self) -> None:
        cls = type(
            "_E",
            (),
            {"site_name": "gitlab", "supported_methods": frozenset()},
        )
        with pytest.raises(RegistryError, match="supported_methods"):
            register_editor(cls, "gitlab")

    def test_duplicate_registration_raises(self) -> None:
        @editor_method(
            kinds=frozenset({"x"}),
            http=("POST", "/x"),
            bindings={},
        )
        def m(self) -> None:
            pass

        cls = type(
            "_E",
            (),
            {"site_name": "gitlab", "supported_methods": frozenset({"m"}), "m": m},
        )
        register_editor(cls, "gitlab")
        with pytest.raises(RegistryError, match="duplicate registration"):
            register_editor(cls, "gitlab")

    def test_method_not_on_class_raises(self) -> None:
        cls = type(
            "_E",
            (),
            {"site_name": "x", "supported_methods": frozenset({"missing"})},
        )
        with pytest.raises(RegistryError, match="has no @editor_method"):
            register_editor(cls, "x")


class TestKindContract:
    def _register_two_editors(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/issues/{id}/notes"),
            bindings={
                "project_id": SelectorGroup("project", "{benign_project_id}"),
                "project_path": SelectorGroup("project", "{benign_project_path}"),
                "issue_iid": Token("{benign_issue_iid}"),
                "body": FreeText(),
            },
            surface_id_per_kind={"gitlab_issue": "note_on_issue"},
            required_editor_args=("project_id", "issue_iid", "body"),
        )
        def create_issue_note(self) -> None:
            pass

        @editor_method(
            kinds=frozenset(),  # dangling — never valid in Option A
            http=("POST", "/projects"),
            bindings={"name": FreeText()},
        )
        def create_project(self) -> None:
            pass

        cls = type(
            "_G",
            (),
            {
                "site_name": "gitlab",
                "supported_methods": frozenset({"create_issue_note", "create_project"}),
                "create_issue_note": create_issue_note,
                "create_project": create_project,
            },
        )
        register_editor(cls, "gitlab")

    def test_valid_methods_matches_decorated_kinds(self) -> None:
        self._register_two_editors()
        contract = kind_contract("gitlab_issue")
        assert isinstance(contract, KindContract)
        assert contract.valid_methods == frozenset({"create_issue_note"})
        assert "create_project" not in contract.valid_methods

    def test_dangling_method_appears_in_no_contract(self) -> None:
        """Empty-kinds decorator subsumes _OPTION_A_DANGLING_METHODS."""
        self._register_two_editors()
        # `create_project` is in the registry but not in any kind's valid_methods
        for kind in ("gitlab_issue", "gitlab_mr", "nonexistent_kind"):
            assert "create_project" not in kind_contract(kind).valid_methods

    def test_available_tokens_includes_identity(self) -> None:
        self._register_two_editors()
        contract = kind_contract("gitlab_issue")
        assert "{benign_user_handle}" in contract.available_tokens

    def test_available_tokens_includes_declared(self) -> None:
        self._register_two_editors()
        contract = kind_contract("gitlab_issue")
        assert "{benign_project_id}" in contract.available_tokens
        assert "{benign_project_path}" in contract.available_tokens
        assert "{benign_issue_iid}" in contract.available_tokens

    def test_unknown_kind_yields_empty_contract(self) -> None:
        self._register_two_editors()
        contract = kind_contract("never_registered")
        assert contract.valid_methods == frozenset()
        # Identity token always present — it's not anchor-derived
        assert contract.available_tokens == frozenset({"{benign_user_handle}"})


class TestAvailableTokensForKind:
    def _register(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/x"),
            bindings={
                "project_id": SelectorGroup("project", "{benign_project_id}"),
                "project_path": SelectorGroup("project", "{benign_project_path}"),
                "issue_iid": Token("{benign_issue_iid}"),
            },
        )
        def m(self) -> None:
            pass

        cls = type(
            "_E",
            (),
            {"site_name": "gitlab", "supported_methods": frozenset({"m"}), "m": m},
        )
        register_editor(cls, "gitlab")

    def test_intersects_with_anchor_keys(self) -> None:
        self._register()
        # Only project_path + issue_iid anchors present — project_id unreachable
        anchors = {"project_path": "foo/bar", "issue_iid": "42"}
        tokens = available_tokens_for_kind("gitlab_issue", anchors)
        assert tokens == frozenset(
            {"{benign_project_path}", "{benign_issue_iid}", "{benign_user_handle}"}
        )
        assert "{benign_project_id}" not in tokens

    def test_identity_token_present_without_any_anchors(self) -> None:
        self._register()
        assert available_tokens_for_kind("gitlab_issue", {}) == frozenset({"{benign_user_handle}"})

    def test_declared_token_not_reachable_via_anchors_is_excluded(self) -> None:
        self._register()
        # Declared tokens include project_id + project_path + issue_iid.
        # Anchors give only project_id → project_path and issue_iid excluded.
        tokens = available_tokens_for_kind("gitlab_issue", {"project_id": "123"})
        assert tokens == frozenset({"{benign_project_id}", "{benign_user_handle}"})


class TestAttachSurfacesForKind:
    def test_matches_legacy_attach_surfaces_shape(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/x"),
            bindings={
                "project_id": SelectorGroup("project", "{benign_project_id}"),
                "issue_iid": Token("{benign_issue_iid}"),
                "body": FreeText(),
            },
            surface_id_per_kind={"gitlab_issue": "note_on_issue"},
            required_editor_args=("project_id", "issue_iid", "body"),
        )
        def create_issue_note(self) -> None:
            pass

        cls = type(
            "_E",
            (),
            {
                "site_name": "gitlab",
                "supported_methods": frozenset({"create_issue_note"}),
                "create_issue_note": create_issue_note,
            },
        )
        register_editor(cls, "gitlab")

        surfaces = attach_surfaces_for_kind("gitlab_issue")
        assert surfaces == (
            {
                "surface_id": "note_on_issue",
                "attach_method": "create_issue_note",
                "required_editor_args": ["project_id", "issue_iid", "body"],
            },
        )

    def test_empty_for_unregistered_kind(self) -> None:
        assert attach_surfaces_for_kind("no_such_kind") == ()


class TestIterSpecs:
    def _register_two_sites(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/x"),
            bindings={},
        )
        def m_g(self) -> None:
            pass

        @editor_method(
            kinds=frozenset({"reddit_submission"}),
            http=("POST", "/y"),
            bindings={},
        )
        def m_r(self) -> None:
            pass

        cls_g = type("_G", (), {"supported_methods": frozenset({"m_g"}), "m_g": m_g})
        cls_r = type("_R", (), {"supported_methods": frozenset({"m_r"}), "m_r": m_r})
        register_editor(cls_g, "gitlab")
        register_editor(cls_r, "reddit")

    def test_filter_by_site(self) -> None:
        self._register_two_sites()
        sites = {s.site for s in iter_specs(site="gitlab")}
        assert sites == {"gitlab"}

    def test_filter_by_kinds(self) -> None:
        self._register_two_sites()
        matched = list(iter_specs(kinds=frozenset({"reddit_submission"})))
        assert [s.method for s in matched] == ["m_r"]

    def test_no_filter_yields_all(self) -> None:
        self._register_two_sites()
        assert len(list(iter_specs())) == 2


class TestSerializeRegistry:
    def test_round_trip_through_json(self) -> None:
        @editor_method(
            kinds=frozenset({"gitlab_issue"}),
            http=("POST", "/issues/{id}/notes"),
            bindings={
                "project_id": SelectorGroup("project", "{benign_project_id}"),
                "issue_iid": Token("{benign_issue_iid}"),
                "body": FreeText(),
            },
            surface_id_per_kind={"gitlab_issue": "note_on_issue"},
            required_editor_args=("project_id", "issue_iid", "body"),
        )
        def create_issue_note(self) -> None:
            pass

        cls = type(
            "_E",
            (),
            {
                "site_name": "gitlab",
                "supported_methods": frozenset({"create_issue_note"}),
                "create_issue_note": create_issue_note,
            },
        )
        register_editor(cls, "gitlab")

        serialized = serialize_registry()
        assert serialized["version"] == 1
        assert len(serialized["specs"]) == 1
        # Full JSON round-trip survives
        rehydrated = json.loads(json.dumps(serialized))
        assert rehydrated == serialized

    def test_expected_schema_fields(self) -> None:
        @editor_method(
            kinds=frozenset({"reddit_submission"}),
            http=("POST", "/f/{forum}/{id}/comment"),
            bindings={
                "forum_name": Token("{benign_forum_name}"),
                "submission_id": Token("{benign_submission_id}"),
                "body": FreeText(),
            },
            surface_id_per_kind={"reddit_submission": "comment_body_thread"},
            required_editor_args=("submission_id", "body"),
        )
        def create_comment(self) -> None:
            pass

        cls = type(
            "_E",
            (),
            {
                "site_name": "reddit",
                "supported_methods": frozenset({"create_comment"}),
                "create_comment": create_comment,
            },
        )
        register_editor(cls, "reddit")

        spec_json = serialize_registry()["specs"][0]
        assert set(spec_json.keys()) == {
            "benchmark",
            "site",
            "method",
            "kinds",
            "http_verb",
            "http_path",
            "bindings",
            "surface_id_per_kind",
            "required_editor_args",
        }
        assert spec_json["http_verb"] == "POST"
        assert spec_json["http_path"] == "/f/{forum}/{id}/comment"
        # Bindings serialization preserves kind + tokens + selector_group
        binding = spec_json["bindings"]["submission_id"]
        assert binding["kind"] == "token"
        assert binding["tokens"] == ["{benign_submission_id}"]
        assert binding["selector_group"] is None
        assert binding["required"] is True

    def test_deterministic_ordering(self) -> None:
        # Two editors, three methods; specs sorted by (site, method)
        @editor_method(kinds=frozenset(), http=("POST", "/x"), bindings={})
        def a(self) -> None:
            pass

        @editor_method(kinds=frozenset(), http=("POST", "/x"), bindings={})
        def b(self) -> None:
            pass

        @editor_method(kinds=frozenset(), http=("POST", "/x"), bindings={})
        def c(self) -> None:
            pass

        cls_g = type(
            "_G",
            (),
            {"supported_methods": frozenset({"a", "b"}), "a": a, "b": b},
        )
        cls_r = type(
            "_R",
            (),
            {"supported_methods": frozenset({"c"}), "c": c},
        )
        register_editor(cls_g, "gitlab")
        register_editor(cls_r, "reddit")

        specs = serialize_registry()["specs"]
        keys = [(s["site"], s["method"]) for s in specs]
        assert keys == sorted(keys)

    def test_benchmark_scoping_keeps_same_site_kind_isolated(self) -> None:
        @editor_method(
            kinds=frozenset({"shared_kind"}),
            http=("POST", "/a"),
            bindings={"body": FreeText()},
        )
        def m_a(self) -> None:
            pass

        @editor_method(
            kinds=frozenset({"shared_kind"}),
            http=("POST", "/b"),
            bindings={"other": Token("{benign_other}")},
        )
        def m_b(self) -> None:
            pass

        cls_a = type("_A", (), {"supported_methods": frozenset({"m_a"}), "m_a": m_a})
        cls_b = type("_B", (), {"supported_methods": frozenset({"m_b"}), "m_b": m_b})
        register_editor(cls_a, "gitlab", benchmark="benchmark_a")
        register_editor(cls_b, "gitlab", benchmark="benchmark_b")

        assert kind_contract("shared_kind", benchmark="benchmark_a").valid_methods == frozenset(
            {"m_a"}
        )
        assert kind_contract("shared_kind", benchmark="benchmark_b").valid_methods == frozenset(
            {"m_b"}
        )
        assert method_spec("gitlab", "m_b", benchmark="benchmark_b").benchmark == "benchmark_b"


class TestCacheInvalidation:
    def test_clear_caches_picks_up_new_registrations(self) -> None:
        @editor_method(kinds=frozenset({"x"}), http=("POST", "/x"), bindings={})
        def m(self) -> None:
            pass

        cls = type("_E", (), {"supported_methods": frozenset({"m"}), "m": m})
        assert kind_contract("x").valid_methods == frozenset()

        register_editor(cls, "gitlab")
        # register_editor calls _clear_caches internally
        assert kind_contract("x").valid_methods == frozenset({"m"})
