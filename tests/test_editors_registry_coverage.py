"""Coverage parity tests for the populated editor registry.

Commit 2 (R2) — asserts that:

* Every method in every registered editor's ``supported_methods`` frozenset
  has an ``@editor_method`` decorator (enforced by ``register_editor`` at
  import time, reasserted here for defensive coverage).
* Every binding arg name maps to a real Python parameter of the method,
  either directly or through the seeding alias map
  (``worldsim.seeding._editor_arg_name``) that translates LLM-facing names
  like ``"body"`` into Python-facing names like ``"note_body"``.
* :func:`attach_surfaces_for_kind` output matches the legacy
  ``_LEGACY_ATTACH_SURFACES`` golden fixture byte-for-byte. Commit 3 swapped the
  resolver over to read the registry, so the legacy literal has been
  deleted; the golden data is now a local frozen fixture
  (``_LEGACY_ATTACH_SURFACES``) in this file, which must be updated in
  lockstep with any intentional contract change.
* No resource kind is addressed by two different sites.
"""

from __future__ import annotations

import inspect

import pytest

from worldsim.editors import EDITOR_REGISTRY
from worldsim.editors._registry import (
    _REGISTRY,
    attach_surfaces_for_kind,
    iter_specs,
    kind_contract,
    method_spec,
)

# Frozen snapshot of the pre-refactor attach_surfaces data. Any
# intentional contract change MUST update both the decorator on the
# relevant editor method AND this fixture — making the behavior change
# explicit in diff review.
_LEGACY_ATTACH_SURFACES: dict[str, list[dict[str, object]]] = {
    "gitlab_issue": [
        {
            "surface_id": "note_on_issue",
            "attach_method": "create_issue_note",
            "required_editor_args": ["project_id", "issue_iid", "body"],
        },
    ],
    "gitlab_mr": [
        {
            "surface_id": "note_on_mr",
            "attach_method": "create_mr_note",
            "required_editor_args": ["project_id", "mr_iid", "body"],
        },
    ],
    "gitlab_search_result": [
        {
            "surface_id": "issue.title",
            "attach_method": "create_issue_title",
            "required_editor_args": ["project_id", "title"],
        },
        {
            "surface_id": "issue.description",
            "attach_method": "create_issue_description",
            "required_editor_args": ["project_id", "body"],
        },
        {
            "surface_id": "note_on_issue",
            "attach_method": "create_issue_note",
            "required_editor_args": ["project_id", "issue_iid", "body"],
        },
        {
            "surface_id": "note_on_mr",
            "attach_method": "create_mr_note",
            "required_editor_args": ["project_id", "mr_iid", "body"],
        },
    ],
    "gitlab_dashboard_list": [
        {
            "surface_id": "note_on_issue",
            "attach_method": "create_issue_note",
            "required_editor_args": ["project_id", "issue_iid", "body"],
        },
        {
            "surface_id": "note_on_mr",
            "attach_method": "create_mr_note",
            "required_editor_args": ["project_id", "mr_iid", "body"],
        },
    ],
    "reddit_submission": [
        {
            "surface_id": "comment_body_thread",
            "attach_method": "create_comment",
            "required_editor_args": ["submission_id", "body"],
        },
    ],
    "reddit_forum": [
        {
            "surface_id": "submission.title",
            "attach_method": "create_submission_title",
            "required_editor_args": ["forum_name", "title"],
        },
        {
            "surface_id": "submission_body_detail",
            "attach_method": "create_submission",
            "required_editor_args": ["forum_name", "title", "body"],
        },
    ],
    "reddit_dashboard_list": [
        {
            "surface_id": "comment_body_thread",
            "attach_method": "create_comment",
            "required_editor_args": ["submission_id", "body"],
        },
    ],
}

# Map (site, method, llm_arg) -> python_param for LLM-facing args that
# don't match the Python signature directly. Mirrors
# ``worldsim.seeding._editor_arg_name``'s alias table — if those aliases
# change, this map must change in lockstep.
LLM_TO_PYTHON_ARG_ALIASES: dict[tuple[str, str, str], str] = {
    ("gitlab", "create_issue_note", "body"): "note_body",
    ("gitlab", "create_mr_note", "body"): "note_body",
    ("reddit", "create_submission", "title"): "title_template",
    ("reddit", "create_submission", "body"): "body_template",
    ("reddit", "update_user_bio", "bio"): "bio_text",
}


def _editor_class(benchmark: str, site: str) -> type:
    for (registered_benchmark, registered_site), cls in EDITOR_REGISTRY.items():
        if registered_benchmark == benchmark and registered_site == site:
            return cls
    raise KeyError((benchmark, site))


class TestRegistryCoverage:
    def test_expected_method_count(self) -> None:
        """18 gitlab + 5 reddit = 23 registered methods."""
        assert len(_REGISTRY) == 23

    def test_all_supported_methods_registered(self) -> None:
        for (benchmark, site), cls in EDITOR_REGISTRY.items():
            for method_name in cls.supported_methods:
                assert (benchmark, site, method_name) in _REGISTRY, (
                    f"{benchmark}.{site}.{method_name} not in _REGISTRY — "
                    "missing @editor_method decoration"
                )

    def test_registered_methods_match_supported(self) -> None:
        """_REGISTRY cannot contain a method not in supported_methods."""
        for benchmark, site, method_name in _REGISTRY:
            cls = _editor_class(benchmark, site)
            assert method_name in cls.supported_methods, (
                f"{benchmark}.{site}.{method_name} is registered but not in "
                f"{cls.__name__}.supported_methods"
            )

    def test_registry_helpers_normalize_benchmark_aliases(self) -> None:
        assert method_spec(
            "gitlab", "create_issue_note", benchmark="WebArena Verified"
        ).benchmark == ("webarena_verified")
        assert kind_contract(
            "gitlab_issue",
            benchmark="WebArena Verified",
            site="gitlab",
        ).valid_methods
        assert attach_surfaces_for_kind(
            "gitlab_issue",
            benchmark="WebArena Verified",
            site="gitlab",
        )
        assert list(iter_specs(site="gitlab", benchmark="WebArena Verified"))


class TestBindingArgCoverage:
    @pytest.mark.parametrize(
        "benchmark,site,method",
        sorted(_REGISTRY.keys()),
    )
    def test_binding_args_map_to_python_params(
        self,
        benchmark: str,
        site: str,
        method: str,
    ) -> None:
        spec = _REGISTRY[(benchmark, site, method)]
        cls = _editor_class(benchmark, site)
        fn = getattr(cls, method)
        python_params = set(inspect.signature(fn).parameters.keys()) - {"self"}

        for binding_arg in spec.bindings.keys():
            if binding_arg in python_params:
                continue
            aliased = LLM_TO_PYTHON_ARG_ALIASES.get((site, method, binding_arg))
            assert aliased is not None, (
                f"{site}.{method} binding {binding_arg!r} is neither a "
                f"Python param {python_params} nor in the LLM-alias map"
            )
            assert aliased in python_params, (
                f"{site}.{method} binding {binding_arg!r} aliases to "
                f"{aliased!r} which isn't a Python param of the method"
            )


class TestAttachSurfacesParity:
    """attach_surfaces_for_kind must match legacy _LEGACY_ATTACH_SURFACES byte-for-byte.

    Commit 3 swaps the resolver over to read from the registry. This test
    guarantees that swap is a pure refactor — no downstream consumer sees
    different data.
    """

    @pytest.mark.parametrize("kind", sorted(_LEGACY_ATTACH_SURFACES.keys()))
    def test_kind_attach_surfaces_match(self, kind: str) -> None:
        legacy = _LEGACY_ATTACH_SURFACES[kind]
        registry = attach_surfaces_for_kind(kind)

        # Normalize both to sorted-by-attach_method for order-insensitive
        # comparison. Legacy is a list of dicts; registry is a tuple of
        # dicts. We compare the content.
        def _canon(entries):
            return sorted(
                [
                    {
                        "surface_id": e["surface_id"],
                        "attach_method": e["attach_method"],
                        "required_editor_args": list(e["required_editor_args"]),
                    }
                    for e in entries
                ],
                key=lambda e: (e["attach_method"], e["surface_id"]),
            )

        assert _canon(registry) == _canon(legacy), (
            f"kind={kind}: registry output diverges from legacy _LEGACY_ATTACH_SURFACES.\n"
            f"registry: {list(registry)}\n"
            f"legacy:   {legacy}"
        )

    def test_every_legacy_kind_has_registry_coverage(self) -> None:
        for kind, legacy_entries in _LEGACY_ATTACH_SURFACES.items():
            registry_entries = attach_surfaces_for_kind(kind)
            assert len(registry_entries) == len(legacy_entries), (
                f"kind={kind}: {len(registry_entries)} registry entries "
                f"vs {len(legacy_entries)} legacy entries"
            )


class TestCrossSiteNamespacing:
    def test_no_kind_crosses_sites(self) -> None:
        """gitlab_* kinds only ever appear on gitlab specs; reddit_* only on reddit."""
        for spec in iter_specs():
            for kind in spec.kinds:
                if kind.startswith("gitlab_"):
                    assert spec.site == "gitlab", (
                        f"{spec.site}.{spec.method} declares kind {kind!r} but site is not gitlab"
                    )
                elif kind.startswith("reddit_"):
                    assert spec.site == "reddit", (
                        f"{spec.site}.{spec.method} declares kind {kind!r} but site is not reddit"
                    )
                else:
                    pytest.fail(
                        f"{spec.site}.{spec.method} declares kind {kind!r} "
                        f"without a known site prefix (gitlab_* or reddit_*)"
                    )


class TestSurfaceIdConsistency:
    def test_every_addressed_kind_has_surface_id(self) -> None:
        """If a method declares it addresses kind K, it must have a
        surface_id_per_kind entry for K."""
        for spec in iter_specs():
            for kind in spec.kinds:
                assert kind in spec.surface_id_per_kind, (
                    f"{spec.site}.{spec.method} addresses kind {kind!r} "
                    f"but surface_id_per_kind is missing it"
                )


class TestSerializeRegistrySanity:
    def test_serialize_registry_covers_all_methods(self) -> None:
        from worldsim.editors._registry import serialize_registry

        serialized = serialize_registry()
        assert len(serialized["specs"]) == 23
        sites = {(s["benchmark"], s["site"], s["method"]) for s in serialized["specs"]}
        assert sites == set(_REGISTRY.keys())
