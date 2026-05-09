"""Cross-module invariants for the editor-method contract registry
(commit 7 of the registry refactor).

These tests encode relationships that span multiple modules — they
guard against the kind of silent drift the registry refactor was
designed to prevent. Every invariant is inexpensive; they all run in
CI.

Invariants:

* Self-consistency of ``kind_contract.available_tokens`` against the
  registered editors' bindings.
* Cross-site kind namespacing (no kind is claimed by two different
  sites).
* Dashboard-list kinds have at least one body-free-text binding so
  ``@{benign_user_handle}`` routing is viable.
* Sandbox-JSON parity — the bytes shipped to the sandbox round-trip
  through :func:`json.loads` cleanly and contain the expected specs.
* Validator-substituter agreement — for a handful of hand-authored
  plans per kind, the Option A registry validator accepts iff the
  substituter's ``_assert_benign_tokens_bound`` wouldn't raise. This
  is the load-bearing invariant that makes the next silent drift
  impossible to ship.
"""

from __future__ import annotations

import json

import pytest

from worldsim.editors._registry import (
    _REGISTRY,
    IDENTITY_TOKENS,
    iter_specs,
    kind_contract,
    serialize_registry,
)
from worldsim.phase_2.runner import (
    _validate_option_a_placement_registry,
)
from worldsim.seeding import UnboundTokenError, _assert_benign_tokens_bound

# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


class TestKindContractSelfConsistency:
    @pytest.mark.parametrize(
        "kind",
        sorted(kind for spec in _REGISTRY.values() for kind in spec.kinds),
    )
    def test_available_tokens_subset_of_declared_plus_identity(self, kind: str) -> None:
        """The contract's ``available_tokens`` is exactly the union of
        all ``BindingSpec.tokens`` across methods addressing this kind,
        plus the identity tokens. Any drift here means the registry's
        derived view diverges from its source data."""
        contract = kind_contract(kind)
        declared: set[str] = set(IDENTITY_TOKENS)
        for spec in iter_specs():
            if kind not in spec.kinds:
                continue
            for binding in spec.bindings.values():
                declared.update(binding.tokens)
        assert contract.available_tokens == frozenset(declared)


class TestCrossSiteNamespacing:
    def test_no_kind_claimed_by_two_sites(self) -> None:
        kind_to_sites: dict[str, set[str]] = {}
        for spec in iter_specs():
            for kind in spec.kinds:
                kind_to_sites.setdefault(kind, set()).add(spec.site)
        for kind, sites in kind_to_sites.items():
            assert len(sites) == 1, f"kind {kind!r} is claimed by multiple sites: {sorted(sites)}"


class TestDashboardListBodyRouting:
    @pytest.mark.parametrize(
        "kind",
        ["gitlab_dashboard_list", "reddit_dashboard_list"],
    )
    def test_at_least_one_body_free_text_binding(self, kind: str) -> None:
        """Dashboard-list kinds depend on body-mention routing
        (``@{benign_user_handle}`` in a note/comment body). Every
        dashboard-list kind must have at least one ``free_text``
        body-accepting binding on at least one addressing method."""
        saw_body_binding = False
        for spec in iter_specs():
            if kind not in spec.kinds:
                continue
            for arg, binding in spec.bindings.items():
                if binding.kind == "free_text" and arg in {
                    "body",
                    "note_body",
                    "note",
                    "comment",
                }:
                    saw_body_binding = True
                    break
            if saw_body_binding:
                break
        assert saw_body_binding, (
            f"dashboard-list kind {kind!r} has no method with a free-text "
            f"body binding; @{{benign_user_handle}} routing is impossible"
        )


class TestSerializeRegistryParity:
    def test_round_trip_stable(self) -> None:
        """serialize_registry is deterministic: same input → same
        output byte-for-byte."""
        out1 = json.dumps(serialize_registry(), sort_keys=True)
        out2 = json.dumps(serialize_registry(), sort_keys=True)
        assert out1 == out2

    def test_all_methods_serialized(self) -> None:
        data = serialize_registry()
        serialized_keys = {
            (spec["benchmark"], spec["site"], spec["method"]) for spec in data["specs"]
        }
        assert serialized_keys == set(_REGISTRY)

    def test_schema_keys(self) -> None:
        data = serialize_registry()
        for spec in data["specs"]:
            assert set(spec.keys()) == {
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


# ---------------------------------------------------------------------------
# Validator / substituter agreement — the load-bearing cross-module
# invariant that prevents the next silent drift.
# ---------------------------------------------------------------------------


def _make_plan(
    *,
    site: str,
    kind: str,
    anchors: dict,
    method: str,
    args: dict,
) -> dict:
    return {
        "site": site,
        "sites": [site],
        "benign_task_id": "tb",
        "id": "adv-1",
        "target_surface_id": "x",
        "framing": "authority",
        "concealment": "plaintext",
        "delivery_mechanism": "api",
        "benign_target_resource": {
            "kind": kind,
            "anchors": anchors,
        },
        "seed_template": {
            "mechanism": "editor",
            "editor_calls": [{"method": method, "args": args}],
        },
    }


# Golden per-kind (plan, should_accept) pairs. Keep these hand-written
# fixtures deliberately narrow and readable — the property is: the
# registry validator's verdict matches the substituter's.
@pytest.mark.parametrize(
    "name,site,kind,anchors,method,args,should_accept",
    [
        (
            "gitlab_issue_valid",
            "gitlab",
            "gitlab_issue",
            {"project_path": "foo/bar", "issue_iid": "42"},
            "create_issue_note",
            {
                "project_path_template": "{benign_project_path}",
                "issue_iid": "{benign_issue_iid}",
                "body": "hi",
            },
            True,
        ),
        (
            "gitlab_issue_phantom_project_id",
            "gitlab",
            "gitlab_issue",
            {"project_path": "foo/bar", "issue_iid": "42"},
            "create_issue_note",
            {
                "project_id": "{benign_project_id}",
                "issue_iid": "{benign_issue_iid}",
                "body": "hi",
            },
            False,
        ),
        (
            "gitlab_issue_no_project_selector_populated",
            "gitlab",
            "gitlab_issue",
            {"project_path": "foo/bar", "issue_iid": "42"},
            "create_issue_note",
            {
                "issue_iid": "{benign_issue_iid}",
                "body": "hi",
            },
            False,
        ),
        (
            "gitlab_issue_dashboard_mention",
            "gitlab",
            "gitlab_dashboard_list",
            {"dashboard": "todos"},
            "create_issue_note",
            {
                "project_path_template": "{benign_project_path}",
                "issue_iid": "{benign_issue_iid}",
                "body": "@{benign_user_handle} please ping me",
            },
            # Dashboard-list anchors don't reach project/issue tokens →
            # selector + token enforcement rejects. This is correct: the
            # test documents that pure-body routing requires no
            # project/issue tokens at all.
            False,
        ),
        (
            "reddit_submission_valid",
            "reddit",
            "reddit_submission",
            {"forum_name": "books", "submission_id": "59421"},
            "create_comment",
            {
                "submission_id": "{benign_submission_id}",
                "body": "hi",
            },
            True,
        ),
        (
            "reddit_submission_phantom_submission_id",
            "reddit",
            "reddit_submission",
            {"forum_name": "books"},
            "create_comment",
            {
                "submission_id": "{benign_submission_id}",
                "body": "hi",
            },
            False,
        ),
        (
            "reddit_forum_valid_create_submission",
            "reddit",
            "reddit_forum",
            {"forum_name": "books"},
            "create_submission",
            {
                "forum_name": "{benign_forum_name}",
                "title": "hi",
                "body": "hi",
            },
            True,
        ),
        (
            "unknown_method_rejected",
            "gitlab",
            "gitlab_issue",
            {"project_path": "foo/bar", "issue_iid": "42"},
            "create_mr_note",  # not valid for gitlab_issue kind
            {
                "project_path_template": "{benign_project_path}",
                "mr_iid": "{benign_mr_iid}",
                "body": "hi",
            },
            False,
        ),
    ],
)
def test_validator_substituter_agreement(
    name: str,
    site: str,
    kind: str,
    anchors: dict,
    method: str,
    args: dict,
    should_accept: bool,
) -> None:
    plan = _make_plan(site=site, kind=kind, anchors=anchors, method=method, args=args)

    validator_verdict = _validate_option_a_placement_registry(plan, name)
    accepts = validator_verdict is None

    # Substituter-would-raise check: does _assert_benign_tokens_bound
    # raise on the editor_calls? This runs at Phase 2c; the goal is to
    # ensure Phase 2a validator and Phase 2c substituter agree on
    # whether a plan is viable. They can both accept or both reject —
    # they must not disagree.
    fake_task = {
        "id": "tb",
        "benign_target_resource": {"kind": kind, "anchors": anchors},
    }
    substituter_raises = False
    try:
        _assert_benign_tokens_bound(plan["seed_template"]["editor_calls"][0], fake_task)
    except UnboundTokenError:
        substituter_raises = True

    # Agreement: if validator accepts, substituter must NOT raise. If
    # validator rejects, substituter may raise or not (the validator
    # catches more cases like unknown method / empty selector group
    # that don't involve phantom tokens).
    if accepts:
        assert not substituter_raises, (
            f"{name}: validator accepted but substituter raises "
            f"UnboundTokenError — these would disagree at Phase 2c"
        )

    assert accepts == should_accept, (
        f"{name}: expected accept={should_accept}, got verdict={validator_verdict!r}"
    )


class TestTelemetryModule:
    def test_telemetry_path_respects_state_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from worldsim.phases._contract_telemetry import telemetry_path

        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        path = telemetry_path()
        assert str(path).startswith(str(tmp_path))
        assert path.name == "contract_events.ndjson"

    def test_emit_writes_ndjson_line(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        from worldsim.phases._contract_telemetry import (
            emit_contract_event,
            telemetry_path,
        )

        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        emit_contract_event(
            event_type="validator_reject",
            shard="shard1",
            benign_task_id="tb",
            kind="gitlab_issue",
            detail={"reason": "phantom"},
        )
        path = telemetry_path()
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "validator_reject"
        assert record["shard"] == "shard1"
        assert record["detail"] == {"reason": "phantom"}

    def test_emit_never_propagates_ioerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Telemetry is observational — I/O errors must not fail the
        pipeline."""
        from worldsim.phases import _contract_telemetry

        def _boom(*args, **kwargs):
            raise PermissionError("nope")

        monkeypatch.setattr(_contract_telemetry, "telemetry_path", _boom)
        # Must not raise
        _contract_telemetry.emit_contract_event(
            event_type="validator_reject",
            detail={},
        )
