"""Tests for the fail-loud substituter (commit 6 of the contract-registry
refactor).

Covers:

* :class:`warp_taskgen.seeding.UnboundTokenError` payload shape + behavior.
* :func:`warp_taskgen.seeding._assert_benign_tokens_bound` accepts valid
  tokens, rejects phantom tokens, and no-ops on records without a
  ``benign_target_resource`` or with a null kind.
* :func:`warp_taskgen.seeding._collect_benign_tokens` walks nested values
  correctly.
* :func:`warp_taskgen.phase_2.phase_2c.verifier._verify_one` categorizes
  ``UnboundTokenError`` as ``error.kind = "contract_violation"``,
  distinct from ``"schema_mismatch"``.
"""

from __future__ import annotations

import pytest

from warp_taskgen.seeding import (
    UnboundTokenError,
    _assert_benign_tokens_bound,
    _collect_benign_tokens,
)


class TestCollectBenignTokens:
    def test_empty_string(self) -> None:
        assert _collect_benign_tokens("") == set()

    def test_string_with_no_tokens(self) -> None:
        assert _collect_benign_tokens("hello world") == set()

    def test_string_with_one_benign_token(self) -> None:
        assert _collect_benign_tokens("{benign_issue_iid}") == {"{benign_issue_iid}"}

    def test_string_with_multiple_tokens(self) -> None:
        got = _collect_benign_tokens("note: project {benign_project_id}, issue {benign_issue_iid}")
        assert got == {"{benign_project_id}", "{benign_issue_iid}"}

    def test_non_benign_tokens_ignored(self) -> None:
        assert _collect_benign_tokens("{topic} {foo.bar}") == set()

    def test_dict_walk(self) -> None:
        value = {
            "project_id": "{benign_project_id}",
            "issue_iid": "{benign_issue_iid}",
            "body": "hi",
        }
        assert _collect_benign_tokens(value) == {
            "{benign_project_id}",
            "{benign_issue_iid}",
        }

    def test_list_walk(self) -> None:
        assert _collect_benign_tokens(["{benign_forum_name}", "plain"]) == {"{benign_forum_name}"}

    def test_nested_structure(self) -> None:
        value = {
            "method": "create_issue_note",
            "args": {
                "project_id": "{benign_project_id}",
                "issue_iid": "{benign_issue_iid}",
                "body": "hi",
            },
        }
        assert _collect_benign_tokens(value) == {
            "{benign_project_id}",
            "{benign_issue_iid}",
        }


class TestAssertBenignTokensBound:
    def _task_gitlab_issue_with_path_only(self) -> dict:
        # Resolver-emitted anchors for gitlab_issue: project_path + issue_iid.
        # No project_id anchor — so {benign_project_id} is unreachable.
        return {
            "id": "t1",
            "benign_target_resource": {
                "kind": "gitlab_issue",
                "anchors": {"project_path": "foo/bar", "issue_iid": "42"},
            },
        }

    def test_bound_tokens_pass(self) -> None:
        task = self._task_gitlab_issue_with_path_only()
        call = {
            "args": {
                "project_path_template": "{benign_project_path}",
                "issue_iid": "{benign_issue_iid}",
                "body": "hi",
            },
        }
        _assert_benign_tokens_bound(call, task)

    def test_phantom_token_raises(self) -> None:
        task = self._task_gitlab_issue_with_path_only()
        call = {
            "args": {
                "project_id": "{benign_project_id}",
                "issue_iid": "{benign_issue_iid}",
                "body": "hi",
            },
        }
        with pytest.raises(UnboundTokenError) as exc_info:
            _assert_benign_tokens_bound(call, task)
        exc = exc_info.value
        assert exc.token == "{benign_project_id}"
        assert exc.task_id == "t1"
        assert exc.kind == "gitlab_issue"
        assert "{benign_project_id}" not in exc.available_tokens
        assert "{benign_project_path}" in exc.available_tokens
        assert "{benign_user_handle}" in exc.available_tokens
        assert exc.anchors == {"project_path": "foo/bar", "issue_iid": "42"}

    def test_user_handle_always_available(self) -> None:
        # Even with zero anchors, {benign_user_handle} should be reachable
        # (identity token — comes from agent context, not resolver anchors).
        # The reading comes from the default Runtime Composition's token scope.
        from warp_taskgen.runtime_composition import RuntimeComposition

        composition = RuntimeComposition.default()
        task = {
            "id": "t2",
            "benign_target_resource": {
                "kind": "gitlab_dashboard_list",
                "anchors": {},
            },
        }
        call = {"args": {"body": "@{benign_user_handle} do X"}}
        _assert_benign_tokens_bound(
            call,
            task,
            seed_registry=composition.seed_registry,
            seed_token_scope=composition.seed_token_scope,
        )

    def test_noop_when_resource_missing(self) -> None:
        # Legacy or shopping task with no benign_target_resource — skip.
        task = {"id": "legacy"}
        call = {"args": {"project_id": "{benign_project_id}"}}
        _assert_benign_tokens_bound(call, task)  # must not raise

    def test_noop_when_kind_is_null(self) -> None:
        task = {
            "id": "pending",
            "benign_target_resource": {"kind": None, "anchors": {}},
        }
        call = {"args": {"project_id": "{benign_project_id}"}}
        _assert_benign_tokens_bound(call, task)

    def test_noop_when_no_benign_tokens_referenced(self) -> None:
        task = self._task_gitlab_issue_with_path_only()
        call = {"args": {"title": "hello", "body": "plain text"}}
        _assert_benign_tokens_bound(call, task)

    def test_noop_when_task_is_not_dict(self) -> None:
        call = {"args": {"project_id": "{benign_project_id}"}}
        _assert_benign_tokens_bound(call, None)
        _assert_benign_tokens_bound(call, "not-a-dict")

    def test_reddit_submission_all_bound(self) -> None:
        task = {
            "id": "r1",
            "benign_target_resource": {
                "kind": "reddit_submission",
                "anchors": {"forum_name": "books", "submission_id": "59421"},
            },
        }
        call = {
            "args": {
                "forum_name": "{benign_forum_name}",
                "submission_id": "{benign_submission_id}",
                "body": "hi",
            },
        }
        _assert_benign_tokens_bound(call, task)

    def test_exception_message_mentions_available_tokens(self) -> None:
        task = self._task_gitlab_issue_with_path_only()
        call = {"args": {"mr_iid": "{benign_mr_iid}"}}
        with pytest.raises(UnboundTokenError) as exc_info:
            _assert_benign_tokens_bound(call, task)
        msg = str(exc_info.value)
        assert "{benign_mr_iid}" in msg
        assert "Available:" in msg


class TestFeasibilityCategorization:
    """Phase 2c must emit ``error.kind = "contract_violation"`` for
    UnboundTokenError (not schema_mismatch) so dashboards can track
    commit-4/6 contract hits separately from JSON shape issues."""

    def test_unbound_token_error_is_valueerror_subclass(self) -> None:
        # Needed so the except chain ordering works correctly: the more
        # specific except UnboundTokenError must precede the generic
        # except (ValueError, RuntimeError).
        assert issubclass(UnboundTokenError, ValueError)

    def test_contract_violation_kind_wired_in_feasibility(self) -> None:
        # Verify the except branch exists in the canonical Phase 2c owner by
        # asserting the source contains the handler. This is a static
        # guard — a full integration test would require a live instance.
        from pathlib import Path

        src = Path("warp_taskgen/phase_2/phase_2c/verifier.py").read_text()
        assert "except UnboundTokenError" in src
        assert 'kind="contract_violation"' in src


class TestSeedTokenScopeIsACompositionProperty:
    """#309: a GitLab identity token must stay reachable on the default path.

    Before the Runtime Composition carried the scope, passing an explicit
    registry silently switched the check to the method-scoped reading, which
    has no identity union — so ``{benign_user_handle}`` in a
    ``create_issue_note`` seed raised ``UnboundTokenError`` even though the
    kind-scoped contract publishes it. The scope now comes from the
    composition, so the default composition keeps the kind-scoped reading and
    only a named POC composition gets the method-scoped one.
    """

    @staticmethod
    def _gitlab_issue_task() -> dict[str, object]:
        return {
            "id": "adv-gitlab-issue-note",
            "site": "gitlab",
            "benchmark": "webarena_verified",
            "benign_target_resource": {
                "kind": "issue",
                "anchors": {"project_id": "42", "issue_iid": "7"},
            },
        }

    def test_default_composition_binds_the_gitlab_identity_token(self) -> None:
        from warp_taskgen.runtime_composition import RuntimeComposition

        composition = RuntimeComposition.default()
        assert composition.seed_token_scope == "kind"

        # Must not raise: the kind-scoped union publishes the identity token.
        _assert_benign_tokens_bound(
            "hi {benign_user_handle}",
            self._gitlab_issue_task(),
            seed_registry=composition.seed_registry,
            seed_token_scope=composition.seed_token_scope,
        )

    def test_method_scope_still_rejects_a_token_the_method_never_declares(self) -> None:
        """The method-scoped reading (the #309 branch) is unchanged.

        This is the exact branch the pre-composition code selected whenever a
        registry was supplied, and it still fails closed.
        """

        from warp_taskgen.seeding.site_contracts import default_seed_registry

        with pytest.raises(UnboundTokenError) as exc_info:
            _assert_benign_tokens_bound(
                "hi {benign_user_handle}",
                self._gitlab_issue_task(),
                seed_registry=default_seed_registry(),
                seed_token_scope="method",
            )
        assert exc_info.value.token == "{benign_user_handle}"

    def test_poc_composition_rejects_a_token_its_method_does_not_declare(self) -> None:
        from warp_taskgen.runtime_composition import classifieds_listing_reply_poc

        composition = classifieds_listing_reply_poc()
        assert composition.seed_token_scope == "method"

        call = {
            "site": "classifieds",
            "benchmark": "visualwebarena",
            "method": "create_listing_reply",
            "args": {"listing_id": "{benign_listing_id}", "body": "hi {benign_user_handle}"},
        }
        task = {
            "id": "adv-classifieds-anchor",
            "site": "classifieds",
            "benchmark": "visualwebarena",
            "benign_target_resource": {
                "kind": "listing",
                "anchors": {"listing_id": 17, "user_handle": "someone"},
            },
        }
        with pytest.raises(UnboundTokenError) as exc_info:
            _assert_benign_tokens_bound(
                call,
                task,
                seed_registry=composition.seed_registry,
                seed_token_scope=composition.seed_token_scope,
            )
        assert exc_info.value.token == "{benign_user_handle}"

    def test_method_scope_requires_a_registry(self) -> None:
        with pytest.raises(ValueError, match="method-scoped seed token checks"):
            _assert_benign_tokens_bound(
                "hi {benign_user_handle}",
                self._gitlab_issue_task(),
                seed_token_scope="method",
            )

    def test_unknown_scope_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="seed_token_scope"):
            _assert_benign_tokens_bound(
                "hi {benign_user_handle}",
                self._gitlab_issue_task(),
                seed_token_scope="site",
            )
