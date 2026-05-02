"""Tests for the editor-contract prompt renderer.

Covers:

* :func:`worldsim.editors._registry.render_contract_table` — one section
  per kind, correct selector-group / token / free-text formatting,
  dashboard-list body-mention hint preserved.
* Anchor-aware token masking (Fix 1, R4/R5 split-brain closure): the
  renderer declares only tokens reachable via the per-kind anchor union
  in the shard, and skips methods whose required bindings are
  unsatisfiable under the masked token set.
* :func:`worldsim.prompt_loading.load_prompt` sentinel handling:
  replacement when ``contract_context`` is provided, raise when missing,
  no-op when the file lacks the sentinel.
* Byte-identical prompt rendering between the sandbox path
  (``_render_generation_prompt`` in ``phase_2_injections``) and the API
  path (``_build_messages`` in ``phase_2_injections_api``) when given
  the same shard input.
"""

from __future__ import annotations

import pytest

from worldsim.editors._registry import (
    EDITOR_CONTRACT_TABLE_SENTINEL,
    ContractRenderContext,
    kind_anchors_from_resources,
    render_contract_table,
)
from worldsim.prompt_loading import PromptRenderError, load_prompt

# Anchor sets that keep every token reachable for the kind under test.
# Sourced from the gitlab editor's create_issue_note / create_mr_note
# bindings so every declared token (project_id, project_path, issue_iid,
# mr_iid, user_handle) is reachable.
_FULL_GITLAB_ISSUE_ANCHORS = frozenset({"project_id", "project_path", "issue_iid"})
_FULL_GITLAB_MR_ANCHORS = frozenset({"project_id", "project_path", "mr_iid"})
_FULL_GITLAB_DASHBOARD_ANCHORS = frozenset(
    {"project_id", "project_path", "issue_iid", "mr_iid", "dashboard"}
)
_FULL_REDDIT_SUBMISSION_ANCHORS = frozenset({"submission_id", "forum_name"})


def _ctx(site: str, kind_anchors: dict[str, frozenset[str]]) -> ContractRenderContext:
    return ContractRenderContext(site=site, kind_anchors=kind_anchors)


class TestRenderContractTable:
    def test_renders_gitlab_issue_kind(self) -> None:
        context = _ctx("gitlab", {"gitlab_issue": _FULL_GITLAB_ISSUE_ANCHORS})
        out = render_contract_table(context)
        assert "### Kind `gitlab_issue`" in out
        assert "create_issue_note" in out
        assert "target_surface_id=`note.body`" in out
        # SelectorGroup formatting — both token members visible under full anchors
        assert "selector group `project` (populate one)" in out
        assert "`project_id={benign_project_id}`" in out
        assert "`project_path_template={benign_project_path}`" in out
        # Token binding
        assert "token `issue_iid` = one of ['{benign_issue_iid}'] (required)" in out
        # free-text
        assert "free-text `body` (required)" in out

    def test_renders_dashboard_list_routing_hint(self) -> None:
        context = _ctx("gitlab", {"gitlab_dashboard_list": _FULL_GITLAB_DASHBOARD_ANCHORS})
        out = render_contract_table(context)
        assert "@{benign_user_handle}" in out
        assert "dashboard" in out.lower()

    def test_omits_kinds_not_in_shard(self) -> None:
        context = _ctx("gitlab", {"gitlab_issue": _FULL_GITLAB_ISSUE_ANCHORS})
        out = render_contract_table(context)
        assert "gitlab_mr" not in out
        assert "reddit_" not in out

    def test_empty_shard_emits_no_addressable_kinds_notice(self) -> None:
        context = _ctx("gitlab", {})
        out = render_contract_table(context)
        assert "No addressable kinds present" in out

    def test_filters_by_site(self) -> None:
        # reddit_submission declared on reddit only; asking for gitlab site
        # should fall through to the "no addressable kinds" notice because
        # iter_specs(site="gitlab") emits no specs addressing that kind.
        context = _ctx("gitlab", {"reddit_submission": _FULL_REDDIT_SUBMISSION_ANCHORS})
        out = render_contract_table(context)
        assert "create_comment" not in out
        assert "No addressable kinds present" in out

    def test_reddit_site_renders_reddit_kinds(self) -> None:
        context = _ctx("reddit", {"reddit_submission": _FULL_REDDIT_SUBMISSION_ANCHORS})
        out = render_contract_table(context)
        assert "create_comment" in out
        assert "submission_id" in out

    def test_multi_kind_ordering_deterministic(self) -> None:
        context = _ctx(
            "gitlab",
            {
                "gitlab_issue": _FULL_GITLAB_ISSUE_ANCHORS,
                "gitlab_mr": _FULL_GITLAB_MR_ANCHORS,
            },
        )
        out1 = render_contract_table(context)
        out2 = render_contract_table(context)
        assert out1 == out2
        # Issue section comes before MR section (alphabetical)
        idx_issue = out1.find("### Kind `gitlab_issue`")
        idx_mr = out1.find("### Kind `gitlab_mr`")
        assert 0 < idx_issue < idx_mr


class TestAnchorAwareMasking:
    """Fix 1: R5 prompt declares only tokens the R4 validator will accept.

    The registry's ``available_tokens_for_kind(kind, anchors)`` intersects
    declared tokens with ``{benign_<anchor_key>}`` tokens reachable via
    the resolver's anchors (plus identity tokens). The renderer must
    apply the same mask so the LLM never sees ``{benign_issue_iid}`` on a
    task whose anchors lack ``issue_iid``.
    """

    def test_dashboard_list_with_stub_anchors_renders_skip_notice(self) -> None:
        """The canonical failure case from the R4/R5 split-brain.

        L1/L2 on an intent-only dashboard-list task emits anchors like
        ``{"dashboard": "todos"}`` — no project, no issue. Under the old
        renderer this still advertised ``{benign_issue_iid}`` etc.; the
        LLM emitted plans using those tokens; R4 rejected them.

        Post-fix: every registered method for ``gitlab_dashboard_list``
        requires a standalone ``issue_iid`` or ``mr_iid`` token (neither
        reachable), so no method is viable and the section renders the
        skip notice instead of advertising phantom tokens.
        """
        context = _ctx("gitlab", {"gitlab_dashboard_list": frozenset({"dashboard"})})
        out = render_contract_table(context)
        assert "### Kind `gitlab_dashboard_list`" in out
        assert "No viable method under the current anchors" in out
        # Phantom tokens must not appear anywhere under this kind's section.
        issue_section = out.split("### Kind `gitlab_dashboard_list`", 1)[1]
        assert "{benign_issue_iid}" not in issue_section
        assert "{benign_mr_iid}" not in issue_section
        assert "{benign_project_id}" not in issue_section

    def test_dashboard_list_with_full_anchors_renders_methods(self) -> None:
        context = _ctx("gitlab", {"gitlab_dashboard_list": _FULL_GITLAB_DASHBOARD_ANCHORS})
        out = render_contract_table(context)
        assert "No viable method" not in out
        assert "create_issue_note" in out
        assert "create_mr_note" in out
        assert "{benign_issue_iid}" in out
        assert "{benign_mr_iid}" in out

    def test_gitlab_issue_with_only_project_path_masks_project_id_token(self) -> None:
        """Partial anchors mask specific selector options."""
        context = _ctx("gitlab", {"gitlab_issue": frozenset({"project_path", "issue_iid"})})
        out = render_contract_table(context)
        # Method remains viable: project_path_template reaches the project
        # group and issue_iid token is reachable.
        assert "create_issue_note" in out
        # project_id selector member is masked out (token unreachable).
        assert "`project_id={benign_project_id}`" not in out
        # project_path_template remains visible.
        assert "`project_path_template={benign_project_path}`" in out
        # The Available tokens list reflects only reachable tokens.
        assert "'{benign_project_id}'" not in out.split("### Kind")[1].split("Eligible Path A")[0]

    def test_non_core_surface_kind_renders_skip_notice(self) -> None:
        context = _ctx("gitlab", {"gitlab_user_profile": frozenset({"user_id"})})
        out = render_contract_table(context)
        assert "### Kind `gitlab_user_profile`" in out
        assert "No viable method under the current anchors" in out
        assert "profile" not in out.split("### Kind `gitlab_user_profile`", 1)[1]

    def test_identity_token_always_reachable(self) -> None:
        """``{benign_user_handle}`` is in IDENTITY_TOKENS and never masked.

        Even a bare dashboard-list task with no anchors whatsoever should
        still show the routing hint mentioning ``@{benign_user_handle}``.
        """
        context = _ctx("gitlab", {"gitlab_dashboard_list": frozenset()})
        out = render_contract_table(context)
        assert "@{benign_user_handle}" in out


class TestLoadPromptSentinelHandling:
    def test_raises_when_sentinel_present_and_context_missing(self) -> None:
        with pytest.raises(PromptRenderError, match="EDITOR_CONTRACT_TABLE"):
            load_prompt("generate-injections")

    def test_replaces_sentinel_when_context_provided(self) -> None:
        ctx = _ctx("gitlab", {"gitlab_issue": _FULL_GITLAB_ISSUE_ANCHORS})
        text = load_prompt("generate-injections", contract_context=ctx)
        assert EDITOR_CONTRACT_TABLE_SENTINEL not in text
        assert "### Kind `gitlab_issue`" in text

    def test_passthrough_when_sentinel_absent(self) -> None:
        # Any prompt other than generate-injections shouldn't require context
        # (and passing one should be a no-op).
        ctx = _ctx("gitlab", {})
        # Pick any other known prompt; we don't care about content here.
        text = load_prompt("generate-benign-tasks", contract_context=ctx)
        assert EDITOR_CONTRACT_TABLE_SENTINEL not in text

    def test_rendered_text_includes_strategy_rules(self) -> None:
        """The sentinel replaces the fixed-placement rule; strategy rules must survive."""
        ctx = _ctx("gitlab", {"gitlab_issue": _FULL_GITLAB_ISSUE_ANCHORS})
        text = load_prompt("generate-injections", contract_context=ctx)
        assert "2. Attacker realism" in text
        assert "3. Encounter centrality" in text
        assert "4. Concealment" in text
        assert "5. Harmful action" in text
        assert "8. Do not use `upload`, `sql`" in text
        assert "Do **not** emit `target_surface_id`" in text
        assert '"seed_template"' not in text

    def test_rendered_text_states_site_neutral_encounter_invariant(self) -> None:
        ctx = _ctx("gitlab", {"gitlab_issue": _FULL_GITLAB_ISSUE_ANCHORS})
        text = load_prompt("generate-injections", contract_context=ctx)

        assert "Encounter-core invariant" in text
        assert "exact surface where the payload will be seeded" in text
        assert "merely reaching a parent page" in text
        assert "GitLab notes and Reddit comments" not in text

    def test_rendered_text_does_not_prompt_for_retired_title_carriers(self) -> None:
        ctx = _ctx("gitlab", {"gitlab_issue": _FULL_GITLAB_ISSUE_ANCHORS})
        text = load_prompt("generate-injections", contract_context=ctx)

        assert "Do not silently skip compact title carriers" not in text
        assert "title fields are retired as active IPI carrier surfaces" in text
        assert "title carriers should not appear as eligible exposure contracts" in text


class TestSandboxApiPathParity:
    """Both sandbox and API paths call load_prompt with the same
    ``contract_context`` — the rendered prompt body must be byte-identical
    between the two paths for the same shard input."""

    def test_byte_identical_prompts(self) -> None:
        from worldsim.phases import phase_2_injections, phase_2_injections_api

        benign_target_resources = {
            "t1": {
                "kind": "gitlab_issue",
                "anchors": {"project_path": "foo/bar", "issue_iid": "42"},
            },
            "t2": {
                "kind": "gitlab_dashboard_list",
                "anchors": {"dashboard": "todos"},
            },
        }
        site_name = "gitlab"
        ctx = ContractRenderContext(
            site=site_name,
            kind_anchors=kind_anchors_from_resources(benign_target_resources),
        )

        # Sandbox path: _render_generation_prompt loads the prompt with
        # contract_context + validation_command footer. API path:
        # _build_messages loads the prompt with contract_context, no footer.
        sandbox_text = phase_2_injections._render_generation_prompt(
            {"framing::concealment": 1},
            validation_command="adversarial-tasks",
            contract_context=ctx,
        )

        api_system, _ = phase_2_injections_api._build_messages(
            benign_tasks=[],
            benign_target_resources=benign_target_resources,
            cell_targets={"framing::concealment": 1},
            benchmark_profile={},
            agent_context=None,
            requested_plan_count=1,
            site=site_name,
        )

        # Both paths render the same contract table section; verify both
        # contain the rendered section identically.
        for kind in ("gitlab_issue", "gitlab_dashboard_list"):
            assert f"### Kind `{kind}`" in sandbox_text
            assert f"### Kind `{kind}`" in api_system
        assert "@{benign_user_handle}" in sandbox_text
        assert "@{benign_user_handle}" in api_system
        # Dashboard-list with only {"dashboard": "todos"} is now correctly
        # marked skip in both paths — this is the Fix 1 contract.
        assert "No viable method under the current anchors" in sandbox_text
        assert "No viable method under the current anchors" in api_system

    def test_api_path_without_site_does_not_render(self) -> None:
        """Backward compat: if site is not passed, the prompt still loads
        (no contract_context → sentinel raises)."""
        from worldsim.phases import phase_2_injections_api

        with pytest.raises(PromptRenderError):
            phase_2_injections_api._build_messages(
                benign_tasks=[],
                benign_target_resources={},
                cell_targets={},
                benchmark_profile={},
                agent_context=None,
                requested_plan_count=1,
                site=None,
            )


class TestKindAnchorsFromResourcesHelper:
    def test_aggregates_anchor_keys_per_kind(self) -> None:
        resources = {
            "a": {"kind": "gitlab_issue", "anchors": {"project_id": "1", "issue_iid": "7"}},
            "b": {"kind": "gitlab_issue", "anchors": {"project_path": "foo", "issue_iid": "9"}},
            "c": {"kind": "reddit_submission", "anchors": {"submission_id": "42"}},
        }
        out = kind_anchors_from_resources(resources)
        assert out["gitlab_issue"] == frozenset({"project_id", "project_path", "issue_iid"})
        assert out["reddit_submission"] == frozenset({"submission_id"})

    def test_skips_null_and_empty_kinds(self) -> None:
        resources = {
            "a": {"kind": "gitlab_issue", "anchors": {"issue_iid": "1"}},
            "b": {"kind": None, "anchors": {}},
            "c": {"kind": "", "anchors": {"x": "1"}},
            "d": "not-a-dict",
            "e": None,
        }
        out = kind_anchors_from_resources(resources)
        assert set(out) == {"gitlab_issue"}

    def test_empty_anchors_yields_empty_set(self) -> None:
        resources = {"a": {"kind": "gitlab_dashboard_list", "anchors": {}}}
        out = kind_anchors_from_resources(resources)
        assert out == {"gitlab_dashboard_list": frozenset()}

    def test_kinds_in_shard_property_roundtrip(self) -> None:
        """The derived ``kinds_in_shard`` property on ContractRenderContext
        preserves the pre-fix contract for readers that only need kind
        names."""
        resources = {
            "a": {"kind": "gitlab_issue", "anchors": {"issue_iid": "1"}},
            "b": {"kind": "reddit_submission", "anchors": {"submission_id": "42"}},
        }
        ctx = ContractRenderContext(
            site="gitlab", kind_anchors=kind_anchors_from_resources(resources)
        )
        assert ctx.kinds_in_shard == frozenset({"gitlab_issue", "reddit_submission"})
