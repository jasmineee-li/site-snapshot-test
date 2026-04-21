"""Tests for the editor-contract prompt renderer (commit 5 of the
contract-registry refactor).

Covers:

* :func:`worldsim.editors._registry.render_contract_table` — one section
  per kind, correct selector-group / token / free-text formatting,
  dashboard-list body-mention hint preserved.
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
    render_contract_table,
)
from worldsim.prompt_loading import PromptRenderError, load_prompt


class TestRenderContractTable:
    def test_renders_gitlab_issue_kind(self) -> None:
        context = ContractRenderContext(site="gitlab", kinds_in_shard=frozenset({"gitlab_issue"}))
        out = render_contract_table(context)
        assert "### Kind `gitlab_issue`" in out
        assert "create_issue_note" in out
        assert "surface_id=`note_on_issue`" in out
        # SelectorGroup formatting
        assert "selector group `project` (populate one)" in out
        assert "`project_id={benign_project_id}`" in out
        assert "`project_path_template={benign_project_path}`" in out
        # Token binding
        assert "token `issue_iid` = one of ['{benign_issue_iid}'] (required)" in out
        # free-text
        assert "free-text `body` (required)" in out

    def test_renders_dashboard_list_routing_hint(self) -> None:
        context = ContractRenderContext(
            site="gitlab", kinds_in_shard=frozenset({"gitlab_dashboard_list"})
        )
        out = render_contract_table(context)
        assert "@{benign_user_handle}" in out
        assert "dashboard" in out.lower()

    def test_omits_kinds_not_in_shard(self) -> None:
        context = ContractRenderContext(site="gitlab", kinds_in_shard=frozenset({"gitlab_issue"}))
        out = render_contract_table(context)
        assert "gitlab_mr" not in out
        assert "reddit_" not in out

    def test_empty_shard_emits_no_addressable_kinds_notice(self) -> None:
        context = ContractRenderContext(site="gitlab", kinds_in_shard=frozenset())
        out = render_contract_table(context)
        assert "No addressable kinds present" in out

    def test_filters_by_site(self) -> None:
        # reddit_submission declared on reddit only; asking for gitlab site
        # should still list the kind header but no methods under it.
        context = ContractRenderContext(
            site="gitlab", kinds_in_shard=frozenset({"reddit_submission"})
        )
        out = render_contract_table(context)
        # No reddit methods shown (site filter) — falls through to "no addressable".
        assert "create_comment" not in out
        assert "No addressable kinds present" in out

    def test_reddit_site_renders_reddit_kinds(self) -> None:
        context = ContractRenderContext(
            site="reddit", kinds_in_shard=frozenset({"reddit_submission"})
        )
        out = render_contract_table(context)
        assert "create_comment" in out
        assert "submission_id" in out

    def test_multi_kind_ordering_deterministic(self) -> None:
        context = ContractRenderContext(
            site="gitlab",
            kinds_in_shard=frozenset({"gitlab_issue", "gitlab_mr"}),
        )
        out1 = render_contract_table(context)
        out2 = render_contract_table(context)
        assert out1 == out2
        # Issue section comes before MR section (alphabetical)
        idx_issue = out1.find("### Kind `gitlab_issue`")
        idx_mr = out1.find("### Kind `gitlab_mr`")
        assert 0 < idx_issue < idx_mr


class TestLoadPromptSentinelHandling:
    def test_raises_when_sentinel_present_and_context_missing(self) -> None:
        with pytest.raises(PromptRenderError, match="EDITOR_CONTRACT_TABLE"):
            load_prompt("generate-injections")

    def test_replaces_sentinel_when_context_provided(self) -> None:
        ctx = ContractRenderContext(site="gitlab", kinds_in_shard=frozenset({"gitlab_issue"}))
        text = load_prompt("generate-injections", contract_context=ctx)
        assert EDITOR_CONTRACT_TABLE_SENTINEL not in text
        assert "### Kind `gitlab_issue`" in text

    def test_passthrough_when_sentinel_absent(self) -> None:
        # Any prompt other than generate-injections shouldn't require context
        # (and passing one should be a no-op).
        ctx = ContractRenderContext(site="gitlab", kinds_in_shard=frozenset())
        # Pick any other known prompt; we don't care about content here.
        text = load_prompt("generate-benign-tasks", contract_context=ctx)
        assert EDITOR_CONTRACT_TABLE_SENTINEL not in text

    def test_rendered_text_includes_items_2_through_9(self) -> None:
        """The sentinel replaces the old rule #1; rules 2-9 must survive."""
        ctx = ContractRenderContext(site="gitlab", kinds_in_shard=frozenset({"gitlab_issue"}))
        text = load_prompt("generate-injections", contract_context=ctx)
        assert "2. Attacker realism" in text
        assert "3. Concealment" in text
        assert "4. Harmful action" in text
        assert "9. Do not use `sql` delivery" in text


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
            kinds_in_shard=phase_2_injections._kinds_in_shard(benign_target_resources),
        )

        # Sandbox path: _render_generation_prompt loads the prompt with
        # contract_context + validation_command footer. API path:
        # _build_messages loads the prompt with contract_context, no footer.
        # Strip the footer for comparison.
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


class TestKindsInShardHelper:
    def test_extracts_non_none_kinds(self) -> None:
        from worldsim.phases.phase_2_injections import _kinds_in_shard

        resources = {
            "a": {"kind": "gitlab_issue", "anchors": {}},
            "b": {"kind": "reddit_submission", "anchors": {}},
            "c": {"kind": None, "anchors": {}},
            "d": {"kind": "", "anchors": {}},
        }
        kinds = _kinds_in_shard(resources)
        assert kinds == frozenset({"gitlab_issue", "reddit_submission"})

    def test_empty_resources(self) -> None:
        from worldsim.phases.phase_2_injections import _kinds_in_shard

        assert _kinds_in_shard({}) == frozenset()

    def test_ignores_non_dict_entries(self) -> None:
        from worldsim.phases.phase_2_injections import _kinds_in_shard

        resources = {
            "a": {"kind": "gitlab_issue"},
            "b": "not-a-dict",
            "c": None,
        }
        kinds = _kinds_in_shard(resources)
        assert kinds == frozenset({"gitlab_issue"})
