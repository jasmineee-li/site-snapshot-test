# ruff: noqa
# Auto-split from tests/test_phase_2_target_resolver.py; shared helpers live in tests/phase_2/target_resolution/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

class TestAnchorContractConformance:
    def test_none_kind_passes(self) -> None:
        _assert_anchor_contract_conformance({"kind": None, "anchors": {}})

    def test_missing_kind_passes(self) -> None:
        # Pending/empty records without a kind field trivially pass.
        _assert_anchor_contract_conformance({"anchors": {}})

    def test_known_kind_passes(self) -> None:
        _assert_anchor_contract_conformance(
            {
                "kind": "gitlab_issue",
                "anchors": {"project_path": "foo/bar", "issue_iid": "42"},
            }
        )

    def test_all_real_resolver_kinds_pass(self) -> None:
        # Every ResourceKind emitted by the resolver must be addressable by
        # at least one registered editor method. If this fails, the
        # resolver and the editor contracts have drifted.
        for kind in (
            "gitlab_issue",
            "gitlab_mr",
            "gitlab_search_result",
            "gitlab_dashboard_list",
            "reddit_submission",
            "reddit_forum",
            "reddit_dashboard_list",
        ):
            _assert_anchor_contract_conformance({"kind": kind, "anchors": {}})

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ResolverContractDriftError, match="no editor method"):
            _assert_anchor_contract_conformance(
                {"kind": "synthetic_never_registered_kind", "anchors": {}}
            )

    def test_derive_benign_target_resource_honors_conformance(self) -> None:
        # Running the real L1 path on a well-formed task must not raise.
        task = _gitlab_task(eval_url="__GITLAB__/byteblaze/a11yproject/-/issues/17")
        record = derive_benign_target_resource(task, PLACEHOLDERS)
        assert record["kind"] == "gitlab_issue"
