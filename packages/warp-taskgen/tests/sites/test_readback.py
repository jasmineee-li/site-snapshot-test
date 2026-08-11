from __future__ import annotations

from tests.sites.test_read_surface import FakeReadSurfaceSite
from worldsim.sites import ReadbackDecision, ReadbackFailure, ReadbackObservation, SiteCatalog
from worldsim.sites.gitlab import GitLabSite
from worldsim.sites.reddit import RedditSite


def test_gitlab_interprets_exact_note_and_description() -> None:
    gitlab = GitLabSite()
    note = gitlab.interpret_readback(
        ReadbackObservation(
            "resource_identity",
            {"note_id": 42},
            '[{"notes":[{"id":"42","note_html":"<p>seeded <b>body</b></p>"}]}]',
        )
    )
    assert note == ReadbackDecision(True, "note_identity_present", "note_id=42", "seeded body")
    absent = gitlab.interpret_readback(
        ReadbackObservation("resource_identity", {"note_id": 42}, '[{"notes":[{"id":142}]}]')
    )
    assert (absent.verified, absent.reason) == (False, "note_identity_absent")
    assert gitlab.interpret_readback(
        ReadbackObservation(
            "resource_signature",
            {"project_id": 1, "issue_iid": 2},
            {"normalized_description": "seeded body", "normalized_signature": "body"},
            signature="body",
        )
    ).verified
    missing_identity = gitlab.interpret_readback(
        ReadbackObservation(
            "resource_signature",
            {},
            {"normalized_description": "seeded body", "normalized_signature": "body"},
            signature="body",
        )
    )
    assert (missing_identity.verified, missing_identity.reason) == (
        False,
        "missing_resource_identity",
    )
    signature_mismatch = gitlab.interpret_readback(
        ReadbackObservation(
            "resource_signature",
            {"project_id": 1, "issue_iid": 2},
            {"normalized_description": "seeded body", "normalized_signature": "other"},
            signature="body",
        )
    )
    assert (signature_mismatch.verified, signature_mismatch.reason) == (
        False,
        "resource_signature_mismatch",
    )
    identity_mismatch = gitlab.interpret_readback(
        ReadbackObservation(
            "resource_signature",
            {"project_id": 1, "issue_iid": 2},
            {
                "normalized_description": "seeded body",
                "normalized_signature": "body",
                "iid": 99,
            },
            signature="body",
        )
    )
    assert (identity_mismatch.verified, identity_mismatch.reason) == (
        False,
        "resource_identity_mismatch",
    )


def test_reddit_requires_exact_first_visible_painted_comment() -> None:
    reddit = RedditSite()
    base = {
        "ok": True,
        "comment_id": "901",
        "first_visible_comment_id": "901",
        "match_found_in_comment": True,
        "visible_at_entry": True,
        "visible_nonspace_chars": 12,
        "requires_expand": False,
    }
    observation = lambda payload: ReadbackObservation(  # noqa: E731
        "comment_visibility", {"comment_id": "901"}, payload
    )
    assert reddit.interpret_readback(observation(base)).verified
    for field, value, reason in (
        ("comment_id", "902", "comment_identity_mismatch"),
        ("first_visible_comment_id", "902", "different_first_visible_comment"),
        ("match_found_in_comment", False, "signature_not_in_seed_comment"),
        ("visible_at_entry", False, "seed_comment_not_visible_at_entry"),
        ("visible_nonspace_chars", 0, "seed_comment_not_painted"),
        ("visible_nonspace_chars", float("nan"), "seed_comment_not_painted"),
        ("requires_expand", True, "seed_comment_requires_expand"),
    ):
        probe = dict(base)
        probe[field] = value
        decision = reddit.interpret_readback(observation(probe))
        assert (decision.verified, decision.reason) == (False, reason)
    for missing_field in (
        "match_found_in_comment",
        "visible_at_entry",
        "visible_nonspace_chars",
        "requires_expand",
    ):
        probe = dict(base)
        del probe[missing_field]
        assert not reddit.interpret_readback(observation(probe)).verified


def test_injected_site_readback_is_scoped_and_errors_fail_closed() -> None:
    class FakeReadbackSite(FakeReadSurfaceSite):
        def interpret_readback(self, observation: ReadbackObservation) -> ReadbackDecision:
            return ReadbackDecision(observation.payload == "verified", "fake_decision")

    bound = SiteCatalog([FakeReadbackSite()]).bind(site="fake", origin="https://fake.local")
    assert bound.interpret_readback(
        ReadbackObservation("resource_identity", {"comment_id": "1"}, "verified")
    ).verified
    assert SiteCatalog().sites == ("gitlab", "reddit")

    class BrokenReadbackSite(FakeReadbackSite):
        def interpret_readback(self, observation: ReadbackObservation) -> ReadbackDecision:
            raise RuntimeError("broken adapter")

    failure = (
        SiteCatalog([BrokenReadbackSite()])
        .bind(site="fake", origin="https://fake.local")
        .interpret_readback(ReadbackObservation("resource_identity", {}, "verified"))
    )
    assert isinstance(failure, ReadbackFailure)
    assert failure.reason == "readback_adapter_error"


def test_bound_readback_rejects_invalid_input_and_unsupported_benchmark() -> None:
    bound = SiteCatalog([GitLabSite()]).bind(site="gitlab", origin="https://gitlab.local")
    failure = bound.interpret_readback(object())  # type: ignore[arg-type]
    assert isinstance(failure, ReadbackFailure)
    assert failure.reason == "invalid_readback_observation"

    unsupported = SiteCatalog([GitLabSite()]).bind(
        benchmark="unsupported", site="gitlab", origin="https://gitlab.local"
    )
    failure = unsupported.interpret_readback(
        ReadbackObservation("resource_identity", {"note_id": 1}, "[]")
    )
    assert isinstance(failure, ReadbackFailure)
    assert failure.reason == "unsupported_benchmark"


def test_observation_snapshots_nested_payload() -> None:
    payload = {
        "normalized_description": "before",
        "normalized_signature": "before",
        "nested": {"value": "before"},
    }
    observation = ReadbackObservation(
        "resource_signature",
        {"project_id": 1, "issue_iid": 2},
        payload,
        signature="before",
    )
    payload["normalized_description"] = "after"
    payload["nested"]["value"] = "after"

    decision = GitLabSite().interpret_readback(observation)
    assert decision.verified
    assert observation.payload["nested"]["value"] == "before"


def test_interpreters_reject_container_identity_tokens() -> None:
    gitlab = GitLabSite().interpret_readback(
        ReadbackObservation(
            "resource_signature",
            {"project_id": [], "issue_iid": {}},
            {"normalized_description": "body", "normalized_signature": "body"},
            signature="body",
        )
    )
    assert (gitlab.verified, gitlab.reason) == (False, "missing_resource_identity")

    reddit = RedditSite().interpret_readback(
        ReadbackObservation(
            "comment_visibility",
            {"comment_id": []},
            {
                "ok": True,
                "comment_id": [],
                "first_visible_comment_id": [],
                "match_found_in_comment": True,
                "visible_at_entry": True,
                "visible_nonspace_chars": 1,
                "requires_expand": False,
            },
        )
    )
    assert (reddit.verified, reddit.reason) == (False, "missing_comment_identity")
