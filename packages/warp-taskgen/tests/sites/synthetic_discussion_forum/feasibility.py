"""Fake feasibility owner for the test-only discussion forum Site."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

from tests.sites.synthetic_discussion_forum.site import ORIGIN, SITE, THREAD_ID


class SyntheticDiscussionForumFeasibilityPolicy:
    benchmark = "webarena_verified"
    site = SITE

    def auth_self_test_path(self) -> str | None:
        return None

    def requires_authenticated_preflight(self) -> bool:
        return False

    def probe_targets(
        self,
        task: Mapping[str, object],
        instance_site_url: str,
    ) -> list[dict[str, str]]:
        if task.get("site") != SITE or task.get("sites") != [SITE]:
            return []
        if instance_site_url.rstrip("/") != ORIGIN:
            return []
        return [{"url": f"{ORIGIN}/threads/{THREAD_ID}", "thread_id": THREAD_ID}]

    def classify_probe(self, *, status_code: int = 200, url: str = "") -> str:
        parsed = urlsplit(url)
        if status_code == 200 and f"{parsed.scheme}://{parsed.netloc}" == ORIGIN:
            return "verified"
        return "unsupported"

    def decide_source_data(self, *, classification: str) -> str:
        return "admissible" if classification == "verified" else "ineligible"

    def counts_toward_run_bailout(self, classification: object) -> bool:
        return classification != "verified"

    def should_bailout_source_data_run(self, *, classification: str) -> bool:
        return classification == "unsupported"

    def restore_drop_on_run_bailout(self, issue: dict[str, object]) -> bool:
        del issue
        return False


feasibility_policy = SyntheticDiscussionForumFeasibilityPolicy()
