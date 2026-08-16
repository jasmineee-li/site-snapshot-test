"""Fake feasibility owner for the test-only discussion forum Site."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

from tests.sites.synthetic_discussion_forum.site import ORIGIN, SITE


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
        start_urls = task.get("start_urls")
        if not isinstance(start_urls, list) or len(start_urls) != 1:
            return []
        parsed = urlsplit(str(start_urls[0]))
        if f"{parsed.scheme}://{parsed.netloc}" != ORIGIN:
            return []
        thread_id = parsed.path.removeprefix("/threads/").rstrip("/")
        if not thread_id.isdigit() or parsed.path != f"/threads/{thread_id}":
            return []
        return [{"url": f"{ORIGIN}/threads/{thread_id}", "thread_id": thread_id}]

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
