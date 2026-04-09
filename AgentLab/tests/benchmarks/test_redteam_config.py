from __future__ import annotations

from agentlab.benchmarks.redteam.config import parse_case_fields


def test_parse_case_fields_builds_pages_and_start_page_fallback() -> None:
    fields = parse_case_fields(
        {
            "doc": "<p>Review the inbox</p>",
            "description": "Open the latest urgent email",
            "simulation_config": {
                "pages": [
                    {
                        "id": "/mail",
                        "base_site_url": "https://mail.example.com",
                        "subdomains": ["/mail/u/0/#inbox"],
                        "details": {"/mail/u/0/#inbox": "Inbox"},
                        "screenshots": ["mail/inbox.png"],
                    }
                ],
                "success_condition": "Open the inbox",
            },
        },
        case_idx=3,
    )

    assert fields["behavior_id"] == "case3"
    assert fields["doc"] == "Review the inbox"
    assert fields["target"] == "Open the latest urgent email"
    assert fields["start_page"] == "https://mail.example.com/mail/u/0/#inbox"
    assert fields["success_condition"] == "Open the inbox"
    assert len(fields["pages"]) == 1
    assert fields["pages"][0].id == "mail"
    assert fields["pages"][0].screenshots == ["mail/inbox.png"]
