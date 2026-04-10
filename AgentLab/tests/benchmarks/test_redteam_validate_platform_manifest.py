from __future__ import annotations

import json
from pathlib import Path

from agentlab.benchmarks.redteam.app_artifacts import (
    load_platform_manifest,
    platform_manifest_docs_path_errors,
    validate_platform_manifest,
)
from agentlab.benchmarks.redteam.validate_platform_manifest import main


def _platform_manifest_payload() -> dict:
    return {
        "apps": [
            {
                "app_id": "shared-mail",
                "platform": "mail",
                "docs_path": "apps/user-manuals/mail",
                "behaviors": [
                    {
                        "behavior_id": "demo-a",
                        "entry_route": "#/inbox",
                        "allowed_routes": ["#/inbox"],
                        "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                    },
                    {
                        "behavior_id": "demo-b",
                        "entry_route": "#/thread",
                        "allowed_routes": ["#/inbox", "#/thread"],
                        "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                        "mapping_rationale": "Uses the same shared inbox app.",
                    },
                ],
            }
        ]
    }


def test_validate_platform_manifest_rejects_missing_and_duplicate_behavior_mappings() -> None:
    payload = _platform_manifest_payload()
    payload["apps"][0]["behaviors"][1]["behavior_id"] = "demo-a"
    payload["apps"][0]["behaviors"][0]["allowed_routes"] = []

    errors = validate_platform_manifest(
        payload,
        known_behavior_ids={"demo-a", "demo-b"},
        require_full_coverage=True,
    )

    assert any("missing non-empty allowed_routes" in error for error in errors)
    assert any("duplicates behavior_id 'demo-a'" in error for error in errors)
    assert any("missing mappings for behavior_ids: demo-b" in error for error in errors)


def test_validate_platform_manifest_allows_partial_benchmark_coverage_by_default() -> None:
    payload = _platform_manifest_payload()
    payload["apps"][0]["behaviors"] = payload["apps"][0]["behaviors"][:1]

    errors = validate_platform_manifest(payload, known_behavior_ids={"demo-a", "demo-b"})

    assert errors == []


def test_validate_platform_manifest_rejects_path_like_ids() -> None:
    payload = _platform_manifest_payload()
    payload["apps"][0]["app_id"] = "../shared-mail"
    payload["apps"][0]["behaviors"][0]["behavior_id"] = "nested/demo-a"

    errors = validate_platform_manifest(payload)

    assert any("apps[0] app_id" in error and "path component" in error for error in errors)
    assert any(
        "apps[0].behaviors[0] behavior_id" in error and "path component" in error
        for error in errors
    )


def test_validate_platform_manifest_rejects_invalid_or_unreachable_mock_endpoints() -> None:
    payload = _platform_manifest_payload()
    payload["apps"][0]["behaviors"][0]["mock_endpoints"] = ["/api/reset"]

    errors = validate_platform_manifest(payload)

    assert any("invalid mock_endpoints" in error and "stay under /mock" in error for error in errors)

    payload = _platform_manifest_payload()
    payload["apps"][0]["behaviors"][0]["domain_bindings"] = [
        {"domain": "mail.example.com", "mode": "shim", "shim_path": "/shim"}
    ]
    payload["apps"][0]["behaviors"][0]["mock_endpoints"] = ["/mock/mail/submit"]

    errors = validate_platform_manifest(payload)

    assert any(
        "mock_endpoints entry '/mock/mail/submit' is not reachable through any primary_spa or mock domain binding"
        in error
        for error in errors
    )


def test_validate_platform_manifest_rejects_absolute_docs_paths() -> None:
    payload = _platform_manifest_payload()
    payload["apps"][0]["docs_path"] = "/tmp/manuals/mail"

    errors = validate_platform_manifest(payload)

    assert any(
        "apps[0] docs_path must be relative to the repository root" in error
        for error in errors
    )


def test_load_platform_manifest_rejects_invalid_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "platform_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_id": "shared-mail",
                        "platform": "mail",
                        "docs_path": "apps/user-manuals/mail",
                        "behaviors": [
                            {
                                "behavior_id": "demo-a",
                                "entry_route": "#/inbox",
                                "allowed_routes": ["#/thread"],
                                "domain_bindings": [{"domain": "mail.example.com", "mode": "primary_spa"}],
                            }
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        load_platform_manifest(manifest_path)
    except ValueError as exc:
        assert "entry_route '/#/inbox' is not present in allowed_routes" in str(exc)
    else:
        raise AssertionError("expected invalid platform manifest to raise ValueError")


def test_platform_manifest_docs_path_errors_checks_relative_paths(tmp_path: Path) -> None:
    payload = _platform_manifest_payload()

    errors = platform_manifest_docs_path_errors(payload, repo_root_path=tmp_path)
    assert errors == [
        f"apps[0] docs_path 'apps/user-manuals/mail' does not exist relative to {tmp_path}"
    ]

    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True)

    assert platform_manifest_docs_path_errors(payload, repo_root_path=tmp_path) == []


def test_validate_platform_manifest_cli_checks_behavior_coverage_and_docs_paths(tmp_path: Path) -> None:
    docs_dir = tmp_path / "apps" / "user-manuals" / "mail"
    docs_dir.mkdir(parents=True)
    (docs_dir / "manual.md").write_text("# Mail\n", encoding="utf-8")

    manifest_path = tmp_path / "platform_manifest.json"
    manifest_path.write_text(json.dumps(_platform_manifest_payload(), indent=2), encoding="utf-8")

    benchmark_path = tmp_path / "behaviors.json"
    benchmark_path.write_text(
        json.dumps(
            [
                {"id": "demo-a", "doc": "Open inbox"},
                {"id": "demo-b", "doc": "Open thread"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--platform-manifest",
                str(manifest_path),
                "--benchmark-file",
                str(benchmark_path),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    benchmark_path.write_text(
        json.dumps([{"id": "demo-a", "doc": "Open inbox"}], indent=2),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--platform-manifest",
                str(manifest_path),
                "--benchmark-file",
                str(benchmark_path),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "--platform-manifest",
                str(manifest_path),
                "--benchmark-file",
                str(benchmark_path),
                "--require-full-coverage",
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )
