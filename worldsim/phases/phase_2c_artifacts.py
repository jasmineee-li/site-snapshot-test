"""Phase 2c artifact writing and report invariants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.phases.phase_2_feasibility import (
    FAILPOINT_DATASET,
    FAILPOINT_DROPPED_SOURCE_DATA,
    FAILPOINT_QUARANTINE,
    FAILPOINT_REPORT,
)
from worldsim.phases.phase_2_output import _merge_preserving_unfiltered_sites
from worldsim.phases.phase_2c_config import _feasibility_status


@dataclass(frozen=True)
class Phase2cArtifactWriteResult:
    verified: list[dict[str, Any]]
    infeasible: list[dict[str, Any]]
    dropped_source_data: list[dict[str, Any]]
    summary: dict[str, Any]
def _write_dropped_source_data_sidecar(
    path: Path,
    dropped_source_data: list[dict[str, Any]],
    *,
    sites_filter: set[str] | None,
) -> list[dict[str, Any]]:
    deduped = _merged_dropped_source_data(
        path,
        dropped_source_data,
        sites_filter=sites_filter,
    )
    write_json_atomic(
        path,
        deduped,
        failpoint_base=FAILPOINT_DROPPED_SOURCE_DATA,
    )
    return deduped


def _merged_dropped_source_data(
    path: Path,
    dropped_source_data: list[dict[str, Any]],
    *,
    sites_filter: set[str] | None,
) -> list[dict[str, Any]]:
    items = _merge_preserving_unfiltered_sites(
        path,
        dropped_source_data,
        sites_filter=sites_filter,
    )
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in items:
        key = (str(item.get("site") or ""), str(item.get("id") or ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)
    return deduped


def _write_phase_2c_artifacts(
    *,
    output_path: Path,
    infeasible_path: Path,
    dropped_source_path: Path,
    report_path: Path,
    verified: list[dict[str, Any]],
    infeasible: list[dict[str, Any]],
    dropped_source_data: list[dict[str, Any]],
    report_summary: dict[str, Any],
    sites_filter: set[str] | None,
    allow_unverified: bool = False,
) -> Phase2cArtifactWriteResult:
    """Write and validate the owned Phase 2c artifact set together."""
    merged_verified = _merge_preserving_unfiltered_sites(
        output_path,
        verified,
        sites_filter=sites_filter,
    )
    merged_infeasible = _merge_preserving_unfiltered_sites(
        infeasible_path,
        infeasible,
        sites_filter=sites_filter,
    )
    merged_dropped_source_data = _merged_dropped_source_data(
        dropped_source_path,
        dropped_source_data,
        sites_filter=sites_filter,
    )
    summary = _phase_2c_report_summary_with_artifacts(
        report_summary,
        verified=merged_verified,
        infeasible=merged_infeasible,
        dropped_source_data=merged_dropped_source_data,
        allow_unverified=allow_unverified,
    )
    _validate_phase_2c_artifact_payloads(
        verified=merged_verified,
        infeasible=merged_infeasible,
        dropped_source_data=merged_dropped_source_data,
        report_summary=summary,
        allow_unverified=allow_unverified,
    )
    write_json_atomic(
        infeasible_path,
        merged_infeasible,
        failpoint_base=FAILPOINT_QUARANTINE,
    )
    write_json_atomic(
        dropped_source_path,
        merged_dropped_source_data,
        failpoint_base=FAILPOINT_DROPPED_SOURCE_DATA,
    )
    write_json_atomic(
        output_path,
        merged_verified,
        failpoint_base=FAILPOINT_DATASET,
    )
    write_json_atomic(
        report_path,
        summary,
        failpoint_base=FAILPOINT_REPORT,
    )
    return Phase2cArtifactWriteResult(
        verified=merged_verified,
        infeasible=merged_infeasible,
        dropped_source_data=merged_dropped_source_data,
        summary=summary,
    )


def _phase_2c_report_summary_with_artifacts(
    report_summary: dict[str, Any],
    *,
    verified: list[dict[str, Any]],
    infeasible: list[dict[str, Any]],
    dropped_source_data: list[dict[str, Any]],
    allow_unverified: bool,
) -> dict[str, Any]:
    summary = dict(report_summary)
    summary["verified_count"] = _count_feasibility_status(verified, "verified")
    summary["infeasible_count"] = len(infeasible)
    if allow_unverified:
        summary["unverified_count"] = _count_feasibility_status(verified, "unverified")
    summary["skipped_already_verified_count"] = _count_idempotency_skipped(verified)
    summary["source_data_dropped_count"] = len(dropped_source_data)
    summary["source_data_dropped_by_kind"] = _source_data_dropped_by_kind(dropped_source_data)
    summary["per_site"] = _phase_2c_per_site_counts(verified, infeasible)
    return summary


def _count_feasibility_status(records: list[dict[str, Any]], status: str) -> int:
    return sum(1 for record in records if _feasibility_status(record) == status)


def _count_idempotency_skipped(records: list[dict[str, Any]]) -> int:
    return sum(
        1
        for record in records
        if isinstance(record.get("feasibility"), dict)
        and "last_reverify_skipped_at" in record["feasibility"]
    )


def _source_data_dropped_by_kind(dropped_source_data: list[dict[str, Any]]) -> dict[str, int]:
    by_kind: dict[str, int] = {}
    for record in dropped_source_data:
        issue = record.get("source_data_issue") if isinstance(record, dict) else None
        kind = str(issue.get("kind") or "unknown") if isinstance(issue, dict) else "unknown"
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return by_kind


def _phase_2c_per_site_counts(
    verified: list[dict[str, Any]],
    infeasible: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    per_site: dict[str, dict[str, int]] = {}

    def bucket_for(record: dict[str, Any]) -> dict[str, int]:
        site = str(record.get("site") or "").strip().lower() or "unknown"
        return per_site.setdefault(
            site,
            {"verified": 0, "infeasible": 0, "skipped": 0, "unverified": 0},
        )

    for record in verified:
        bucket = bucket_for(record)
        feasibility = record.get("feasibility") if isinstance(record, dict) else None
        status = _feasibility_status(record)
        if status == "unverified":
            bucket["unverified"] += 1
        elif isinstance(feasibility, dict) and "last_reverify_skipped_at" in feasibility:
            bucket["skipped"] += 1
        elif status == "verified":
            bucket["verified"] += 1
    for record in infeasible:
        bucket_for(record)["infeasible"] += 1
    return per_site


def _validate_phase_2c_artifact_payloads(
    *,
    verified: list[dict[str, Any]],
    infeasible: list[dict[str, Any]],
    dropped_source_data: list[dict[str, Any]],
    report_summary: dict[str, Any],
    allow_unverified: bool = False,
) -> None:
    if allow_unverified:
        if report_summary.get("unverified_count") != _count_feasibility_status(
            verified,
            "unverified",
        ):
            raise ValueError(
                "Phase 2c artifact invariant failed: report unverified_count "
                "does not match output dataset unverified records"
            )
    if report_summary.get("verified_count") != _count_feasibility_status(verified, "verified"):
        raise ValueError(
            "Phase 2c artifact invariant failed: report verified_count "
            "does not match output dataset verified records"
        )
    if report_summary.get("infeasible_count") != len(infeasible):
        raise ValueError(
            "Phase 2c artifact invariant failed: report infeasible_count "
            "does not match infeasible sidecar length"
        )
    expected_by_kind = _source_data_dropped_by_kind(dropped_source_data)
    if report_summary.get("source_data_dropped_count") != len(dropped_source_data):
        raise ValueError(
            "Phase 2c artifact invariant failed: report source_data_dropped_count "
            "does not match sidecar length"
        )
    if report_summary.get("source_data_dropped_by_kind") != expected_by_kind:
        raise ValueError(
            "Phase 2c artifact invariant failed: report source_data_dropped_by_kind "
            "does not match sidecar contents"
        )
    for record in dropped_source_data:
        issue = record.get("source_data_issue") if isinstance(record, dict) else None
        if not isinstance(issue, dict) or not issue.get("kind"):
            raise ValueError(
                "Phase 2c artifact invariant failed: dropped source-data record "
                "is missing source_data_issue.kind"
            )
    allowed_verified_statuses = {"verified"}
    if allow_unverified:
        allowed_verified_statuses.add("unverified")
    for record in verified:
        status = _feasibility_status(record)
        if status not in allowed_verified_statuses:
            raise ValueError(
                "Phase 2c artifact invariant failed: verified dataset contains "
                f"task {record.get('id')!r} with feasibility.status={status!r}"
            )
    for record in infeasible:
        status = _feasibility_status(record)
        if status != "infeasible":
            raise ValueError(
                "Phase 2c artifact invariant failed: infeasible dataset contains "
                f"task {record.get('id')!r} with feasibility.status={status!r}"
            )
