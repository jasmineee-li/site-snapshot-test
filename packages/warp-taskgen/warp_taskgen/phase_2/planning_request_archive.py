"""Observational request and output joins for the ordinary Phase 2a planner."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.host_api_observability import summarize_provider_response
from warp_taskgen.host_api_request_archive import archive_model_request
from warp_taskgen.phase_2.planning_types import SiteInjectionResult

logger = logging.getLogger(__name__)
_PLANNING_ARCHIVE: ContextVar[PlanningRequestArchive | None] = ContextVar(
    "phase_2a_request_archive", default=None
)


@dataclass
class PlanningRequestArchive:
    """One admitted shard invocation, shared only by its own transport retries."""

    run_root: Path
    label: str
    site: str | None
    input_task_ids: list[str]
    call_id: str = field(default_factory=lambda: uuid4().hex)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.run_root / "phase_2" / "planning_requests" / self.call_id

    def record_request(
        self,
        kwargs: dict[str, Any],
        *,
        configured_model: str,
        client_provider: str | None,
        dispatched_task_ids: list[str],
    ) -> int:
        requests = self.diagnostics.setdefault("requests", [])
        index = len(requests) + 1
        reference: dict[str, Any] = {"call_id": self.call_id, "request_index": index}
        try:
            reference.update(
                archive_model_request(
                    kwargs,
                    metadata={
                        "schema_version": 1,
                        "phase": "phase_2a",
                        "call_id": self.call_id,
                        "request_index": index,
                        "label": self.label,
                        "site": self.site,
                        "input_task_ids": self.input_task_ids,
                        "dispatched_task_ids": dispatched_task_ids,
                        "configured_model": configured_model,
                        "resolved_client_provider": client_provider,
                    },
                    path=self.directory / f"{index}.json",
                    run_root=self.run_root,
                    write_json=write_json_atomic,
                )
            )
            if reference["status"] == "partial_retention":
                logger.warning("Phase 2a request retention incomplete, call=%s", self.call_id)
        except Exception as exc:
            reference.update(status="retention_failed", error_type=type(exc).__name__)
            self._warn(exc)
        requests.append(reference)
        self._persist()
        return index

    def record_response(self, index: int, response: Any) -> None:
        try:
            summary = summarize_provider_response(response)
            self.diagnostics.setdefault("responses", []).append(
                {"request_index": index, "response": summary}
            )
        except Exception as exc:
            self.diagnostics.setdefault("observer_errors", []).append(type(exc).__name__)
            self._warn(exc)
        self._persist()

    def record_error(self, index: int, error: BaseException) -> None:
        # Exception text can contain provider request content or credentials.
        self.diagnostics.setdefault("transport_errors", []).append(
            {"request_index": index, "error_type": type(error).__name__}
        )
        self._persist()

    def record_parse(self, plans: list[dict[str, Any]] | None) -> None:
        self.diagnostics["tool_parse"] = {
            "request_index": len(self.diagnostics.get("requests", [])),
            "status": "no_plans_tool" if plans is None else "parsed",
            "plan_count": len(plans) if plans is not None else 0,
        }
        self._persist()

    def record_parse_error(self, error: Exception) -> None:
        self.diagnostics["tool_parse"] = {
            "request_index": len(self.diagnostics.get("requests", [])),
            "status": "parse_error",
            "error_type": type(error).__name__,
        }
        self._persist()

    def record_output(
        self, result: SiteInjectionResult, *, status: str, output_path: Path | None
    ) -> None:
        try:
            output: dict[str, Any] = {
                "status": status,
                "output_task_ids": [str(task.get("id") or "") for task in result.adversarial_tasks],
                "validation_error_count": len(result.errors),
                "validation_errors": [str(error)[:600] for error in result.errors[:20]],
                "validation_errors_truncated": len(result.errors) > 20
                or any(len(str(error)) > 600 for error in result.errors),
                "validation_errors_sha256": hashlib.sha256(
                    json.dumps(result.errors, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
            }
            if output_path is not None:
                output["path"] = output_path.relative_to(self.run_root).as_posix()
                output["file_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
            self.diagnostics["host_output"] = output
        except Exception as exc:
            self.diagnostics.setdefault("observer_errors", []).append(type(exc).__name__)
            self._warn(exc)
        self._persist()

    def _persist(self) -> None:
        # Deterministic planners and reused shards must never invent a request.
        if not self.diagnostics.get("requests"):
            return
        try:
            write_json_atomic(
                self.directory / "diagnostics.json",
                {
                    "schema_version": 1,
                    "phase": "phase_2a",
                    "call_id": self.call_id,
                    "label": self.label,
                    "site": self.site,
                    "input_task_ids": self.input_task_ids,
                    **self.diagnostics,
                },
            )
        except Exception as exc:
            self.diagnostics.setdefault("observer_errors", []).append(type(exc).__name__)
            self._warn(exc)

    def _warn(self, error: Exception) -> None:
        logger.warning(
            "Phase 2a request retention failed (%s), call=%s", type(error).__name__, self.call_id
        )


@contextmanager
def bind_planning_request_archive(
    run_root: Path, *, label: str, site: str | None, input_task_ids: list[str]
) -> Iterator[PlanningRequestArchive]:
    archive = PlanningRequestArchive(run_root, label, site, list(input_task_ids))
    token = _PLANNING_ARCHIVE.set(archive)
    try:
        yield archive
    finally:
        _PLANNING_ARCHIVE.reset(token)


def current_planning_request_archive() -> PlanningRequestArchive | None:
    """Standalone API calls retain only inside an explicit feature binding."""
    return _PLANNING_ARCHIVE.get()


def finish_planning_shard(
    archive: PlanningRequestArchive | None,
    result: SiteInjectionResult,
    *,
    status: str,
    output_path: Path | None = None,
) -> SiteInjectionResult:
    if archive is not None:
        archive.record_output(result, status=status, output_path=output_path)
    return result
