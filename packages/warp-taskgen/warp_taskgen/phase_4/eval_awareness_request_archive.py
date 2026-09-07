"""Observational SDK requests for the default eval-awareness rewrite iterator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.host_api_observability import summarize_provider_response
from warp_taskgen.host_api_request_archive import archive_model_request
from warp_taskgen.task_paths import safe_task_path_component

logger = logging.getLogger(__name__)


@dataclass
class RewriteRequestArchive:
    """One rewrite invocation, including its transport and semantic retries."""

    task_dir_root: Path
    root_task_id: str
    parent_task_id: str
    iteration: int
    repair_ordinal: int = 0
    call_id: str = field(default_factory=lambda: uuid4().hex)
    configured_model: str | None = None
    resolved_client_provider: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return (
            self.task_dir_root
            / safe_task_path_component(self.root_task_id)
            / "eval_awareness_requests"
            / self.call_id
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": "phase_4:eval_awareness_rewrite",
            "path_base": "phase4_task_dir_root",
            "capture_boundary": "model_facing_sdk_arguments",
            "transport": "anthropic_messages_stream",
            "call_id": self.call_id,
            "root_task_id": self.root_task_id,
            "parent_task_id": self.parent_task_id,
            "iteration": self.iteration,
            "repair_ordinal": self.repair_ordinal,
            "configured_model": self.configured_model,
            "resolved_client_provider": self.resolved_client_provider,
        }

    def record_request(self, kwargs: dict[str, Any], *, semantic_attempt: int) -> int:
        requests = self.diagnostics.setdefault("requests", [])
        index = len(requests) + 1
        reference: dict[str, Any] = {
            "request_index": index,
            "semantic_attempt": semantic_attempt,
        }
        try:
            reference.update(
                archive_model_request(
                    kwargs,
                    metadata={**self._metadata(), **reference},
                    path=self.directory / f"{index}.json",
                    run_root=self.task_dir_root,
                    write_json=write_json_atomic,
                )
            )
        except Exception as exc:
            reference.update(status="retention_failed", error_type=type(exc).__name__)
            self._warn(exc)
        requests.append(reference)
        self._persist()
        return index

    def record_response(self, index: int, response: Any) -> None:
        try:
            self.diagnostics.setdefault("responses", []).append(
                {"request_index": index, "response": summarize_provider_response(response)}
            )
        except Exception as exc:
            self._warn(exc)
        self._persist()

    def record_error(self, index: int, error: BaseException) -> None:
        # Provider exception text can contain credentials or full request content.
        self.diagnostics.setdefault("transport_errors", []).append(
            {"request_index": index, "error_type": type(error).__name__}
        )
        self._persist()

    def record_parse(self, index: int, status: str) -> None:
        self.diagnostics.setdefault("parses", []).append({"request_index": index, "status": status})
        self._persist()

    def record_output(self, rewrite: dict[str, Any]) -> dict[str, Any] | None:
        """Join to the iterator's existing durable rewrite slot, without task mutation."""
        if not self.diagnostics.get("requests"):
            return None
        variant_status = rewrite.get("variant_status")
        status = variant_status if isinstance(variant_status, dict) else {}
        self.diagnostics["host_output"] = {
            "status": status.get("status", "ok"),
            "failure_class": status.get("failure_class"),
            "checkpoint_path": (
                Path(safe_task_path_component(self.root_task_id))
                / "eval_awareness_iterator_checkpoint.json"
            ).as_posix(),
            "iteration": self.iteration,
            "repair_ordinal": self.repair_ordinal,
        }
        self._persist()
        return {
            "call_id": self.call_id,
            "path_base": "phase4_task_dir_root",
            "path": (self.directory / "diagnostics.json")
            .relative_to(self.task_dir_root)
            .as_posix(),
            "repair_ordinal": self.repair_ordinal,
            "requests": self.diagnostics["requests"],
            "observer_errors": list(self.diagnostics.get("observer_errors", [])),
        }

    def _persist(self) -> None:
        try:
            write_json_atomic(
                self.directory / "diagnostics.json", {**self._metadata(), **self.diagnostics}
            )
        except Exception as exc:
            self._warn(exc)

    def _warn(self, error: Exception) -> None:
        self.diagnostics.setdefault("observer_errors", []).append(type(error).__name__)
        logger.warning(
            "Phase 4 rewrite request retention failed (%s), call=%s",
            type(error).__name__,
            self.call_id,
        )
