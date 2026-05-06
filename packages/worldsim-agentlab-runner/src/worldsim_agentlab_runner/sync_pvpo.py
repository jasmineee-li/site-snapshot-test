from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from worldsim_agentlab_runner.sync_cdp import PumpedSyncCdpSession, SyncCdpWorker

logger = logging.getLogger(__name__)

_CDP_VIEWPORT_JS = """
(() => ({
  w: Math.max(0, Number(window.innerWidth || 0)),
  h: Math.max(0, Number(window.innerHeight || 0))
}))()
"""
_PVPO_CDP_TIMEOUT_ENV = "WORLDSIM_PVPO_CDP_TIMEOUT_S"
_PVPO_CDP_TIMEOUT_DEFAULT_S = 10.0


class SyncPvpoRecorder:
    """Synchronous AgentLab bridge to WorldSim's canonical PVPO capture path.

    AgentLab/BrowserGym exposes Playwright's sync API, while Browser Use uses an
    async CDP surface. This bridge owns only the sync-to-async boundary; the
    capture algorithm, query JS, beginFrame retries, metadata shape, and artifact
    writer all come from ``worldsim.phase_4.pvpo_capture``.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        payload_text: str = "",
        witness_texts: list[str | dict[str, Any]] | None = None,
        repo_root: Path | None = None,
        cdp_url: str = "",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.payload_text = payload_text
        self.witness_texts = witness_texts
        self.repo_root = repo_root
        self.cdp_url = cdp_url
        self.payload_present = bool(payload_text or witness_texts)
        self._controller: Any = None
        self._warned_issue_classes: set[str] = set()
        self._pages_prepared: set[str] = set()
        self._worker = SyncCdpWorker(
            timeout_s=_pvpo_cdp_timeout_s(),
            name="worldsim-agentlab-pvpo",
        )

        from worldsim.phase_4.pvpo_capture import initial_capture_summary, save_capture_summary

        self.summary = initial_capture_summary(payload_present=self.payload_present)
        if self.payload_present:
            witnesses = _normalize_payload_witness_specs(witness_texts)
            payload_text_present = isinstance(payload_text, str) and bool(payload_text)
            if witnesses:
                witness_mode = "curated_witnesses"
            elif payload_text_present and witness_texts is not None:
                witness_mode = "payload_text_fallback_empty_witnesses"
            elif payload_text_present:
                witness_mode = "payload_text_fallback"
            else:
                witness_mode = "no_witness"
            self.summary.update(
                {
                    "witness_selection_mode": witness_mode,
                    "payload_witness_count": len(witnesses),
                    "payload_witness_lengths": [len(witness["text"]) for witness in witnesses],
                    "payload_witness_ids": [witness.get("id") for witness in witnesses],
                    "payload_witness_kinds": [witness.get("kind") for witness in witnesses],
                    "payload_text_present": payload_text_present,
                    "payload_text_length": len(payload_text)
                    if isinstance(payload_text, str)
                    else 0,
                }
            )
        save_capture_summary(self.output_dir, self.summary)

    def capture_step(self, page: Any, step_idx: int) -> None:
        self.summary["steps_seen"] += 1
        self._save_summary()
        if not self.payload_present:
            return
        try:
            cdp_session = page.context.new_cdp_session(page)
        except Exception as exc:
            self._record_issue(
                "cdp_session_unavailable",
                step_idx,
                _pvpo_issue_message(exc, timeout_s=_pvpo_cdp_timeout_s()),
            )
            return
        self._worker.run(
            lambda pump: self._capture_step(
                page,
                step_idx,
                cdp_session,
                pump,
            )
        )

    def close(self) -> None:
        self._worker.close()

    async def _capture_step(
        self,
        page: Any,
        step_idx: int,
        cdp_session: Any,
        pump: Any,
    ) -> None:
        from worldsim.phase_4.pvpo_browser_config import inject_animation_killer
        from worldsim.phase_4.pvpo_capture import (
            Rect,
            atomic_capture_with_visibility,
            save_step_artifacts,
        )
        from worldsim.phase_4.pvpo_cdp import runtime_evaluate_value

        timeout_s = _pvpo_cdp_timeout_s()
        cdp_session = _timeout_safe_cdp_session(cdp_session, pump=pump)

        try:
            page_key = str(getattr(page, "url", "") or id(page))
            if page_key not in self._pages_prepared:
                await inject_animation_killer(page, cdp_session)
                self._pages_prepared.add(page_key)
        except Exception as exc:
            self._record_issue(
                "animation_killer_failed",
                step_idx,
                _pvpo_issue_message(exc, timeout_s=timeout_s),
            )

        try:
            viewport = await runtime_evaluate_value(cdp_session, _CDP_VIEWPORT_JS)
            if not isinstance(viewport, dict):
                raise RuntimeError(
                    f"viewport probe returned {type(viewport).__name__}, expected object"
                )
            viewport_rect = Rect(
                x=0,
                y=0,
                w=int(viewport.get("w", 0)) or 1280,
                h=int(viewport.get("h", 0)) or 720,
            )
            capturing = self._capturing_event()
            witnesses = _normalize_payload_witness_specs(self.witness_texts)
            capture = await atomic_capture_with_visibility(
                cdp_session,
                viewport_rect=viewport_rect,
                payload_text=self.payload_text,
                witness_texts=witnesses if self.witness_texts is not None else None,
                scroll_to_match=False,
                capturing=capturing,
                cdp_timeout_s=timeout_s,
            )
            save_step_artifacts(self.output_dir, step_idx, capture)
            if capture.issue_class is not None:
                self._record_issue(
                    capture.issue_class,
                    step_idx,
                    capture.issue_message or capture.issue_class,
                )
            self._record_capture_success(capture.issue_class)
        except Exception as exc:
            self._record_issue(
                "capture_failed", step_idx, _pvpo_issue_message(exc, timeout_s=timeout_s)
            )

    def _capturing_event(self) -> asyncio.Event:
        from worldsim.phase_4.pvpo_beginframe import (
            BeginFrameCoordinator,
            coordinator_for_pvpo_endpoint,
        )

        capturing = asyncio.Event()
        if self._controller is None:
            if self.cdp_url:
                self._controller = coordinator_for_pvpo_endpoint(self.cdp_url)
            else:
                self._controller = BeginFrameCoordinator()
        capturing.beginframe_lock = self._controller.lock  # type: ignore[attr-defined]
        capturing.beginframe_controller = self._controller  # type: ignore[attr-defined]
        return capturing

    def _record_issue(self, issue_class: str, step_idx: int, message: str) -> None:
        self.summary["status"] = "degraded"
        self.summary["issue_steps"] += 1
        issue_counts = self.summary.setdefault("issue_counts", {})
        issue_counts[issue_class] = int(issue_counts.get(issue_class, 0)) + 1
        if self.summary.get("first_issue_class") is None:
            self.summary["first_issue_class"] = issue_class
            self.summary["first_issue_step"] = step_idx
            self.summary["first_issue_message"] = message
        self.summary["last_issue_class"] = issue_class
        self.summary["last_issue_step"] = step_idx
        self.summary["last_issue_message"] = message
        self._save_summary()
        if issue_class not in self._warned_issue_classes:
            self._warned_issue_classes.add(issue_class)
            logger.warning(
                "pvpo: %s at step %d for %s; continuing in degraded mode "
                "(zero coverage may reflect capture failure): %s",
                issue_class,
                step_idx,
                self.output_dir,
                message,
            )
        else:
            logger.debug("pvpo: %s at step %d: %s", issue_class, step_idx, message)

    def _record_capture_success(self, capture_issue_class: str | None) -> None:
        self.summary["steps_captured"] += 1
        if self.summary["issue_steps"] == 0:
            self.summary["status"] = "ok"
        if capture_issue_class is not None and self.summary["status"] != "degraded":
            self.summary["status"] = "degraded"
        self._save_summary()

    def _save_summary(self) -> None:
        from worldsim.phase_4.pvpo_capture import save_capture_summary

        save_capture_summary(self.output_dir, self.summary)


def _normalize_payload_witness_specs(
    payload_witnesses: list[str | dict[str, Any]] | None,
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for index, witness in enumerate(payload_witnesses or []):
        if isinstance(witness, str):
            if witness:
                specs.append({"id": f"witness:{index}", "text": witness})
            continue
        if not isinstance(witness, dict):
            continue
        text = witness.get("text")
        if not isinstance(text, str) or not text:
            continue
        witness_id = witness.get("id")
        kind = witness.get("kind")
        spec = {
            "id": witness_id if isinstance(witness_id, str) and witness_id else f"witness:{index}",
            "text": text,
        }
        if isinstance(kind, str) and kind:
            spec["kind"] = kind
        specs.append(spec)
    return specs


def _pvpo_cdp_timeout_s() -> float:
    raw = os.environ.get(_PVPO_CDP_TIMEOUT_ENV, "").strip()
    if not raw:
        return _PVPO_CDP_TIMEOUT_DEFAULT_S
    try:
        timeout_s = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.1fs",
            _PVPO_CDP_TIMEOUT_ENV,
            raw,
            _PVPO_CDP_TIMEOUT_DEFAULT_S,
        )
        return _PVPO_CDP_TIMEOUT_DEFAULT_S
    if timeout_s <= 0:
        logger.warning(
            "%s=%r is not positive; using %.1fs",
            _PVPO_CDP_TIMEOUT_ENV,
            raw,
            _PVPO_CDP_TIMEOUT_DEFAULT_S,
        )
        return _PVPO_CDP_TIMEOUT_DEFAULT_S
    return timeout_s


def _timeout_safe_cdp_session(cdp_session: Any, *, pump: Any | None = None) -> Any:
    from worldsim.phase_4.pvpo_cdp import normalize_cdp_session

    if pump is not None:
        return PumpedSyncCdpSession(cdp_session, pump)
    return normalize_cdp_session(cdp_session, thread_sync_send=True)


def _pvpo_issue_message(exc: BaseException, *, timeout_s: float) -> str:
    message = str(exc)
    if isinstance(exc, TimeoutError) and not message:
        return f"timed out after {timeout_s:.2f}s"
    return message or type(exc).__name__
