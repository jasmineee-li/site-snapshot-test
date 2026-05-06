from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SUMMARY_NAME = "capture_summary.json"
_ANIMATION_KILLER_JS = """
(() => {
  if (document.getElementById('worldsim-animation-killer')) return;
  const style = document.createElement('style');
  style.id = 'worldsim-animation-killer';
  style.textContent = '* { animation: none !important; transition: none !important; }';
  (document.head || document.documentElement).appendChild(style);
})();
"""
_PVPO_QUERY_TEMPLATE = r"""
(() => {
  const witnesses = __WORLDSIM_WITNESSES_JSON__;
  const scrollToMatch = __WORLDSIM_SCROLL_TO_MATCH__;
  const viewportW = window.innerWidth || document.documentElement.clientWidth || 1280;
  const viewportH = window.innerHeight || document.documentElement.clientHeight || 720;
  const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
  const lower = (s) => norm(s).toLowerCase();
  const resolveBg = () => {
    let el = document.elementFromPoint(Math.max(0, Math.floor(viewportW / 2)), Math.max(0, Math.floor(viewportH / 2))) || document.body || document.documentElement;
    while (el) {
      const color = getComputedStyle(el).backgroundColor || '';
      const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (match && color !== 'rgba(0, 0, 0, 0)') {
        return {r: Number(match[1]), g: Number(match[2]), b: Number(match[3])};
      }
      el = el.parentElement;
    }
    return {r: 255, g: 255, b: 255};
  };
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const text = node.nodeValue || '';
    const normalized = lower(text);
    for (const witness of witnesses) {
      const needle = lower(witness.text);
      if (!needle) continue;
      const normalizedOffset = normalized.indexOf(needle);
      if (normalizedOffset < 0) continue;
      const rawOffset = Math.max(0, text.toLowerCase().indexOf(String(witness.text).toLowerCase()));
      const start = rawOffset >= 0 ? rawOffset : normalizedOffset;
      const end = Math.min(text.length, start + String(witness.text).length);
      const parent = node.parentElement;
      if (scrollToMatch && parent && parent.scrollIntoView) {
        parent.scrollIntoView({block: 'center', inline: 'nearest'});
      }
      const entries = [];
      for (let i = start; i < end; i++) {
        const range = document.createRange();
        range.setStart(node, i);
        range.setEnd(node, Math.min(i + 1, text.length));
        const rect = range.getBoundingClientRect();
        const char = text.slice(i, i + 1);
        const visible = rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.right >= 0 && rect.top <= viewportH && rect.left <= viewportW;
        entries.push({
          char,
          isSpace: /\s/.test(char),
          layoutVisible: Boolean(visible),
          liveRect: {
            x: Math.max(0, Math.floor(rect.left)),
            y: Math.max(0, Math.floor(rect.top)),
            w: Math.max(0, Math.ceil(rect.width)),
            h: Math.max(0, Math.ceil(rect.height))
          }
        });
      }
      return {
        entries,
        backgroundColor: resolveBg(),
        matchFound: true,
        matchOffset: start,
        matchedWitnessId: witness.id || null,
        matchedWitnessText: witness.text || null,
        pageUrl: location.href
      };
    }
  }
  return {
    entries: [],
    backgroundColor: resolveBg(),
    matchFound: false,
    matchOffset: -1,
    matchedWitnessId: null,
    matchedWitnessText: null,
    pageUrl: location.href
  };
})();
"""


@dataclass
class SyncPvpoRecorder:
    output_dir: Path
    payload_text: str = ""
    witness_texts: list[str | dict[str, Any]] | None = None
    repo_root: Path | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.payload_present = bool(self.payload_text or self.witness_texts)
        self.summary: dict[str, Any] = {
            "status": "initialized" if self.payload_present else "disabled_no_payload",
            "payload_present": self.payload_present,
            "steps_seen": 0,
            "steps_captured": 0,
            "issue_steps": 0,
            "first_issue_class": None,
            "first_issue_step": None,
            "first_issue_message": None,
            "last_issue_class": None,
            "last_issue_step": None,
            "last_issue_message": None,
            "issue_counts": {},
            "capture_implementation": "worldsim-agentlab-sync-pvpo-v1",
        }
        self.persist_summary()

    def capture_step(self, page: Any, step_idx: int) -> None:
        self.summary["steps_seen"] += 1
        if not self.payload_present:
            self.persist_summary()
            return
        screenshot_png = b""
        metadata: dict[str, Any] = {
            "step_idx": step_idx,
            "visibility_vec": [],
            "background_color": [255, 255, 255],
            "has_damage": True,
            "page_url": None,
            "issue_class": None,
            "issue_message": None,
            "match_found": False,
            "match_offset": -1,
            "matched_witness_id": None,
            "matched_witness_text": None,
            "screenshot_bytes": 0,
            "clip": self._viewport_clip(page),
        }
        try:
            cdp = page.context.new_cdp_session(page)
            cdp.send("Runtime.evaluate", {"expression": _ANIMATION_KILLER_JS, "awaitPromise": True})
            cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
            try:
                raw = cdp.send(
                    "Runtime.evaluate",
                    {
                        "expression": self._query_js(),
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                )
                metadata.update(_unwrap_runtime_payload(raw))
                frame = cdp.send(
                    "HeadlessExperimental.beginFrame",
                    {
                        "screenshot": {
                            "format": "png",
                            "quality": 100,
                            "clip": _clip_for_cdp(metadata["clip"]),
                        }
                    },
                )
                screenshot_png = base64.b64decode(frame.get("screenshotData") or "")
                metadata["has_damage"] = bool(frame.get("hasDamage", True))
                if not screenshot_png:
                    metadata["issue_class"] = "begin_frame_empty_screenshot"
                    metadata["issue_message"] = "HeadlessExperimental.beginFrame returned no screenshotData"
            finally:
                try:
                    cdp.send("Emulation.setVirtualTimePolicy", {"policy": "advance"})
                except Exception:
                    pass
        except Exception as exc:
            metadata["issue_class"] = "pvpo_capture_error"
            metadata["issue_message"] = f"{type(exc).__name__}: {exc}"
            if _allow_page_screenshot_fallback():
                try:
                    screenshot_png = page.screenshot(type="png")
                    metadata["issue_class"] = "begin_frame_failed_page_screenshot_fallback"
                    metadata["issue_message"] = f"{type(exc).__name__}: {exc}"
                except Exception:
                    screenshot_png = b""

        metadata["screenshot_bytes"] = len(screenshot_png)
        self._write_step(step_idx, metadata, screenshot_png)
        self.summary["steps_captured"] += 1
        if metadata.get("issue_class"):
            self._record_issue(step_idx, metadata)
        self.summary["status"] = "degraded" if self.summary["issue_steps"] else "ok"
        self.persist_summary()

    def persist_summary(self) -> None:
        pvpo_dir = self.output_dir / "pvpo"
        pvpo_dir.mkdir(parents=True, exist_ok=True)
        (pvpo_dir / _SUMMARY_NAME).write_text(
            json.dumps(self.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_step(self, step_idx: int, metadata: dict[str, Any], screenshot_png: bytes) -> None:
        screenshots = self.output_dir / "screenshots"
        pvpo_dir = self.output_dir / "pvpo"
        screenshots.mkdir(parents=True, exist_ok=True)
        pvpo_dir.mkdir(parents=True, exist_ok=True)
        (screenshots / f"step_{step_idx}.png").write_bytes(screenshot_png)
        (pvpo_dir / f"step_{step_idx}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _record_issue(self, step_idx: int, metadata: dict[str, Any]) -> None:
        issue_class = str(metadata.get("issue_class") or "unknown")
        self.summary["issue_steps"] += 1
        if self.summary.get("first_issue_class") is None:
            self.summary["first_issue_class"] = issue_class
            self.summary["first_issue_step"] = step_idx
            self.summary["first_issue_message"] = metadata.get("issue_message")
        self.summary["last_issue_class"] = issue_class
        self.summary["last_issue_step"] = step_idx
        self.summary["last_issue_message"] = metadata.get("issue_message")
        counts = self.summary.setdefault("issue_counts", {})
        counts[issue_class] = int(counts.get(issue_class, 0)) + 1

    def _query_js(self) -> str:
        template = self._root_query_template() or _PVPO_QUERY_TEMPLATE
        witnesses = _normalize_witness_specs(self.payload_text, self.witness_texts)
        return template.replace("__WORLDSIM_WITNESSES_JSON__", json.dumps(witnesses)).replace(
            "__WORLDSIM_SCROLL_TO_MATCH__", "false"
        )

    def _root_query_template(self) -> str | None:
        if self.repo_root is None:
            return None
        path = self.repo_root / "worldsim" / "phase_4" / "pvpo_query.js"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    @staticmethod
    def _viewport_clip(page: Any) -> dict[str, int]:
        viewport = getattr(page, "viewport_size", None) or {}
        width = int(viewport.get("width") or page.evaluate("() => window.innerWidth") or 1280)
        height = int(viewport.get("height") or page.evaluate("() => window.innerHeight") or 720)
        return {"x": 0, "y": 0, "w": width, "h": height}


def _normalize_witness_specs(
    payload_text: str,
    witness_texts: list[str | dict[str, Any]] | None,
) -> list[dict[str, str]]:
    raw_values: list[str | dict[str, Any]] = []
    if witness_texts:
        raw_values.extend(witness_texts)
    if payload_text and not raw_values:
        raw_values.append(payload_text)
    out: list[dict[str, str]] = []
    for idx, value in enumerate(raw_values):
        text = value.get("text") if isinstance(value, dict) else value
        if not isinstance(text, str) or not text:
            continue
        witness_id = value.get("id") if isinstance(value, dict) else None
        out.append({"id": str(witness_id or f"witness:{idx}"), "text": text})
    return out


def _unwrap_runtime_payload(raw: dict[str, Any]) -> dict[str, Any]:
    value = ((raw.get("result") or {}).get("value") or {}) if isinstance(raw, dict) else {}
    if not isinstance(value, dict):
        value = {}
    bg = value.get("backgroundColor") if isinstance(value.get("backgroundColor"), dict) else {}
    return {
        "visibility_vec": value.get("entries") if isinstance(value.get("entries"), list) else [],
        "background_color": [
            int(bg.get("r", 255) or 255),
            int(bg.get("g", 255) or 255),
            int(bg.get("b", 255) or 255),
        ],
        "match_found": bool(value.get("matchFound")),
        "match_offset": _int_or_default(value.get("matchOffset"), -1),
        "matched_witness_id": value.get("matchedWitnessId"),
        "matched_witness_text": value.get("matchedWitnessText"),
        "page_url": value.get("pageUrl"),
    }


def _clip_for_cdp(clip: dict[str, int]) -> dict[str, Any]:
    return {
        "x": float(clip["x"]),
        "y": float(clip["y"]),
        "width": float(clip["w"]),
        "height": float(clip["h"]),
        "scale": 1.0,
    }


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _allow_page_screenshot_fallback() -> bool:
    return os.environ.get("WORLDSIM_AGENTLAB_PVPO_ALLOW_PAGE_SCREENSHOT_FALLBACK", "").lower() in {
        "1",
        "true",
        "yes",
    }
