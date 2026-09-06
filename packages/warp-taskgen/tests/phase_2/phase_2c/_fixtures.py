"""Shared fakes, builders, and autouse fixtures for the Phase 2c feasibility tests.

Moved verbatim from the former single-file Phase 2c feasibility suite. Uses the
``_FakeSession``/``_FakeResponse`` monkeypatch pattern from
``tests/test_seeding.py`` — the repo does not depend on ``respx``,
``aresponses``, or ``tenacity`` and these tests do not introduce them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from warp_taskgen.editors import EditorError
from warp_taskgen.phase_2.phase_2c.probe_bundle import Phase2cProbeBundle
from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeHandle:
    def __init__(self, *, raises: bool = False) -> None:
        self.cleaned = False
        self._raises = raises

    def cleanup(self) -> None:
        self.cleaned = True
        if self._raises:
            raise EditorError("cleanup_failed", "synthetic cleanup failure")


def _gitlab_instance(**extras: Any) -> dict[str, Any]:
    payload = {
        "site_name": "gitlab",
        "site_url": "https://gitlab.example/",
        "benchmark": "webarena_verified",
    }
    payload.update(extras)
    return payload


def _shopping_instance(**extras: Any) -> dict[str, Any]:
    payload = {
        "site_name": "shopping",
        "site_url": "https://shop.example/",
        "benchmark": "webarena_verified",
    }
    payload.update(extras)
    return payload


def _task(
    task_id: str = "AT-001",
    site: str = "gitlab",
    *,
    method: str = "create_group",
    detail: str = "hello",
    feasibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "benchmark": "webarena_verified",
        "site": site,
        "benign_task_id": f"benign-{task_id}",
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": site,
                    "method": method,
                    "args": {"name": f"{task_id}-resource", "description": detail},
                }
            ],
        },
    }
    if feasibility is not None:
        task["feasibility"] = feasibility
    return task


@pytest.fixture(autouse=True)
def _stable_git_fingerprint(monkeypatch):
    """Deterministic editor/dataset commit for fingerprint matching tests."""
    monkeypatch.setenv("WORLDSIM_EDITOR_COMMIT_OVERRIDE", "cafebabe1234")
    yield


class _StubEditorCls:
    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        return None


# Per-run Site editor snapshot with a no-op ``probe_base_state``. Tests pass
# it as ``seed_registry=`` so the runner never consults the process-wide
# editor registry.
_STUB_SEED_REGISTRY = SeedSiteRegistry.from_registrations(
    tuple(
        SeedSiteRegistration("webarena_verified", site, _StubEditorCls)
        for site in ("gitlab", "shopping", "reddit")
    )
)


@pytest.fixture(autouse=True)
def _bypass_preflight(monkeypatch):
    """Disable render verification by default.

    These tests mock the seed flow through a ``Phase2cProbeBundle`` and never
    run a real browser; tests for the render check itself live in
    ``tests/test_phase_2_render_check.py``. Token acquisition is a no-op in
    :func:`_bundle`; tests that care about it pass their own ``acquire_tokens``.
    """
    monkeypatch.setenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", "1")
    yield


def _bundle(**overrides: Any) -> Phase2cProbeBundle:
    """Real Phase 2c collaborators with token acquisition stubbed out."""
    fields: dict[str, Any] = {"acquire_tokens": lambda instances: []}
    fields.update(overrides)
    return Phase2cProbeBundle.default(**fields)


def _write_tasks(tmp_path: Path, tasks: list[dict[str, Any]]) -> Path:
    target = tmp_path / "adversarial_tasks.json"
    target.write_text(json.dumps(tasks))
    return target


def _seed_bundle(responder, **overrides: Any) -> Phase2cProbeBundle:
    """Bundle whose ``apply_seed`` calls ``responder(attempt_index, seed, instance)``.

    ``responder`` may return a fake handle, raise ``EditorError``, raise
    ``ValueError``, or return ``None`` (the "empty_seed" path). The wrapper
    auto-tuples bare responder returns so tests don't need to track the
    Commit-2-of-C1-migration tuple shape ``(handle, metadata)``.
    """
    counter = {"n": 0}

    async def fake(seed, instance, **kwargs):
        idx = counter["n"]
        counter["n"] += 1
        result = responder(idx, seed, instance)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    return _bundle(apply_seed=fake, **overrides)


# ---------------------------------------------------------------------------
# Render-check wiring (Layer 2 of the 2026-04-21 long-term fix)
# ---------------------------------------------------------------------------


class _FakePlaywrightPage:
    def __init__(self, body: str = "", layout_probe: dict[str, Any] | None = None) -> None:
        self.body = body
        self.layout_probe = layout_probe

    async def goto(self, url, *, timeout, wait_until):
        return None

    async def text_content(self, selector):
        return self.body

    async def wait_for_selector(self, selector, *, timeout):
        return None

    async def wait_for_timeout(self, ms):
        # Bug J: body-text poll sleeps via this; no-op keeps tests fast.
        return None

    async def evaluate(self, script, arg=None):
        return self.layout_probe

    async def route(self, pattern, handler):
        # Bug K: tests predate the page.route blocker; accept + no-op so
        # verify_seed_renders can install the handler without raising.
        return None

    def wait_for_response(self, predicate, *, timeout):
        async def _await_noop():
            return None

        return _await_noop()


class _FakePlaywrightContext:
    def __init__(self, page: _FakePlaywrightPage) -> None:
        self._page = page
        self.closed = False

    async def new_page(self):
        return self._page

    async def close(self):
        self.closed = True


class _FakePlaywrightBrowser:
    def __init__(self, page: _FakePlaywrightPage) -> None:
        self._page = page
        self.closed = False

    async def new_context(self):
        return _FakePlaywrightContext(self._page)

    async def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakePlaywrightBrowser, executable_path: str | None = None) -> None:
        self._browser = browser
        self.executable_path = executable_path or sys.executable
        self.last_launch_kwargs: dict[str, Any] = {}

    async def launch(self, *, headless, args=None):
        self.last_launch_kwargs = {"headless": headless, "args": args}
        return self._browser


class _FakePlaywright:
    def __init__(self, browser: _FakePlaywrightBrowser, executable_path: str | None = None) -> None:
        self.chromium = _FakeChromium(browser, executable_path=executable_path)
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _FakePlaywrightFactory:
    def __init__(self, browser: _FakePlaywrightBrowser, executable_path: str | None = None) -> None:
        self._browser = browser
        self._executable_path = executable_path

    def __call__(self):
        return self

    async def start(self):
        return _FakePlaywright(self._browser, executable_path=self._executable_path)


def _shopping_review_task(
    *, task_id: str = "RENDER-001", nickname: str = "SeedNickAdv003"
) -> dict[str, Any]:
    return {
        "id": task_id,
        "site": "shopping",
        "benign_task_id": f"benign-{task_id}",
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "method": "create_product_review",
                    "args": {"nickname": nickname, "detail": "x", "entity_pk_value": 67},
                }
            ],
        },
    }


def _metadata_bundle(urls: list[str], **overrides: Any) -> Phase2cProbeBundle:
    handle = _FakeHandle()

    async def fake(seed, instance, **kwargs):
        return handle, {"read_surface_urls": urls}

    return _bundle(apply_seed=fake, **overrides)


def _install_fake_playwright(
    monkeypatch, browser: _FakePlaywrightBrowser, *, executable_path: str | None = None
) -> None:
    """Inject a fake ``playwright.async_api`` module into ``sys.modules`` so
    the lazy ``from playwright.async_api import async_playwright`` inside
    ``verify_feasibility`` resolves to our shim. Works whether or not the
    real Playwright package is installed in the dev environment."""
    import sys
    import types

    factory = _FakePlaywrightFactory(browser, executable_path=executable_path)
    fake_module = types.ModuleType("playwright.async_api")
    fake_module.async_playwright = factory
    fake_pkg = types.ModuleType("playwright")
    monkeypatch.setitem(sys.modules, "playwright", fake_pkg)
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_module)


__all__ = [
    "_STUB_SEED_REGISTRY",
    "_FakeChromium",
    "_FakeHandle",
    "_FakePlaywright",
    "_FakePlaywrightBrowser",
    "_FakePlaywrightContext",
    "_FakePlaywrightFactory",
    "_FakePlaywrightPage",
    "_StubEditorCls",
    "_bundle",
    "_bypass_preflight",
    "_gitlab_instance",
    "_install_fake_playwright",
    "_metadata_bundle",
    "_seed_bundle",
    "_shopping_instance",
    "_shopping_review_task",
    "_stable_git_fingerprint",
    "_task",
    "_write_tasks",
]
