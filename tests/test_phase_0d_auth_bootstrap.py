"""Tests for Phase 0d auth artifact bootstrapping.

Covers:
- Successful generator invocation writes ``storage_state.json`` + completion marker
- Idempotent skip when input hash matches existing completion
- Refresh when the generator_script source bytes change (hash miss)
- Refresh when credentials change
- Missing ``generate`` callable raises ``AuthBootstrapError`` (caught, reported in state)
- Invalid ``generate`` signature raises ``AuthBootstrapError``
- No ``generator_script`` declared + missing artifact -> graceful skip
- Sites with non-storage_state auth_mechanism are ignored
- CLI dispatcher wiring: ``worldsim phase 0d`` reaches the module's ``run()``
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest

from worldsim.phases import phase_0d_auth_bootstrap as bootstrap

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_context(
    profiles_dir: Path,
    site: str,
    *,
    auth_mechanism: dict | None,
    credentials: object = None,
) -> Path:
    """Write a flat AGENT_CONTEXT_<site>.json mirroring Phase 0c output."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    ctx = {
        "response_format": {
            "requires_structured_output": False,
            "output_schema": None,
            "per_task_format_field": None,
            "description": "test",
        },
        "authentication": {
            "pre_authenticated": True,
            "credentials": credentials,
            "description": "test",
        },
        "agent_prompt_template": None,
        "site_context": {"platform_name": "Test", "description": "Test."},
    }
    if auth_mechanism is not None:
        ctx["auth_mechanism"] = auth_mechanism
    path = profiles_dir / f"AGENT_CONTEXT_{site}.json"
    path.write_text(json.dumps(ctx), encoding="utf-8")
    return path


def _write_generator(
    benchmark_root: Path,
    rel_path: str,
    body: str,
) -> Path:
    """Write a generator_script at ``benchmark_root / rel_path`` and return the full path."""
    full = benchmark_root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body, encoding="utf-8")
    return full


def _setup_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """Create state_dir, benchmark_root, profiles_dir and point worldsim at them."""
    state_dir = tmp_path / "logs"
    bench = tmp_path / "benchmark"
    profiles = state_dir / "phase_0c"
    state_dir.mkdir()
    bench.mkdir()
    profiles.mkdir(parents=True)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    return state_dir, bench, profiles


_GOOD_GENERATOR = """
def generate(credentials, site_url, output_path, **kwargs):
    import json
    output_path.write_text(json.dumps({"cookies": [], "origins": [], "test_marker": "v1"}))
"""


_MISSING_GENERATE = """
# no 'generate' callable at top level
def setup(*args, **kwargs):
    pass
"""


_BAD_SIGNATURE = """
def generate(only_one_param):
    pass
"""


_POSITIONAL_ONLY_SIGNATURE = """
def generate(credentials, /, site_url, output_path):
    pass
"""


_EMPTY_WRITE = """
def generate(credentials, site_url, output_path, **kwargs):
    output_path.write_text("")  # empty file — orchestrator must reject
"""


_ASYNC_GENERATOR = """
async def generate(credentials, site_url, output_path, **kwargs):
    import json
    output_path.write_text(json.dumps({"cookies": [], "via": "async"}))
"""


def _make_args(benchmark: Path, instances: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(benchmark=benchmark, instances=instances)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_sync_generator_writes_artifact(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/make_auth.py", _GOOD_GENERATOR)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/make_auth.py",
                    "per_task_refresh": False,
                },
            },
            credentials={"username": "u", "password": "p"},
        )

        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0

        artifact = state_dir / "phase_0d" / "shopping" / "storage_state.json"
        completion = state_dir / "phase_0d" / "shopping" / "completion.json"
        assert artifact.exists()
        assert completion.exists()
        data = json.loads(artifact.read_text())
        assert data["test_marker"] == "v1"
        marker = json.loads(completion.read_text())
        assert marker["site"] == "shopping"
        assert marker["generator_script"] == "scripts/make_auth.py"
        assert isinstance(marker["input_hash"], str) and len(marker["input_hash"]) == 64

    def test_async_generator_is_awaited(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/async_auth.py", _ASYNC_GENERATOR)
        _write_context(
            profiles,
            "reddit",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/reddit_state.json",
                    "generator_script": "scripts/async_auth.py",
                },
            },
        )
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        data = json.loads((state_dir / "phase_0d" / "reddit" / "storage_state.json").read_text())
        assert data["via"] == "async"

    def test_run_persists_benchmark_and_instances_paths_for_resume(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/make_auth.py", _GOOD_GENERATOR)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/make_auth.py",
                },
            },
        )
        instances_path = tmp_path / "instances.json"
        instances_path.write_text(
            json.dumps(
                {
                    "benchmark_name": "demo",
                    "benchmark_codebase": str(bench),
                    "instances": [
                        {
                            "site_name": "shopping",
                            "site_url": "http://shopping.test",
                        }
                    ],
                }
            )
        )

        rc = asyncio.run(bootstrap.run(_make_args(bench, instances_path)))

        assert rc == 0
        state = json.loads((state_dir / "pipeline_state.json").read_text())
        assert state["benchmark_path"] == str(bench)
        assert state["instances_path"] == str(instances_path)

    def test_no_sites_with_storage_state_is_noop(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_context(
            profiles,
            "public",
            auth_mechanism={"type": "none", "notes": "public site"},
        )
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        assert not (state_dir / "phase_0d" / "public").exists()


# ---------------------------------------------------------------------------
# Idempotency + content-hash invalidation
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_skip_when_input_hash_matches(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/make_auth.py", _GOOD_GENERATOR)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/make_auth.py",
                },
            },
            credentials={"username": "u", "password": "p"},
        )

        # First run — generates.
        asyncio.run(bootstrap.run(_make_args(bench)))
        artifact = state_dir / "phase_0d" / "shopping" / "storage_state.json"
        first_mtime = artifact.stat().st_mtime_ns

        # Second run — should skip, not touch the file.
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        assert artifact.stat().st_mtime_ns == first_mtime

    def test_refresh_when_generator_source_changes(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/make_auth.py", _GOOD_GENERATOR)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/make_auth.py",
                },
            },
            credentials={"username": "u", "password": "p"},
        )

        asyncio.run(bootstrap.run(_make_args(bench)))
        completion_path = state_dir / "phase_0d" / "shopping" / "completion.json"
        first_hash = json.loads(completion_path.read_text())["input_hash"]

        # Mutate the script; hash must change -> regenerate.
        new_body = _GOOD_GENERATOR.replace("v1", "v2")
        _write_generator(bench, "scripts/make_auth.py", new_body)

        asyncio.run(bootstrap.run(_make_args(bench)))
        second_hash = json.loads(completion_path.read_text())["input_hash"]
        assert first_hash != second_hash, "input hash must change when generator bytes change"

        data = json.loads((state_dir / "phase_0d" / "shopping" / "storage_state.json").read_text())
        assert data["test_marker"] == "v2"

    def test_refresh_when_credentials_change(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/make_auth.py", _GOOD_GENERATOR)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/make_auth.py",
                },
            },
            credentials={"username": "u", "password": "p"},
        )
        asyncio.run(bootstrap.run(_make_args(bench)))
        first_hash = json.loads(
            (state_dir / "phase_0d" / "shopping" / "completion.json").read_text()
        )["input_hash"]

        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/make_auth.py",
                },
            },
            credentials={"username": "u", "password": "different"},
        )
        asyncio.run(bootstrap.run(_make_args(bench)))
        second_hash = json.loads(
            (state_dir / "phase_0d" / "shopping" / "completion.json").read_text()
        )["input_hash"]
        assert first_hash != second_hash

    def test_refresh_when_site_url_changes(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/make_auth.py", _GOOD_GENERATOR)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/make_auth.py",
                },
            },
        )
        instances_path = tmp_path / "instances.json"
        instances_path.write_text(
            json.dumps(
                {
                    "benchmark_name": "demo",
                    "benchmark_codebase": str(bench),
                    "instances": [
                        {
                            "site_name": "shopping",
                            "site_url": "http://shopping-v1.test",
                        }
                    ],
                }
            )
        )

        asyncio.run(bootstrap.run(_make_args(bench, instances_path)))
        completion_path = state_dir / "phase_0d" / "shopping" / "completion.json"
        first = json.loads(completion_path.read_text())

        instances_path.write_text(
            json.dumps(
                {
                    "benchmark_name": "demo",
                    "benchmark_codebase": str(bench),
                    "instances": [
                        {
                            "site_name": "shopping",
                            "site_url": "http://shopping-v2.test",
                        }
                    ],
                }
            )
        )

        asyncio.run(bootstrap.run(_make_args(bench, instances_path)))
        second = json.loads(completion_path.read_text())

        assert first["input_hash"] != second["input_hash"]
        assert second["site_url"] == "http://shopping-v2.test"


# ---------------------------------------------------------------------------
# Contract violations + generator failures
# ---------------------------------------------------------------------------


class TestContractFailures:
    def test_generator_script_path_missing_is_failure(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/does_not_exist.py",
                },
            },
        )
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 1  # failure recorded

    def test_generator_without_generate_callable_fails(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/no_generate.py", _MISSING_GENERATE)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/no_generate.py",
                },
            },
        )
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 1

    def test_generator_bad_signature_fails(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/bad_sig.py", _BAD_SIGNATURE)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/bad_sig.py",
                },
            },
        )
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 1

    def test_generator_empty_output_fails(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_generator(bench, "scripts/empty.py", _EMPTY_WRITE)
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "scripts/empty.py",
                },
            },
        )
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 1

    def test_missing_generator_and_missing_artifact_is_graceful_skip(self, tmp_path, monkeypatch):
        """No generator_script + no artifact -> graceful skip (Phase 3 runtime enforces)."""
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_context(
            profiles,
            "gitlab",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/gitlab_state.json",
                    # no generator_script — operator will stage it by hand
                },
            },
        )
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        # Skipped, not failed.
        assert rc == 0
        assert not (state_dir / "phase_0d" / "gitlab" / "storage_state.json").exists()

    def test_relative_generator_script_escape_is_rejected(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        escaped = tmp_path / "escape.py"
        escaped.write_text(_GOOD_GENERATOR, encoding="utf-8")
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": "../escape.py",
                },
            },
        )

        rc = asyncio.run(bootstrap.run(_make_args(bench)))

        assert rc == 1
        assert not (state_dir / "phase_0d" / "shopping" / "storage_state.json").exists()

    def test_relative_trust_path_escape_is_rejected(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        escaped = tmp_path / "state.json"
        escaped.write_text(json.dumps({"cookies": [{"name": "session"}]}), encoding="utf-8")
        _write_context(
            profiles,
            "shopping",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {
                    "path": "../state.json",
                },
            },
        )

        rc = asyncio.run(bootstrap.run(_make_args(bench)))

        assert rc == 1
        assert not (state_dir / "phase_0d" / "shopping" / "storage_state.json").exists()


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCliDispatch:
    def test_cli_parses_phase_0d(self):
        from worldsim.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["phase", "0d", "--benchmark", "/tmp/bench"])
        assert args.phase == "0d"
        assert args.benchmark == Path("/tmp/bench")

    def test_cli_requires_benchmark_for_phase_0d(self, tmp_path, monkeypatch, capsys):
        from worldsim.main import _dispatch_phase

        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        args = argparse.Namespace(
            phase="0d",
            benchmark=None,
            command="phase",
            resume=False,
            config=None,
            instances=None,
            agent_model=None,
            agent_provider=None,
            generate_novel=False,
            full_baseline=False,
            max_tasks_per_site=None,
            allow_unknown_auth=False,
        )
        rc = _dispatch_phase(args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "0d" in err and "benchmark" in err.lower()


# ---------------------------------------------------------------------------
# Built-in form_login bootstrap
# ---------------------------------------------------------------------------


class _FakePlaywrightHarness:
    """Mock harness that mimics ``playwright.async_api.async_playwright``.

    Records every call the bootstrap makes so tests can assert the full
    interaction surface (navigate -> fill -> fill -> click -> wait_for_url ->
    storage_state) without requiring a real browser install.

    ``success_url`` is returned from ``page.url`` during ``wait_for_url``; set
    it to a URL that does *not* contain ``success_url_substring`` to simulate
    a login-timeout failure.
    """

    def __init__(
        self,
        *,
        success_url: str = "https://app.test/dashboard",
        storage_state_payload: dict | None = None,
        goto_raises: Exception | None = None,
        wait_raises: Exception | None = None,
    ) -> None:
        self.success_url = success_url
        self.storage_state_payload = storage_state_payload or {
            "cookies": [{"name": "session", "value": "abc"}],
            "origins": [],
        }
        self.goto_raises = goto_raises
        self.wait_raises = wait_raises
        self.calls: list[tuple] = []

    def __call__(self):
        return self

    async def __aenter__(self):
        return _FakePlaywright(self)

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakePlaywright:
    def __init__(self, harness: _FakePlaywrightHarness) -> None:
        self.harness = harness
        self.chromium = _FakeChromium(harness)


class _FakeChromium:
    def __init__(self, harness: _FakePlaywrightHarness) -> None:
        self.harness = harness

    async def launch(self, *, headless: bool = True):
        self.harness.calls.append(("launch", {"headless": headless}))
        return _FakeBrowser(self.harness)


class _FakeBrowser:
    def __init__(self, harness: _FakePlaywrightHarness) -> None:
        self.harness = harness

    async def new_context(self):
        self.harness.calls.append(("new_context", {}))
        return _FakeContext(self.harness)

    async def close(self):
        self.harness.calls.append(("browser.close", {}))


class _FakeContext:
    def __init__(self, harness: _FakePlaywrightHarness) -> None:
        self.harness = harness

    async def new_page(self):
        self.harness.calls.append(("new_page", {}))
        return _FakePage(self.harness)

    async def storage_state(self, *, path: str):
        self.harness.calls.append(("storage_state", {"path": path}))
        Path(path).write_text(json.dumps(self.harness.storage_state_payload))

    async def close(self):
        self.harness.calls.append(("context.close", {}))


class _FakePage:
    def __init__(self, harness: _FakePlaywrightHarness) -> None:
        self.harness = harness
        self.url = ""

    async def goto(self, url, *, timeout):
        self.harness.calls.append(("goto", {"url": url, "timeout": timeout}))
        if self.harness.goto_raises is not None:
            raise self.harness.goto_raises
        self.url = url

    async def fill(self, selector, value, *, timeout):
        self.harness.calls.append(
            ("fill", {"selector": selector, "value": value, "timeout": timeout})
        )

    async def click(self, selector, *, timeout):
        self.harness.calls.append(("click", {"selector": selector, "timeout": timeout}))

    async def wait_for_url(self, predicate, *, timeout):
        self.harness.calls.append(("wait_for_url", {"timeout": timeout}))
        if self.harness.wait_raises is not None:
            raise self.harness.wait_raises
        self.url = self.harness.success_url
        if callable(predicate) and not predicate(self.url):
            raise TimeoutError(
                f"predicate rejected url={self.url!r} (simulated wait_for_url timeout)"
            )


_MISSING = object()


def _install_fake_playwright(monkeypatch, harness) -> None:
    """Monkeypatch ``_bootstrap_via_form_login`` to inject a mock Playwright factory.

    Phase 0d's production code path lazily imports ``playwright.async_api``;
    tests must not require the real package. We wrap the real function so the
    full bootstrap flow still runs (credential checks, URL normalization,
    post-write JSON validation) but the browser calls go through ``harness``.
    """
    original = bootstrap._bootstrap_via_form_login

    async def _patched(*, spec, site_url, output_path, timeout_ms=30_000, playwright_factory=None):
        return await original(
            spec=spec,
            site_url=site_url,
            output_path=output_path,
            timeout_ms=timeout_ms,
            playwright_factory=harness,
        )

    monkeypatch.setattr(bootstrap, "_bootstrap_via_form_login", _patched)


def _write_form_login_site(
    profiles_dir: Path,
    site: str,
    *,
    mech_type: str = "storage_state",
    nested_recipe: bool = True,
    credentials: object = _MISSING,
    declared_path: str = "auth/gitlab_state.json",
    success_key: str = "success_url_substring",
) -> Path:
    """Write an AGENT_CONTEXT_<site>.json declaring a form_login recipe.

    When ``mech_type == "storage_state"`` and ``nested_recipe`` is True, the
    recipe lives under ``storage_state.form_login``. When ``mech_type ==
    "form_login"``, the recipe lives at the top level.
    """
    recipe = {
        "login_url": "https://app.test/login",
        "username_selector": "input[name='user']",
        "password_selector": "input[name='pw']",
        "submit_selector": "button[type='submit']",
        success_key: "/dashboard",
    }
    if mech_type == "storage_state":
        if nested_recipe:
            mech = {
                "type": "storage_state",
                "storage_state": {
                    "path": declared_path,
                    "form_login": recipe,
                },
            }
        else:
            mech = {
                "type": "storage_state",
                "storage_state": {"path": declared_path},
            }
    else:
        mech = {"type": "form_login", "form_login": recipe}
    creds = (
        credentials if credentials is not _MISSING else {"username": "alice", "password": "hunter2"}
    )
    return _write_context(profiles_dir, site, auth_mechanism=mech, credentials=creds)


class TestFormLoginBootstrap:
    """Phase 0d's built-in form_login bootstrapper."""

    def test_form_login_nested_under_storage_state_writes_artifact(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_form_login_site(profiles, "gitlab", mech_type="storage_state", nested_recipe=True)

        harness = _FakePlaywrightHarness()
        _install_fake_playwright(monkeypatch, harness)

        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0, f"expected success, got rc={rc}; calls={harness.calls}"

        artifact = state_dir / "phase_0d" / "gitlab" / "storage_state.json"
        completion = state_dir / "phase_0d" / "gitlab" / "completion.json"
        assert artifact.exists(), "bootstrap must write storage_state.json"
        assert completion.exists()
        payload = json.loads(artifact.read_text())
        assert payload["cookies"][0]["name"] == "session"
        marker = json.loads(completion.read_text())
        assert marker["dispatch"] == "form_login"
        assert marker["form_login"]["login_url"] == "https://app.test/login"
        assert marker["form_login"]["success_url_substring"] == "/dashboard"

        # Verify the fake harness observed the expected call sequence.
        call_names = [c[0] for c in harness.calls]
        assert call_names == [
            "launch",
            "new_context",
            "new_page",
            "goto",
            "fill",
            "fill",
            "click",
            "wait_for_url",
            "storage_state",
            "context.close",
            "browser.close",
        ], f"unexpected call trace: {call_names}"

    def test_form_login_top_level_type_also_bootstraps(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_form_login_site(profiles, "partner", mech_type="form_login")

        harness = _FakePlaywrightHarness()
        _install_fake_playwright(monkeypatch, harness)

        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        assert (state_dir / "phase_0d" / "partner" / "storage_state.json").exists()

    def test_form_login_accepts_legacy_success_substring_alias(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_form_login_site(
            profiles,
            "gitlab",
            mech_type="storage_state",
            nested_recipe=True,
            success_key="success_substring",  # legacy alias
        )

        harness = _FakePlaywrightHarness()
        _install_fake_playwright(monkeypatch, harness)

        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        assert (state_dir / "phase_0d" / "gitlab" / "storage_state.json").exists()

    def test_form_login_idempotent_on_second_run(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_form_login_site(profiles, "gitlab")

        harness = _FakePlaywrightHarness()
        _install_fake_playwright(monkeypatch, harness)

        # First run: fills the harness with calls.
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        artifact = state_dir / "phase_0d" / "gitlab" / "storage_state.json"
        first_mtime = artifact.stat().st_mtime_ns
        first_call_count = len(harness.calls)
        assert first_call_count > 0

        # Second run: must skip (input hash matches) — zero new Playwright calls.
        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        assert artifact.stat().st_mtime_ns == first_mtime
        assert len(harness.calls) == first_call_count, (
            "second run must not invoke Playwright when input hash matches"
        )

    def test_form_login_credentials_change_triggers_regeneration(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_form_login_site(profiles, "gitlab")

        harness = _FakePlaywrightHarness()
        _install_fake_playwright(monkeypatch, harness)

        asyncio.run(bootstrap.run(_make_args(bench)))
        first_hash = json.loads(
            (state_dir / "phase_0d" / "gitlab" / "completion.json").read_text()
        )["input_hash"]

        # Rotate credentials — must regenerate.
        _write_form_login_site(
            profiles, "gitlab", credentials={"username": "alice", "password": "NEW"}
        )
        asyncio.run(bootstrap.run(_make_args(bench)))
        second_hash = json.loads(
            (state_dir / "phase_0d" / "gitlab" / "completion.json").read_text()
        )["input_hash"]
        assert first_hash != second_hash

    def test_form_login_timeout_reports_failure(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_form_login_site(profiles, "gitlab")

        # Simulate a wait_for_url that never sees success_url_substring.
        harness = _FakePlaywrightHarness(
            wait_raises=TimeoutError("success_url_substring never appeared")
        )
        _install_fake_playwright(monkeypatch, harness)

        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 1  # failure recorded
        assert not (state_dir / "phase_0d" / "gitlab" / "storage_state.json").exists()

    def test_form_login_missing_credentials_is_hard_failure(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        _write_form_login_site(profiles, "gitlab", credentials=None)

        harness = _FakePlaywrightHarness()
        _install_fake_playwright(monkeypatch, harness)

        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 1


# ---------------------------------------------------------------------------
# Dispatch precedence
# ---------------------------------------------------------------------------


class TestDispatchPrecedence:
    """Verify generator_script > form_login > trust_path > skip."""

    def _make_spec(
        self,
        *,
        mech_type: str = "storage_state",
        generator_script: str | None = None,
        form_login: dict | None = None,
        declared_path: str = "",
    ):
        return bootstrap._SiteSpec(
            site_name="site",
            mech_type=mech_type,
            declared_path=declared_path,
            generator_script=generator_script,
            form_login=form_login,
            per_task_refresh=False,
            credentials={"username": "u", "password": "p"},
            agent_context_source=Path("/tmp/fake.json"),
        )

    def test_generator_script_wins_over_form_login(self, tmp_path):
        spec = self._make_spec(
            generator_script="scripts/x.py",
            form_login={
                "login_url": "http://x/login",
                "username_selector": "#u",
                "password_selector": "#p",
                "submit_selector": "#s",
                "success_url_substring": "/ok",
            },
        )
        assert bootstrap._choose_dispatch(spec, benchmark_root=tmp_path) == "generator_script"

    def test_form_login_wins_over_trust_path(self, tmp_path):
        # Declared path exists AND form_login recipe present -> form_login wins.
        declared = tmp_path / "auth" / "state.json"
        declared.parent.mkdir(parents=True)
        declared.write_text(json.dumps({"cookies": []}))
        spec = self._make_spec(
            declared_path="auth/state.json",
            form_login={
                "login_url": "http://x/login",
                "username_selector": "#u",
                "password_selector": "#p",
                "submit_selector": "#s",
                "success_url_substring": "/ok",
            },
        )
        assert bootstrap._choose_dispatch(spec, benchmark_root=tmp_path) == "form_login"

    def test_trust_path_when_only_artifact_present(self, tmp_path):
        declared = tmp_path / "auth" / "state.json"
        declared.parent.mkdir(parents=True)
        declared.write_text(json.dumps({"cookies": [], "via": "prestaged"}))
        spec = self._make_spec(declared_path="auth/state.json")
        assert bootstrap._choose_dispatch(spec, benchmark_root=tmp_path) == "trust_path"

    def test_skip_when_nothing_actionable(self, tmp_path):
        spec = self._make_spec(declared_path="auth/does_not_exist.json")
        assert bootstrap._choose_dispatch(spec, benchmark_root=tmp_path) == "skip"

    def test_trust_path_copies_declared_into_logs(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        pre_staged = bench / "auth" / "state.json"
        pre_staged.parent.mkdir(parents=True)
        pre_staged.write_text(json.dumps({"cookies": [], "via": "prestaged"}))
        _write_context(
            profiles,
            "prestaged_site",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {"path": "auth/state.json"},
            },
        )

        rc = asyncio.run(bootstrap.run(_make_args(bench)))
        assert rc == 0
        dst = state_dir / "phase_0d" / "prestaged_site" / "storage_state.json"
        assert dst.exists()
        payload = json.loads(dst.read_text())
        assert payload["via"] == "prestaged"
        marker = json.loads(
            (state_dir / "phase_0d" / "prestaged_site" / "completion.json").read_text()
        )
        assert marker["dispatch"] == "trust_path"

    def test_trust_path_auth_drift_invalidates_idempotent_skip(self, tmp_path, monkeypatch):
        state_dir, bench, profiles = _setup_dirs(tmp_path, monkeypatch)
        pre_staged = bench / "auth" / "state.json"
        pre_staged.parent.mkdir(parents=True)
        pre_staged.write_text(json.dumps({"cookies": [], "via": "v1"}))
        _write_context(
            profiles,
            "prestaged_site",
            auth_mechanism={
                "type": "storage_state",
                "storage_state": {"path": "auth/state.json"},
            },
        )

        assert asyncio.run(bootstrap.run(_make_args(bench))) == 0
        artifact = state_dir / "phase_0d" / "prestaged_site" / "storage_state.json"
        assert json.loads(artifact.read_text())["via"] == "v1"

        pre_staged.write_text(json.dumps({"cookies": [], "via": "v2"}))

        assert asyncio.run(bootstrap.run(_make_args(bench))) == 0
        assert json.loads(artifact.read_text())["via"] == "v2"


# ---------------------------------------------------------------------------
# Validator — schema consistency for form_login
# ---------------------------------------------------------------------------


class TestValidatorFormLogin:
    def test_validator_accepts_nested_form_login_under_storage_state(self):
        from worldsim._sandbox_validator import validate_agent_context

        data = {
            "response_format": {
                "requires_structured_output": False,
                "output_schema": None,
                "per_task_format_field": None,
                "description": "test",
            },
            "authentication": {
                "pre_authenticated": False,
                "credentials": {"username": "u", "password": "p"},
                "description": "form login",
            },
            "auth_mechanism": {
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/state.json",
                    "form_login": {
                        "login_url": "https://x/login",
                        "username_selector": "#u",
                        "password_selector": "#p",
                        "submit_selector": "#s",
                        "success_url_substring": "/dashboard",
                    },
                },
            },
            "agent_prompt_template": None,
            "site_context": {"platform_name": "X", "description": "X."},
        }
        errors = validate_agent_context(data, site_name="gitlab")
        assert errors == [], f"unexpected errors: {errors}"


def test_verify_generate_signature_rejects_positional_only_required_params(tmp_path):
    script = tmp_path / "bad_generator.py"
    script.write_text(_POSITIONAL_ONLY_SIGNATURE, encoding="utf-8")

    module = bootstrap._load_module(script, "shopping")

    with pytest.raises(bootstrap.AuthBootstrapError, match="positional-only"):
        bootstrap._verify_generate_signature(module.generate, str(script))

    def test_validator_rejects_form_login_without_credentials(self):
        from worldsim._sandbox_validator import validate_agent_context

        data = {
            "response_format": {
                "requires_structured_output": False,
                "output_schema": None,
                "per_task_format_field": None,
                "description": "test",
            },
            "authentication": {
                "pre_authenticated": False,
                "credentials": None,  # missing — must fail
                "description": "form login",
            },
            "auth_mechanism": {
                "type": "form_login",
                "form_login": {
                    "login_url": "https://x/login",
                    "username_selector": "#u",
                    "password_selector": "#p",
                    "submit_selector": "#s",
                    "success_url_substring": "/dashboard",
                },
            },
            "agent_prompt_template": None,
            "site_context": {"platform_name": "X", "description": "X."},
        }
        errors = validate_agent_context(data, site_name="gitlab")
        assert any("credentials" in e for e in errors), (
            f"expected a credentials error, got: {errors}"
        )

    def test_validator_accepts_success_url_substring_canonical_name(self):
        from worldsim._sandbox_validator import _validate_auth_mechanism

        mech = {
            "type": "form_login",
            "form_login": {
                "login_url": "https://x/login",
                "username_selector": "#u",
                "password_selector": "#p",
                "submit_selector": "#s",
                "success_url_substring": "/ok",
            },
        }
        assert _validate_auth_mechanism(mech) == []

    def test_validator_accepts_legacy_success_substring_alias(self):
        from worldsim._sandbox_validator import _validate_auth_mechanism

        mech = {
            "type": "form_login",
            "form_login": {
                "login_url": "https://x/login",
                "username_selector": "#u",
                "password_selector": "#p",
                "submit_selector": "#s",
                "success_substring": "/ok",
            },
        }
        assert _validate_auth_mechanism(mech) == []

    def test_validator_rejects_form_login_without_any_success_marker(self):
        from worldsim._sandbox_validator import _validate_auth_mechanism

        mech = {
            "type": "form_login",
            "form_login": {
                "login_url": "https://x/login",
                "username_selector": "#u",
                "password_selector": "#p",
                "submit_selector": "#s",
            },
        }
        errors = _validate_auth_mechanism(mech)
        assert any("success" in e for e in errors), f"expected success-marker error: {errors}"
