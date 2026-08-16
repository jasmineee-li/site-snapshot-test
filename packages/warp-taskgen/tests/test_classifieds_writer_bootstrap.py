from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from scripts.bootstrap_classifieds_writer import (
    CLASSIFIEDS_PASSWORD_ENV,
    CLASSIFIEDS_USERNAME_ENV,
    ClassifiedsWriterBootstrapError,
    mint_classifieds_writer_storage_state,
)

SITE_URL = "http://127.0.0.1:18080"


class _FakeResponse:
    status = 200


@pytest.mark.parametrize(
    "site_url",
    [
        "https://classifieds.example",
        "http://localhost:18080",
        "http://127.0.0.1",
        "http://user:pass@127.0.0.1:18080",
        "http://127.0.0.1:18080/private",
    ],
)
def test_writer_bootstrap_rejects_non_canary_origins(site_url: str, tmp_path: Path) -> None:
    with pytest.raises(ClassifiedsWriterBootstrapError, match="configured loopback"):
        mint_classifieds_writer_storage_state(
            site_url,
            tmp_path / "writer.json",
            env={
                CLASSIFIEDS_USERNAME_ENV: "writer",
                CLASSIFIEDS_PASSWORD_ENV: "secret",
            },
            playwright_factory=lambda: pytest.fail("browser must not start"),
            repo_root=tmp_path / "checkout",
        )


class _FakeLocator:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self._calls = calls

    def fill(self, value: str) -> None:
        self._calls.append(("fill", value))

    def click(self) -> None:
        self._calls.append(("click", ""))

    def check(self) -> None:
        self._calls.append(("check", ""))


class _FakePage:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls
        self.url = ""

    def goto(self, url: str, *, wait_until: str) -> _FakeResponse:
        self.calls.append(("goto", url))
        self.url = url
        return _FakeResponse()

    def locator(self, selector: str) -> _FakeLocator:
        self.calls.append(("locator", selector))
        return _FakeLocator(self.calls)

    def get_by_role(self, role: str, *, name: str, exact: bool) -> _FakeLocator:
        self.calls.append(("role", f"{role}:{name}:{exact}"))
        return _FakeLocator(self.calls)

    def wait_for_load_state(self, state: str) -> None:
        self.calls.append(("wait", state))

    def content(self) -> str:
        return '<a href="/index.php?page=logout">Log out</a>'


class _FakeContext:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    def new_page(self) -> _FakePage:
        return _FakePage(self.calls)

    def storage_state(self) -> dict[str, object]:
        return {
            "cookies": [
                {
                    "name": "oc_userId",
                    "value": "opaque-user",
                    "domain": "127.0.0.1",
                    "expires": 2_000_000_000,
                },
                {
                    "name": "oc_userSecret",
                    "value": "opaque-secret",
                    "domain": "127.0.0.1",
                    "expires": 2_000_000_000,
                },
            ],
            "origins": [],
        }

    def close(self) -> None:
        self.calls.append(("close", "context"))


class _FakeBrowser:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    def new_context(self) -> _FakeContext:
        return _FakeContext(self.calls)

    def close(self) -> None:
        self.calls.append(("close", "browser"))


class _FakeChromium:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    def launch(self, *, headless: bool) -> _FakeBrowser:
        self.calls.append(("launch", str(headless)))
        return _FakeBrowser(self.calls)


class _FakePlaywright:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.chromium = _FakeChromium(calls)

    def __enter__(self) -> _FakePlaywright:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _fake_factory(calls: list[tuple[str, str]]):
    return lambda: _FakePlaywright(calls)


def _env() -> dict[str, str]:
    return {
        CLASSIFIEDS_USERNAME_ENV: "configured-user",
        CLASSIFIEDS_PASSWORD_ENV: "configured-pass",
    }


def test_mints_atomic_private_state_using_pinned_login_surface(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    output = tmp_path / "secrets" / "classifieds-writer.json"

    metadata = mint_classifieds_writer_storage_state(
        SITE_URL,
        output,
        env=_env(),
        playwright_factory=_fake_factory(calls),
    )

    assert metadata == {
        "authenticated": True,
        "cookies_present": True,
        "output_path": str(output.resolve()),
        "password_present": True,
        "username_present": True,
    }
    state = json.loads(output.read_text(encoding="utf-8"))
    assert state["cookies"]
    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR
    assert ("goto", f"{SITE_URL}/index.php?page=login") in calls
    assert ("locator", "#email") in calls
    assert ("locator", "#password") in calls
    assert ("locator", "#remember") in calls
    assert ("check", "") in calls
    assert ("role", "button:Log in:True") in calls
    assert ("goto", f"{SITE_URL}/index.php?page=user&action=dashboard") in calls
    serialized_metadata = json.dumps(metadata)
    assert "configured-user" not in serialized_metadata
    assert "configured-pass" not in serialized_metadata


def test_rejects_session_only_login_state_before_publishing(tmp_path: Path) -> None:
    class SessionContext(_FakeContext):
        def storage_state(self) -> dict[str, object]:
            return {
                "cookies": [{"name": "osclass", "value": "ephemeral"}],
                "origins": [],
            }

    class SessionBrowser(_FakeBrowser):
        def new_context(self) -> SessionContext:
            return SessionContext(self.calls)

    class SessionChromium(_FakeChromium):
        def launch(self, *, headless: bool) -> SessionBrowser:
            self.calls.append(("launch", str(headless)))
            return SessionBrowser(self.calls)

    class SessionPlaywright(_FakePlaywright):
        def __init__(self, calls: list[tuple[str, str]]) -> None:
            self.chromium = SessionChromium(calls)

    output = tmp_path / "writer.json"
    with pytest.raises(ClassifiedsWriterBootstrapError, match="persistent participant cookies"):
        mint_classifieds_writer_storage_state(
            SITE_URL,
            output,
            env=_env(),
            playwright_factory=lambda: SessionPlaywright([]),
        )
    assert not output.exists()


def test_accepts_pinned_dashboard_redirect_to_user_items(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    class RedirectedPage(_FakePage):
        def goto(self, url: str, *, wait_until: str) -> _FakeResponse:
            response = super().goto(url, wait_until=wait_until)
            if url.endswith("page=user&action=dashboard"):
                self.url = f"{SITE_URL}/index.php?page=user&action=items"
            return response

    class RedirectedContext(_FakeContext):
        def new_page(self) -> RedirectedPage:
            return RedirectedPage(self.calls)

    class RedirectedBrowser(_FakeBrowser):
        def new_context(self) -> RedirectedContext:
            return RedirectedContext(self.calls)

    class RedirectedChromium(_FakeChromium):
        def launch(self, *, headless: bool) -> RedirectedBrowser:
            self.calls.append(("launch", str(headless)))
            return RedirectedBrowser(self.calls)

    class RedirectedPlaywright(_FakePlaywright):
        def __init__(self, calls: list[tuple[str, str]]) -> None:
            self.chromium = RedirectedChromium(calls)

    output = tmp_path / "writer.json"
    metadata = mint_classifieds_writer_storage_state(
        SITE_URL,
        output,
        env=_env(),
        playwright_factory=lambda: RedirectedPlaywright(calls),
    )

    assert metadata["authenticated"] is True
    assert output.exists()


@pytest.mark.parametrize("missing", [CLASSIFIEDS_USERNAME_ENV, CLASSIFIEDS_PASSWORD_ENV])
def test_requires_named_environment_credentials_without_starting_browser(
    tmp_path: Path,
    missing: str,
) -> None:
    env = _env()
    del env[missing]
    started = False

    def factory():
        nonlocal started
        started = True
        raise AssertionError("browser must not start before credential validation")

    with pytest.raises(ClassifiedsWriterBootstrapError, match="required environment"):
        mint_classifieds_writer_storage_state(
            SITE_URL,
            tmp_path / "writer.json",
            env=env,
            playwright_factory=factory,
        )
    assert started is False


def test_rejects_dashboard_without_logout_marker_and_does_not_publish_state(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    class PageWithoutLogout(_FakePage):
        def content(self) -> str:
            return "<main>login form</main>"

    class ContextWithoutLogout(_FakeContext):
        def new_page(self) -> PageWithoutLogout:
            return PageWithoutLogout(self.calls)

    class BrowserWithoutLogout(_FakeBrowser):
        def new_context(self) -> ContextWithoutLogout:
            return ContextWithoutLogout(self.calls)

    class ChromiumWithoutLogout(_FakeChromium):
        def launch(self, *, headless: bool) -> BrowserWithoutLogout:
            self.calls.append(("launch", str(headless)))
            return BrowserWithoutLogout(self.calls)

    class PlaywrightWithoutLogout(_FakePlaywright):
        def __init__(self, calls: list[tuple[str, str]]) -> None:
            self.chromium = ChromiumWithoutLogout(calls)

    output = tmp_path / "writer.json"
    with pytest.raises(ClassifiedsWriterBootstrapError, match="logout marker"):
        mint_classifieds_writer_storage_state(
            SITE_URL,
            output,
            env=_env(),
            playwright_factory=lambda: PlaywrightWithoutLogout(calls),
        )
    assert not output.exists()


def test_rejects_cookie_output_inside_source_checkout(tmp_path: Path) -> None:
    output = tmp_path / "secrets" / "writer.json"

    with pytest.raises(ClassifiedsWriterBootstrapError, match="outside the source checkout"):
        mint_classifieds_writer_storage_state(
            SITE_URL,
            output,
            env=_env(),
            playwright_factory=_fake_factory([]),
            repo_root=tmp_path,
        )
    assert not output.exists()


def test_default_cli_guard_finds_checkout_root_from_script_location() -> None:
    from scripts import bootstrap_classifieds_writer as module

    output = Path(module.__file__).resolve().parents[1] / "would-leak-writer-state.json"
    with pytest.raises(ClassifiedsWriterBootstrapError, match="outside the source checkout"):
        mint_classifieds_writer_storage_state(
            SITE_URL,
            output,
            env=_env(),
            playwright_factory=_fake_factory([]),
        )
    assert not output.exists()


def test_failure_metadata_does_not_echo_environment_values(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    from scripts import bootstrap_classifieds_writer as module

    monkeypatch.setenv(CLASSIFIEDS_USERNAME_ENV, "configured-user")
    monkeypatch.delenv(CLASSIFIEDS_PASSWORD_ENV, raising=False)
    assert (
        module.main(["--site-url", SITE_URL, "--output-path", str(tmp_path / "writer.json")]) == 1
    )
    output = capsys.readouterr().out
    assert "configured-user" not in output
    assert "configured-pass" not in output
    assert "password_present" not in output
