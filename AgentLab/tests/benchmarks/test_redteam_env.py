from __future__ import annotations

import json
import shutil
from pathlib import Path
import urllib.error
import urllib.request

import pytest

from agentlab.benchmarks.redteam.config import Page, RedteamEnvArgs
from agentlab.benchmarks.redteam.env import RedteamEnv
import agentlab.benchmarks.redteam.data_validator as data_validator_module
from agentlab.benchmarks.redteam.eval_harness import start_app_server, stop_app_server


def _write_canonical_app(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "js").mkdir(exist_ok=True)
    (app_dir / "css").mkdir(exist_ok=True)
    (app_dir / "benign").mkdir(exist_ok=True)
    (app_dir / "adversarial_demo_v0").mkdir(exist_ok=True)

    (app_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "js" / "state.js").write_text("const AppState = {};\n", encoding="utf-8")
    (app_dir / "js" / "components.js").write_text("const Components = {};\n", encoding="utf-8")
    (app_dir / "js" / "views.js").write_text("const Views = {};\n", encoding="utf-8")
    (app_dir / "js" / "app.js").write_text("const App = {};\n", encoding="utf-8")
    (app_dir / "js" / "data.js").write_text("const SOURCE = 'canonical';\n", encoding="utf-8")
    (app_dir / "benign" / "data.js").write_text("const SOURCE = 'benign';\n", encoding="utf-8")
    (app_dir / "adversarial_demo_v0" / "data.js").write_text(
        "const SOURCE = 'adversarial';\n",
        encoding="utf-8",
    )
    (app_dir / ".claude_prompt.md").write_text("prompt\n", encoding="utf-8")
    (app_dir / ".claude_output.log").write_text("output\n", encoding="utf-8")
    (app_dir / "function-tasks").mkdir(exist_ok=True)
    (app_dir / "function-tasks" / "task_1.py").write_text("def verify(server_url): return True, 'ok'\n", encoding="utf-8")
    (app_dir / "function-tasks.json").write_text("[]\n", encoding="utf-8")
    (app_dir / "real-tasks").mkdir(exist_ok=True)
    (app_dir / "real-tasks.json").write_text("[]\n", encoding="utf-8")
    (app_dir / "sanity_check_function.py").write_text("print('ok')\n", encoding="utf-8")
    (app_dir / "sanity_check_real.py").write_text("print('ok')\n", encoding="utf-8")
    (app_dir / "APP_DESCRIPTION.md").write_text("app description\n", encoding="utf-8")
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": 3,
                "app_id": "shared-mail",
                "platform": "mail",
                "docs_path": "apps/user-manuals/mail",
                "docs_snapshot": {"manuals_git_ref": "abc1234"},
                "shared_seed_version": 1,
                "shared_seed_hash": "sha256:test-seed",
                "behavior_ids": ["demo"],
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "results").mkdir(exist_ok=True)


def _mail_pages() -> list[Page]:
    return [
        Page(
            id="mail",
            base_site_url="https://mail.example.com",
            subdomains=["/mail/u/0/#inbox", "/mail/u/0/#settings"],
        )
    ]


def _make_env_args(
    app_dir: Path,
    *,
    variant: str,
    start_page: str | None = None,
    pages: list[Page] | None = None,
    safe_behavior: str = "safe goal",
) -> RedteamEnvArgs:
    return RedteamEnvArgs(
        task_name=f"task.{variant}",
        app_id="shared-mail",
        target="target",
        attacker_model="",
        pages=pages or [],
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        behavior_id="demo",
        entry_route="#inbox",
        allowed_routes=["#/inbox"],
        active_variant="adversarial_demo_v0",
        attack_type="direct",
        adversarial_mode="single_turn",
        test_mode="benign" if variant == "benign" else "adversarial",
        doc="attack goal",
        safe_behavior=safe_behavior,
        app_dir=str(app_dir),
        variant=variant,
        start_page=start_page,
    )


def test_app_mode_generation_uses_isolated_runtime_copy(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)

    benign_env = RedteamEnv(
        _make_env_args(app_dir, variant="benign"),
        exp_dir=tmp_path / "exp-benign",
    )
    adversarial_env = RedteamEnv(
        _make_env_args(app_dir, variant="adversarial_demo_v0"),
        exp_dir=tmp_path / "exp-adversarial",
    )

    benign_env._generate_sites()
    adversarial_env._generate_sites()

    assert benign_env.runtime_app_dir == (tmp_path / "exp-benign" / "app_runtime")
    assert adversarial_env.runtime_app_dir == (tmp_path / "exp-adversarial" / "app_runtime")
    assert benign_env.runtime_app_dir != adversarial_env.runtime_app_dir
    assert (benign_env.runtime_app_dir / "js" / "data.js").read_text(encoding="utf-8") == (
        "const SOURCE = 'benign';\n"
    )
    assert (
        adversarial_env.runtime_app_dir / "js" / "data.js"
    ).read_text(encoding="utf-8") == "const SOURCE = 'adversarial';\n"
    assert (app_dir / "js" / "data.js").read_text(encoding="utf-8") == (
        "const SOURCE = 'canonical';\n"
    )
    assert not (benign_env.runtime_app_dir / ".claude_prompt.md").exists()
    assert not (benign_env.runtime_app_dir / "function-tasks.json").exists()
    assert not (benign_env.runtime_app_dir / "function-tasks").exists()
    assert not (benign_env.runtime_app_dir / "sanity_check_function.py").exists()
    assert not (benign_env.runtime_app_dir / "app_manifest.json").exists()
    assert not (benign_env.runtime_app_dir / "APP_DESCRIPTION.md").exists()
    assert not (benign_env.runtime_app_dir / "results").exists()
    assert not (benign_env.runtime_app_dir / "benign").exists()
    assert not (benign_env.runtime_app_dir / "adversarial_demo_v0").exists()


@pytest.mark.parametrize(
    ("start_page", "expected_url"),
    [
        ("#inbox", "https://mail.example.com/mail/u/0/#/inbox"),
        (
            "https://mail.example.com/mail/u/0/#inbox",
            "https://mail.example.com/mail/u/0/#/inbox",
        ),
    ],
)
def test_resolve_app_start_url_normalizes_hash_routes(
    tmp_path: Path,
    start_page: str | None,
    expected_url: str,
) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)

    env = RedteamEnv(
        _make_env_args(
            app_dir,
            variant="benign",
            start_page=start_page,
            pages=_mail_pages(),
        ),
        exp_dir=tmp_path / "exp",
    )

    assert env._resolve_app_start_url() == expected_url


def test_reset_uses_safe_behavior_as_agent_goal(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)

    captured: dict[str, object] = {}

    class _FakePage:
        def __init__(self) -> None:
            self.context = object()
            self.init_scripts: list[str] = []

        def goto(self, url: str, wait_until: str | None = None) -> None:
            captured["goto_url"] = url
            captured["goto_wait_until"] = wait_until

        def add_init_script(self, script: str) -> None:
            self.init_scripts.append(script)

    class _FakeUnwrapped:
        def __init__(self) -> None:
            self.page = _FakePage()

        def _get_obs(self) -> dict[str, object]:
            return {"url": "about:blank"}

    class _FakeBrowserEnv:
        def __init__(self) -> None:
            self.unwrapped = _FakeUnwrapped()

        def reset(self, seed: int = None):
            captured["seed"] = seed
            return {"goal_object": []}, {}

        def close(self) -> None:
            return None

    def fake_make(_env_id: str, **kwargs):
        captured["task_kwargs"] = kwargs["task_kwargs"]
        return _FakeBrowserEnv()

    monkeypatch.setattr("agentlab.benchmarks.redteam.env.gym.make", fake_make)
    monkeypatch.setattr("agentlab.benchmarks.redteam.env.RedteamEnv._start_server", lambda self: setattr(self, "server_port", 7777))
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.routing.setup_route_interception",
        lambda **kwargs: (captured.setdefault("route_setup", kwargs["server_port"]), None)[1],
    )
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.routing.install_browser_route_guard",
        lambda page, *, policy: captured.setdefault("route_guard_policy", policy),
    )

    env = RedteamEnv(
        _make_env_args(
            app_dir,
            variant="benign",
            start_page="#inbox",
            pages=_mail_pages(),
            safe_behavior="Open the inbox safely",
        ),
        exp_dir=tmp_path / "exp",
    )

    obs, info = env.reset(seed=7)

    assert captured["task_kwargs"]["goal"] == "Open the inbox safely"
    assert obs["goal_object"] == [{"type": "text", "text": "Open the inbox safely"}]
    assert info["doc"] == "Open the inbox safely"
    assert "route_guard_policy" in captured


def test_reset_tears_down_existing_browser_and_server_state(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)

    events: list[str] = []

    class _FakePage:
        def __init__(self) -> None:
            self.context = object()
            self.init_scripts: list[str] = []

        def goto(self, url: str, wait_until: str | None = None) -> None:
            events.append(f"goto:{url}")

        def add_init_script(self, script: str) -> None:
            self.init_scripts.append(script)

    class _FakeUnwrapped:
        def __init__(self) -> None:
            self.page = _FakePage()

        def _get_obs(self) -> dict[str, object]:
            return {"url": "about:blank"}

    class _FakeBrowserEnv:
        def __init__(self, label: str) -> None:
            self.label = label
            self.closed = False
            self.unwrapped = _FakeUnwrapped()

        def reset(self, seed: int = None):
            events.append(f"reset:{self.label}:{seed}")
            return {"goal_object": []}, {}

        def close(self) -> None:
            self.closed = True
            events.append(f"close:{self.label}")

    old_browser = _FakeBrowserEnv("old")
    new_browser = _FakeBrowserEnv("new")

    monkeypatch.setattr("agentlab.benchmarks.redteam.env.gym.make", lambda _env_id, **kwargs: new_browser)
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.env.RedteamEnv._start_server",
        lambda self: (
            events.append("start-server"),
            setattr(self, "server_proc", object()),
            setattr(self, "server_port", 7777),
        ),
    )
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.env.RedteamEnv._stop_server_process",
        lambda self: (
            events.append("stop-server"),
            setattr(self, "server_proc", None),
            setattr(self, "server_port", None),
        ),
    )
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.routing.setup_route_interception",
        lambda **kwargs: (events.append(f"route:{kwargs['server_port']}"), None)[1],
    )
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.routing.install_browser_route_guard",
        lambda page, *, policy: events.append("route-guard"),
    )

    env = RedteamEnv(
        _make_env_args(app_dir, variant="benign", start_page="#inbox", pages=_mail_pages()),
        exp_dir=tmp_path / "exp",
    )
    env.browser_env = old_browser
    env.server_proc = object()
    env.server_port = 4321

    env.reset(seed=11)

    assert old_browser.closed is True
    assert env.browser_env is new_browser
    assert env.server_port == 7777
    assert events[:6] == ["close:old", "stop-server", "start-server", "reset:new:11", "route:7777", "route-guard"]


def test_reset_cleans_up_partial_setup_failure(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)

    stop_ports: list[int | None] = []

    class _FakeBrowserEnv:
        def __init__(self) -> None:
            self.closed = False
            self.unwrapped = type(
                "_FakeUnwrapped",
                (),
                {"page": type("_FakePage", (), {"context": object(), "add_init_script": lambda self, script: None})()},
            )()

        def reset(self, seed: int = None):
            return {"goal_object": []}, {}

        def close(self) -> None:
            self.closed = True

    browser_env = _FakeBrowserEnv()

    monkeypatch.setattr("agentlab.benchmarks.redteam.env.gym.make", lambda _env_id, **kwargs: browser_env)
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.env.RedteamEnv._start_server",
        lambda self: (
            setattr(self, "server_proc", object()),
            setattr(self, "server_port", 7777),
        ),
    )

    def fake_stop_server(self) -> None:
        stop_ports.append(self.server_port)
        self.server_proc = None
        self.server_port = None

    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.env.RedteamEnv._stop_server_process",
        fake_stop_server,
    )
    monkeypatch.setattr(
        "agentlab.benchmarks.redteam.routing.setup_route_interception",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    env = RedteamEnv(
        _make_env_args(app_dir, variant="benign", start_page="#inbox", pages=_mail_pages()),
        exp_dir=tmp_path / "exp",
    )

    with pytest.raises(RuntimeError, match="boom"):
        env.reset(seed=5)

    assert browser_env.closed is True
    assert env.browser_env is None
    assert env.server_port is None
    assert 7777 in stop_ports


def test_resolve_app_start_url_requires_external_route_metadata(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)

    env = RedteamEnv(
        _make_env_args(app_dir, variant="benign", start_page="#inbox", pages=[]),
        exp_dir=tmp_path / "exp",
    )

    with pytest.raises(RuntimeError, match="base_site_url"):
        env._resolve_app_start_url()


def test_template_server_only_serves_public_app_surface(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)
    shutil.copy2(
        Path(data_validator_module.__file__).resolve().parent / "templates" / "server.py",
        app_dir / "server.py",
    )

    proc, port = start_app_server(app_dir)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/js/app.js", timeout=5) as response:
            assert response.status == 200
            assert "const App = {};" in response.read().decode("utf-8")
            assert response.headers["Content-Security-Policy"] is not None
            assert response.headers["X-Content-Type-Options"] == "nosniff"

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Security-Policy"] is not None
            assert response.headers["X-Content-Type-Options"] == "nosniff"

        for hidden_path in (
            "/.claude_prompt.md",
            "/app_manifest.json",
            "/function-tasks.json",
            "/sanity_check_function.py",
            "/function-tasks/task_1.py",
            "/js/../app_manifest.json",
            "/js/../server.py",
            "/js/%2e%2e/app_manifest.json",
        ):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{hidden_path}", timeout=5)
            assert exc_info.value.code == 404

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/app_manifest.json",
                    method="HEAD",
                ),
                timeout=5,
            )
        assert exc_info.value.code == 404
    finally:
        stop_app_server(proc)


def test_start_server_reaps_app_process_on_timeout(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _write_canonical_app(app_dir)
    monkeypatch.setenv("SERVER_EVENTS_LOG", str(tmp_path / "ambient-events.log"))

    env = RedteamEnv(
        _make_env_args(app_dir, variant="benign"),
        exp_dir=tmp_path / "exp",
    )

    captured: dict[str, object] = {}

    def fake_start_app_server(app_dir_arg, port=None, timeout=10.0, env=None):
        captured["app_dir"] = app_dir_arg
        captured["port"] = port
        captured["timeout"] = timeout
        captured["env"] = env
        raise RuntimeError("Server did not start within 15.0s on port ephemeral")

    monkeypatch.setattr("agentlab.benchmarks.redteam.env.start_app_server", fake_start_app_server)

    with pytest.raises(RuntimeError, match="did not start within 15.0s"):
        env._start_server()

    assert captured["app_dir"] == env.exp_dir / "app_runtime"
    assert captured["port"] is None
    assert captured["timeout"] == 15.0
    assert captured["env"]["SERVER_EVENTS_LOG"] == str(
        (env.exp_dir / "server_events.log").resolve()
    )
    assert env.server_proc is None
    assert env.server_port is None
