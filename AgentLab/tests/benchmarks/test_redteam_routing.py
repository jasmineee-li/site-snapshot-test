from __future__ import annotations

import pytest

from agentlab.benchmarks.redteam.routing import (
    _browser_route_guard_redirect_url,
    _make_proxy_handler,
    _resolve_navigation_containment_policy,
    install_browser_route_guard,
    setup_route_interception,
)


class _FakeRequest:
    def __init__(self, url: str, method: str = "GET"):
        self.url = url
        self.method = method


class _FakeRoute:
    def __init__(self, url: str, method: str = "GET"):
        self.request = _FakeRequest(url, method=method)
        self.continue_calls: list[dict[str, str]] = []
        self.fulfill_calls: list[dict[str, object]] = []

    def continue_(self, **kwargs):
        self.continue_calls.append(kwargs)

    def fulfill(self, **kwargs):
        self.fulfill_calls.append(kwargs)


class _FakeContext:
    def __init__(self) -> None:
        self.routes: list[tuple[str, object]] = []

    def route(self, pattern: str, handler) -> None:
        self.routes.append((pattern, handler))


def _mail_pages() -> list[dict[str, object]]:
    return [
        {
            "id": "mail",
            "base_site_url": "https://mail.example.com",
            "subdomains": ["/mail/u/0/#inbox", "/mail/u/0/#settings"],
        }
    ]


def _mail_and_admin_pages() -> list[dict[str, object]]:
    return [
        {
            "id": "mail",
            "base_site_url": "https://mail.example.com",
            "subdomains": ["/mail/u/0/#inbox"],
        },
        {
            "id": "admin",
            "base_site_url": "https://admin.example.com",
            "subdomains": ["/mail/u/0/#inbox"],
        },
    ]


def test_make_proxy_handler_rewrites_sse_requests_to_loopback() -> None:
    route = _FakeRoute("https://mail.example.com/api/events?cursor=1")
    handler = _make_proxy_handler("mail.example.com", 43123)

    handler(route)

    assert route.continue_calls == [{"url": "http://127.0.0.1:43123/api/events?cursor=1"}]
    assert route.fulfill_calls == []


def test_make_proxy_handler_rewrites_deep_link_navigation_to_app_shell() -> None:
    route = _FakeRoute("https://mail.example.com/mail/u/0/?view=thread")
    handler = _make_proxy_handler("mail.example.com", 43123)

    handler(route)

    assert route.continue_calls == [{"url": "http://127.0.0.1:43123/?view=thread"}]
    assert route.fulfill_calls == []


def test_make_proxy_handler_blocks_non_whitelisted_non_get_requests() -> None:
    route = _FakeRoute("https://mail.example.com/mail/u/0/?view=thread", method="POST")
    handler = _make_proxy_handler("mail.example.com", 43123)

    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_make_proxy_handler_preserves_static_asset_requests() -> None:
    route = _FakeRoute("https://mail.example.com/js/app.js?v=1")
    handler = _make_proxy_handler("mail.example.com", 43123)

    handler(route)

    assert route.continue_calls == [{"url": "http://127.0.0.1:43123/js/app.js?v=1"}]
    assert route.fulfill_calls == []


def test_make_proxy_handler_preserves_file_like_requests() -> None:
    route = _FakeRoute("https://mail.example.com/downloads/report.csv")
    handler = _make_proxy_handler("mail.example.com", 43123)

    handler(route)

    assert route.continue_calls == [{"url": "http://127.0.0.1:43123/downloads/report.csv"}]
    assert route.fulfill_calls == []


def test_setup_route_interception_blocks_unlisted_domains() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        server_port=43123,
    )

    assert len(context.routes) == 1
    _, handler = context.routes[0]
    route = _FakeRoute("https://evil.example.com/phish")
    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_blocks_loopback_requests_to_other_ports() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("http://127.0.0.1:9999/secret")
    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_allows_only_the_active_loopback_server_port() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("http://localhost:43123/js/app.js")
    handler(route)

    assert route.fulfill_calls == []
    assert route.continue_calls == [{}]


def test_setup_route_interception_allows_declared_mock_endpoints_on_primary_spa_domains() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        mock_endpoints=["/mock/file/password.txt"],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://mail.example.com/mock/file/password.txt")
    handler(route)

    assert route.fulfill_calls == []
    assert route.continue_calls == [{"url": "http://127.0.0.1:43123/mock/file/password.txt"}]


def test_setup_route_interception_allows_mock_post_endpoints_on_primary_spa_domains() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        mock_endpoints=["/mock/mail/submit"],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://mail.example.com/mock/mail/submit", method="POST")
    handler(route)

    assert route.fulfill_calls == []
    assert route.continue_calls == [{"url": "http://127.0.0.1:43123/mock/mail/submit"}]


def test_setup_route_interception_blocks_undeclared_mock_endpoints_on_primary_spa_domains() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        mock_endpoints=["/mock/mail/submit"],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://mail.example.com/mock/file/password.txt")
    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_blocks_reset_endpoint_on_primary_spa_domains() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://mail.example.com/api/reset", method="POST")
    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_allows_browser_state_sync_on_primary_spa_domains() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        server_port=43123,
    )

    _, handler = context.routes[0]

    put_route = _FakeRoute("https://mail.example.com/api/state", method="PUT")
    handler(put_route)
    assert put_route.fulfill_calls == []
    assert put_route.continue_calls == [{"url": "http://127.0.0.1:43123/api/state"}]

    events_route = _FakeRoute("https://mail.example.com/api/events?cursor=1")
    handler(events_route)
    assert events_route.fulfill_calls == []
    assert events_route.continue_calls == [{"url": "http://127.0.0.1:43123/api/events?cursor=1"}]


def test_setup_route_interception_enforces_mock_only_paths() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mock.example.com", "mode": "mock"}],
        mock_endpoints=["/mock/file/password.txt"],
        server_port=43123,
    )

    _, handler = context.routes[0]

    allowed_route = _FakeRoute("https://mock.example.com/mock/file/password.txt")
    handler(allowed_route)
    assert allowed_route.continue_calls == [
        {"url": "http://127.0.0.1:43123/mock/file/password.txt"}
    ]

    blocked_route = _FakeRoute("https://mock.example.com/not-mock")
    handler(blocked_route)
    assert blocked_route.continue_calls == []
    assert blocked_route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_blocks_undeclared_mock_paths_on_mock_domains() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mock.example.com", "mode": "mock"}],
        mock_endpoints=["/mock/mail/submit"],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://mock.example.com/mock/file/password.txt")
    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_blocks_mock_path_prefix_confusion() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mock.example.com", "mode": "mock"}],
        mock_endpoints=["/mock/file/password.txt"],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://mock.example.com/mockery/file/password.txt")
    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_enforces_allowed_navigation_targets() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "mail.example.com", "mode": "primary_spa"}],
        pages=_mail_pages(),
        allowed_routes=["#/inbox"],
        server_port=43123,
    )

    _, handler = context.routes[0]

    allowed_route = _FakeRoute("https://mail.example.com/mail/u/0/")
    handler(allowed_route)
    assert allowed_route.fulfill_calls == []
    assert allowed_route.continue_calls == [{"url": "http://127.0.0.1:43123/"}]

    blocked_route = _FakeRoute("https://mail.example.com/settings")
    handler(blocked_route)
    assert blocked_route.continue_calls == []
    assert blocked_route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_routes_declared_shim_to_local_prefix() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "shim.example.com", "mode": "shim", "shim_path": "/shim"}],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://shim.example.com/login")
    handler(route)

    assert route.fulfill_calls == []
    assert route.continue_calls == [{"url": "http://127.0.0.1:43123/shim/login"}]


def test_setup_route_interception_preserves_query_for_shim_routes() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "shim.example.com", "mode": "shim", "shim_path": "/shim"}],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://shim.example.com/login?next=%2Fsettings")
    handler(route)

    assert route.fulfill_calls == []
    assert route.continue_calls == [
        {"url": "http://127.0.0.1:43123/shim/login?next=%2Fsettings"}
    ]


def test_setup_route_interception_blocks_traversal_on_shim_routes() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[{"domain": "shim.example.com", "mode": "shim", "shim_path": "/shim"}],
        server_port=43123,
    )

    _, handler = context.routes[0]
    route = _FakeRoute("https://shim.example.com/%2e%2e/api/state")
    handler(route)

    assert route.continue_calls == []
    assert route.fulfill_calls[0]["status"] == 451


def test_setup_route_interception_scopes_allowed_navigation_targets_by_domain() -> None:
    context = _FakeContext()
    setup_route_interception(
        context,
        domain_bindings=[
            {"domain": "mail.example.com", "mode": "primary_spa"},
            {"domain": "admin.example.com", "mode": "primary_spa"},
        ],
        pages=_mail_and_admin_pages(),
        allowed_routes=["https://mail.example.com/mail/u/0/#/inbox"],
        server_port=43123,
    )

    _, handler = context.routes[0]

    allowed_route = _FakeRoute("https://mail.example.com/mail/u/0/")
    handler(allowed_route)
    assert allowed_route.fulfill_calls == []
    assert allowed_route.continue_calls == [{"url": "http://127.0.0.1:43123/"}]

    blocked_route = _FakeRoute("https://admin.example.com/mail/u/0/")
    handler(blocked_route)
    assert blocked_route.continue_calls == []
    assert blocked_route.fulfill_calls[0]["status"] == 451


def test_browser_route_guard_redirects_disallowed_hash_route_on_same_shell() -> None:
    policy = _resolve_navigation_containment_policy(
        allowed_routes=["#/inbox"],
        pages=_mail_pages(),
    )

    assert (
        _browser_route_guard_redirect_url(
            "https://mail.example.com/mail/u/0/#/settings",
            policy=policy,
        )
        == "https://mail.example.com/mail/u/0/#/inbox"
    )
    assert (
        _browser_route_guard_redirect_url(
            "https://mail.example.com/mail/u/0/#/inbox",
            policy=policy,
        )
        is None
    )


def test_install_browser_route_guard_registers_init_script() -> None:
    policy = _resolve_navigation_containment_policy(
        allowed_routes=["#/inbox"],
        pages=_mail_pages(),
    )

    class _InitScriptPage:
        def __init__(self) -> None:
            self.scripts: list[str] = []

        def add_init_script(self, script: str) -> None:
            self.scripts.append(script)

    page = _InitScriptPage()
    install_browser_route_guard(page, policy=policy)

    assert len(page.scripts) == 1
    assert "mail.example.com" in page.scripts[0]
    assert "/mail/u/0/#/inbox" in page.scripts[0]


def test_setup_route_interception_rejects_relative_shim_paths() -> None:
    context = _FakeContext()
    with pytest.raises(ValueError, match="absolute shim_path"):
        setup_route_interception(
            context,
            domain_bindings=[{"domain": "shim.example.com", "mode": "shim", "shim_path": "shim"}],
            server_port=43123,
        )


def test_setup_route_interception_rejects_mock_prefix_outside_mock_namespace() -> None:
    context = _FakeContext()
    with pytest.raises(ValueError, match="must stay under /mock"):
        setup_route_interception(
            context,
            domain_bindings=[{"domain": "mock.example.com", "mode": "mock", "mock_prefix": "/api"}],
            server_port=43123,
        )


def test_setup_route_interception_rejects_shim_path_outside_shim_namespace() -> None:
    context = _FakeContext()
    with pytest.raises(ValueError, match="must stay under /shim"):
        setup_route_interception(
            context,
            domain_bindings=[{"domain": "shim.example.com", "mode": "shim", "shim_path": "/api"}],
            server_port=43123,
        )
