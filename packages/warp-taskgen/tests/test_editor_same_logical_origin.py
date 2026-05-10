"""Unit tests for the proxy-aware cross-origin relaxation in editors/base.py.

Prevents regression of the editor's ``_resolve_form_action`` treating a
proxy-rewritten origin (Magento's baked-in ``base_url`` at the proxy port)
as a different origin from ``site_url`` at the raw backend port.

The test surface exercises ``_same_logical_origin`` directly with explicit
``ProxyInfo`` values — not via module globals — so each test is self-
contained and parallel-safe.
"""

from __future__ import annotations

from urllib.parse import urlparse

from worldsim import http_proxy
from worldsim.editors.base import _same_logical_origin
from worldsim.http_proxy import ProxyInfo


def _u(url: str):
    return urlparse(url)


# ---------------------------------------------------------------------------
# Pure function behaviour: _same_logical_origin takes proxy_info explicitly.
# ---------------------------------------------------------------------------


def test_identical_origins_are_equivalent():
    assert _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:7770/customer/account/"),
    )


def test_different_schemes_are_not_equivalent():
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("https://3.12.221.9:7770/"),
    )


def test_different_hosts_are_not_equivalent():
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://attacker.example:7770/"),
    )


def test_different_ports_without_proxy_info_are_not_equivalent():
    # proxy_info not supplied: default strict behavior.
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:17770/"),
    )


def test_proxy_port_is_equivalent_when_proxy_info_supplied():
    info = ProxyInfo(
        port_offset=10000,
        site_ports=frozenset({7770, 7780, 8023, 9999, 8888, 3030}),
    )
    assert _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:17770/customer/account/editPost/"),
        proxy_info=info,
    )


def test_reverse_direction_requires_site_port_in_allowlist():
    # site_port=17770 is not in the raw-ports allowlist — strict mode.
    info = ProxyInfo(port_offset=10000, site_ports=frozenset({7770}))
    assert not _same_logical_origin(
        _u("http://3.12.221.9:17770"),
        _u("http://3.12.221.9:7770/"),
        proxy_info=info,
    )
    # With site_port=7770 and action_port=17770, equivalence holds.
    assert _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:17770/"),
        proxy_info=info,
    )


def test_unknown_backend_port_is_rejected_even_with_proxy():
    # Attack-surface guard: a form action pointing at a port that is NOT
    # the proxy-equivalent of the raw site port must be rejected even with
    # proxy_info supplied. Magento cannot coerce us into POSTing to :9999
    # just because we accept :17770.
    info = ProxyInfo(port_offset=10000, site_ports=frozenset({7770}))
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:9999/"),
        proxy_info=info,
    )


def test_missing_ports_are_rejected():
    info = ProxyInfo(port_offset=10000, site_ports=frozenset({7770}))
    assert not _same_logical_origin(
        _u("http://3.12.221.9"),
        _u("http://3.12.221.9:17770/"),
        proxy_info=info,
    )
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9/"),
        proxy_info=info,
    )


def test_empty_site_ports_never_matches_cross_port():
    info = ProxyInfo(port_offset=10000, site_ports=frozenset())
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:17770/"),
        proxy_info=info,
    )


# ---------------------------------------------------------------------------
# Session-adapter plumbing: proxy_info_from_session + install_proxy
# ---------------------------------------------------------------------------


def test_proxy_info_from_session_returns_none_without_proxy():
    # A plain session has no ProxyingHTTPAdapter mounted -> None.
    import requests

    session = requests.Session()
    assert http_proxy.proxy_info_from_session(session, "http://3.12.221.9:7770") is None


def test_make_proxied_session_exposes_proxy_info():
    session = http_proxy.make_proxied_session(
        token="dummy-token",
        port_offset=10000,
        site_ports=[7770, 7780],
    )
    info = http_proxy.proxy_info_from_session(session, "http://3.12.221.9:7770")
    assert info is not None
    assert info.port_offset == 10000
    assert info.site_ports == frozenset({7770, 7780})


def test_install_proxy_round_trip_via_new_sessions():
    import requests

    uninstall = http_proxy.install_proxy(
        token="dummy-token",
        port_offset=10000,
        site_ports=[7770],
    )
    try:
        # Sessions constructed AFTER install_proxy inherit the adapter and
        # therefore carry the ProxyInfo reachable via proxy_info_from_session.
        patched = requests.Session()
        info = http_proxy.proxy_info_from_session(patched, "http://3.12.221.9:7770")
        assert info is not None
        assert info.port_offset == 10000
        assert info.site_ports == frozenset({7770})
        # And the editor's origin check accepts :17770 given that info.
        assert _same_logical_origin(
            _u("http://3.12.221.9:7770"),
            _u("http://3.12.221.9:17770/foo"),
            proxy_info=info,
        )
    finally:
        uninstall()

    # After uninstall, fresh sessions are unproxied again.
    fresh = requests.Session()
    assert http_proxy.proxy_info_from_session(fresh, "http://3.12.221.9:7770") is None
