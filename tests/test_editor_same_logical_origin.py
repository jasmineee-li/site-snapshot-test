"""Unit tests for the proxy-aware cross-origin relaxation in editors/base.py.

Prevents regression of the editor's ``_resolve_form_action`` treating a
proxy-rewritten origin (Magento's baked-in ``base_url`` at the proxy port)
as a different origin from ``site_url`` at the raw backend port.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from worldsim import http_proxy
from worldsim.editors.base import _same_logical_origin


@pytest.fixture(autouse=True)
def _reset_proxy_state():
    """Ensure tests start and end without a patched requests.Session."""
    # Snapshot + restore the module-level state without touching
    # requests.Session (we don't need a real session for these tests).
    saved_offset = http_proxy._installed_port_offset
    saved_ports = http_proxy._installed_site_ports
    yield
    http_proxy._installed_port_offset = saved_offset
    http_proxy._installed_site_ports = saved_ports


def _u(url: str):
    return urlparse(url)


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


def test_different_ports_without_proxy_are_not_equivalent():
    # No proxy installed: default strict behavior.
    http_proxy._installed_port_offset = None
    http_proxy._installed_site_ports = frozenset()
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:17770/"),
    )


def test_proxy_port_is_equivalent_when_proxy_installed():
    http_proxy._installed_port_offset = 10000
    http_proxy._installed_site_ports = frozenset({7770, 7780, 8023, 9999, 8888, 3030})
    assert _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:17770/customer/account/editPost/"),
    )


def test_reverse_direction_is_equivalent():
    # site_url was the proxy port, action was the raw port.
    http_proxy._installed_port_offset = 10000
    http_proxy._installed_site_ports = frozenset({7770})
    # site_port 17770 not in site_ports -> strict check fails; reverse
    # direction only passes when site_port itself is a known raw port.
    assert not _same_logical_origin(
        _u("http://3.12.221.9:17770"),
        _u("http://3.12.221.9:7770/"),
    )
    # But with site_port=7770 and action_port=17770, equivalence holds.
    assert _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:17770/"),
    )


def test_unknown_backend_port_is_rejected_even_with_proxy():
    # Even with proxy installed, a form action pointing at a port that
    # isn't on the site_ports allowlist must be rejected. This is the
    # attack-surface guard: Magento cannot coerce us into POSTing to
    # :99999 just because we accept :17770.
    http_proxy._installed_port_offset = 10000
    http_proxy._installed_site_ports = frozenset({7770})
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9:9999/"),
    )


def test_missing_ports_are_rejected():
    http_proxy._installed_port_offset = 10000
    http_proxy._installed_site_ports = frozenset({7770})
    assert not _same_logical_origin(
        _u("http://3.12.221.9"),
        _u("http://3.12.221.9:17770/"),
    )
    assert not _same_logical_origin(
        _u("http://3.12.221.9:7770"),
        _u("http://3.12.221.9/"),
    )


def test_install_and_uninstall_proxy_round_trip():
    # Smoke test on the public API: install_proxy populates the module-
    # level state, uninstall clears it.
    uninstall = http_proxy.install_proxy(
        token="dummy-token",
        port_offset=10000,
        site_ports=[7770],
    )
    try:
        offset, ports = http_proxy.get_installed_proxy_info()
        assert offset == 10000
        assert ports == frozenset({7770})
        assert _same_logical_origin(
            _u("http://3.12.221.9:7770"),
            _u("http://3.12.221.9:17770/foo"),
        )
    finally:
        uninstall()
    offset, ports = http_proxy.get_installed_proxy_info()
    assert offset is None
    assert ports == frozenset()
