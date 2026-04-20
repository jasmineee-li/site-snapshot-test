"""Client-side verification-proxy wiring.

When the user's benchmark is fronted by a token-gated nginx reverse proxy
(see ``scripts/deploy_benchmark_proxy.sh`` and ``VerificationProxy`` in
``worldsim/config.py``), integration-test code must:

1. Rewrite every outbound site-port URL to the proxy port (``real_port +
   port_offset``).
2. Attach ``X-Worldsim-Token: <token>`` so nginx accepts the request.

This module provides a ``requests.adapters.HTTPAdapter`` subclass that does
both transparently and an ``install_proxy`` helper that patches
``requests.Session.__init__`` process-wide so ALL newly-created sessions
pick up the adapter. That is necessary because ``apply_data_seed``,
``acquire_tokens_for_instances``, and each editor's ``probe_base_state``
construct their own ``requests.Session()`` internally; threading a session
through every call site would require a much larger refactor.

Scope: rewriting is restricted to the site-port allowlist supplied at
install time. Ports outside that set (notably the ``reset_endpoint`` ports
at ``site_port + 1``, which are not covered by the current nginx config)
pass through unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter

# Default HTTP/HTTPS ports that upstream apps (GitLab, Magento) generate into
# redirect Location headers when proxied. These are rewritten to the most
# recently observed proxy port for the same hostname, which is how we handle
# the case where nginx cannot pass X-Forwarded-Port to the backend.
_DEFAULT_HTTP_PORTS: frozenset[int] = frozenset({80, 443})


class ProxyingHTTPAdapter(HTTPAdapter):
    """HTTPAdapter that rewrites allowlisted ports and injects an auth header.

    The rewrite is applied at ``send()`` time so the caller's ``PreparedRequest``
    URL is not mutated from the caller's perspective — only the outbound wire
    request hits the proxy port.

    Two rewrite paths:
      - Primary: if the URL port is in ``site_ports``, add ``port_offset``
        and route through the proxy.
      - Redirect bounce: upstream apps (e.g. GitLab ``external_url``,
        Magento ``base_url``) generate absolute redirect Locations that
        omit the port or bake in the internal port. When we see a request
        for a hostname we've already proxied, with either no port or a
        non-allowlisted default port, we redirect through the same proxy
        port observed last for that hostname.
    """

    def __init__(
        self,
        *,
        token: str,
        port_offset: int,
        site_ports: frozenset[int],
        header_name: str = "X-Worldsim-Token",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._token = token
        self._port_offset = port_offset
        self._site_ports = site_ports
        self._header_name = header_name
        # Hostname -> most recently used proxy port. Populated as we send
        # through the proxy so that redirect Locations like
        # ``http://<host>/`` can be bounced back to the same proxy port.
        self._last_proxy_port_by_host: dict[str, int] = {}

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        rewritten_url = self._maybe_rewrite_url(request.url or "")
        if rewritten_url is not None:
            request.url = rewritten_url
            # Host header must match the rewritten netloc so nginx's
            # server_name matching + logging line up.
            parts = urlsplit(rewritten_url)
            if parts.netloc:
                request.headers["Host"] = parts.netloc
            if parts.hostname and parts.port:
                self._last_proxy_port_by_host[parts.hostname] = parts.port
            if self._token:
                request.headers[self._header_name] = self._token
        return super().send(request, **kwargs)

    def _maybe_rewrite_url(self, url: str) -> str | None:
        if not url:
            return None
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return None
        hostname = parts.hostname or ""
        port = parts.port
        new_port: int | None = None
        if port is not None and port in self._site_ports:
            new_port = port + self._port_offset
        elif (
            port is None or port in _DEFAULT_HTTP_PORTS
        ) and hostname in self._last_proxy_port_by_host:
            # Bare-host redirect from an upstream that emitted its own
            # external_url / base_url: reuse the proxy port from the last
            # proxied request to the same hostname.
            new_port = self._last_proxy_port_by_host[hostname]
        if new_port is None:
            return None
        new_netloc = f"{hostname}:{new_port}"
        if parts.username:
            auth = parts.username
            if parts.password:
                auth += f":{parts.password}"
            new_netloc = f"{auth}@{new_netloc}"
        return urlunsplit(parts._replace(netloc=new_netloc))


def make_proxied_session(
    *,
    token: str,
    port_offset: int,
    site_ports: Iterable[int],
) -> requests.Session:
    """Construct a standalone proxied ``requests.Session``.

    Primarily useful for tests that want to route a single known session
    through the proxy without affecting process-wide behaviour. Most test
    wiring should prefer :func:`install_proxy`, which makes every
    ``requests.Session()`` proxied for the duration of the test session.
    """
    adapter = ProxyingHTTPAdapter(
        token=token,
        port_offset=port_offset,
        site_ports=frozenset(int(p) for p in site_ports),
    )
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def install_proxy(
    *,
    token: str,
    port_offset: int,
    site_ports: Iterable[int],
) -> Callable[[], None]:
    """Patch ``requests.Session.__init__`` so all new sessions proxy.

    Returns an ``uninstall`` callable that restores the original
    ``Session.__init__``. The patch is idempotent per-session: mounting the
    adapter over both ``http://`` and ``https://`` replaces any previously
    mounted adapter, but calling ``install_proxy`` twice without uninstalling
    would nest the patches. Integration-test wiring uses a session-scoped
    pytest fixture to enforce install/uninstall symmetry.
    """
    frozen_ports = frozenset(int(p) for p in site_ports)
    original_init = requests.Session.__init__

    def _patched_init(self: requests.Session, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        adapter = ProxyingHTTPAdapter(
            token=token,
            port_offset=port_offset,
            site_ports=frozen_ports,
        )
        self.mount("http://", adapter)
        self.mount("https://", adapter)

    requests.Session.__init__ = _patched_init  # type: ignore[method-assign]

    def _uninstall() -> None:
        requests.Session.__init__ = original_init  # type: ignore[method-assign]

    return _uninstall
