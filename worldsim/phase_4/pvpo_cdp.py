"""Compatibility helpers for PVPO CDP calls.

PVPO originally targeted Playwright CDP sessions exposing
``send(method, params)``. Browser-Use 0.12.6 instead exposes a typed
``cdp_client.send.<Domain>.<method>(..., session_id=...)`` surface. This
module normalizes those session types so the PVPO code can issue raw CDP
commands without depending on one browser library's transport shape.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any


@dataclass(frozen=True)
class CDPSessionAdapter:
    """Normalize Browser-Use or Playwright CDP sessions to ``send(method, params)``."""

    session: Any
    thread_sync_send: bool = False

    async def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        direct_send = getattr(self.session, "send", None)
        if callable(direct_send):
            if self.thread_sync_send:
                result = await asyncio.to_thread(direct_send, method, params or {})
            else:
                result = direct_send(method, params or {})
            if isawaitable(result):
                result = await result
            return result if isinstance(result, dict) else {}

        cdp_client = getattr(self.session, "cdp_client", None)
        session_id = getattr(self.session, "session_id", None)
        if cdp_client is None or session_id is None:
            raise TypeError(
                "PVPO expected a CDP session with either send(method, params) or "
                "cdp_client + session_id attributes"
            )

        domain, sep, command = method.partition(".")
        if not sep:
            raise ValueError(f"invalid CDP method {method!r}; expected Domain.method")

        sender_root = getattr(cdp_client, "send", None)
        if sender_root is None:
            raise TypeError("CDP session has no cdp_client.send dispatcher")

        domain_sender = getattr(sender_root, domain, None)
        if domain_sender is None:
            raise AttributeError(f"CDP client has no domain sender {domain!r}")

        command_sender = getattr(domain_sender, command, None)
        if command_sender is None:
            raise AttributeError(f"CDP client has no command sender {method!r}")

        kwargs: dict[str, Any] = {"session_id": session_id}
        if params:
            kwargs["params"] = params
        return await command_sender(**kwargs)


def normalize_cdp_session(session: Any, *, thread_sync_send: bool = False) -> CDPSessionAdapter:
    """Return a thin adapter exposing ``send(method, params)``."""
    return CDPSessionAdapter(session=session, thread_sync_send=thread_sync_send)


async def runtime_evaluate(
    cdp_session: Any,
    expression: str,
    *,
    return_by_value: bool = True,
    await_promise: bool = True,
) -> dict[str, Any]:
    """Evaluate JavaScript directly over CDP."""
    return await normalize_cdp_session(cdp_session).send(
        "Runtime.evaluate",
        {
            "expression": expression,
            "returnByValue": return_by_value,
            "awaitPromise": await_promise,
        },
    )


async def runtime_evaluate_value(
    cdp_session: Any,
    expression: str,
    *,
    await_promise: bool = True,
) -> Any:
    """Evaluate JavaScript and return ``result.value`` or raise on JS errors."""
    raw = await runtime_evaluate(
        cdp_session,
        expression,
        return_by_value=True,
        await_promise=await_promise,
    )
    if "exceptionDetails" in raw:
        raise RuntimeError(f"JavaScript evaluation failed: {raw['exceptionDetails']}")
    return (raw.get("result") or {}).get("value")
