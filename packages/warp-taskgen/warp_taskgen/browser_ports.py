"""Browser-facing port validation helpers."""

from __future__ import annotations

from urllib.parse import urlsplit

# Chromium blocks navigation to these ports to prevent cross-protocol abuse.
# Keep this list intentionally local and explicit so benchmark topology
# generation can fail before a browser run spends tasks on unreachable origins.
CHROMIUM_RESTRICTED_PORTS: frozenset[int] = frozenset(
    {
        1,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        20,
        21,
        22,
        23,
        25,
        37,
        42,
        43,
        53,
        69,
        77,
        79,
        87,
        95,
        101,
        102,
        103,
        104,
        109,
        110,
        111,
        113,
        115,
        117,
        119,
        123,
        135,
        137,
        139,
        143,
        161,
        179,
        389,
        427,
        465,
        512,
        513,
        514,
        515,
        526,
        530,
        531,
        532,
        540,
        548,
        554,
        556,
        563,
        587,
        601,
        636,
        989,
        990,
        993,
        995,
        1719,
        1720,
        1723,
        2049,
        3659,
        4045,
        5060,
        5061,
        6000,
        6566,
        6665,
        6666,
        6667,
        6668,
        6669,
        6697,
        10080,
    }
)


def is_chromium_restricted_port(port: int | None) -> bool:
    return port in CHROMIUM_RESTRICTED_PORTS


def chromium_restricted_port_for_url(raw_url: str) -> int | None:
    parsed = urlsplit(raw_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid URL port in {raw_url!r}") from exc
    if port is None:
        return None
    return port if is_chromium_restricted_port(port) else None
