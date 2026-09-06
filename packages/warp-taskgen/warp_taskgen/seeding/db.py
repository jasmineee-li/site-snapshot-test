"""Seed-time database connection helpers."""

from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

from warp_taskgen.db_urls import parse_supported_db_connection

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _parse_runtime_db_connection(
    db_connection: Any,
    *,
    purpose: str,
) -> urllib.parse.ParseResult:
    try:
        return parse_supported_db_connection(db_connection, purpose=purpose)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _resolve_token_source_path(token_source: str) -> Path:
    path = Path(token_source).expanduser().resolve(strict=False)
    allowed_roots = _allowed_token_source_roots()
    if not any(path.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError(
            "token_source must be under one of: "
            + ", ".join(str(root) for root in sorted(allowed_roots))
        )
    return path


def _allowed_token_source_roots() -> set[Path]:
    roots = {(Path.cwd() / "logs" / "phase_0d").resolve(strict=False)}
    state_dir = os.environ.get("WORLDSIM_STATE_DIR")
    if state_dir:
        roots.add((Path(state_dir).expanduser() / "phase_0d").resolve(strict=False))
    return roots


def _connect_db(parsed: urllib.parse.ParseResult) -> Any:
    scheme = parsed.scheme.lower()
    if scheme == "mysql":
        import pymysql

        return pymysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=(parsed.path or "").lstrip("/"),
        )
    if scheme in ("postgresql", "postgres"):
        import psycopg2

        return psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            dbname=(parsed.path or "").lstrip("/"),
        )
    raise RuntimeError(f"unsupported DB dialect for HTTP seed verification: {scheme}")


def _configure_read_only_connection(conn: Any, scheme: str) -> None:
    try:
        if hasattr(conn, "autocommit"):
            conn.autocommit = False
        with conn.cursor() as cursor:
            if scheme == "mysql":
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
                cursor.execute("START TRANSACTION READ ONLY")
            elif scheme in ("postgresql", "postgres"):
                cursor.execute("BEGIN")
                cursor.execute("SET TRANSACTION READ ONLY")
            else:
                raise RuntimeError(f"unsupported DB dialect: {scheme}")
    except Exception as exc:
        raise RuntimeError("could not enable read-only transaction guard") from exc


def _quote_identifier(identifier: str, scheme: str) -> str:
    if not _IDENTIFIER_PATTERN.match(identifier):
        raise RuntimeError(f"invalid SQL identifier {identifier!r}")
    quote = "`" if scheme == "mysql" else '"'
    return ".".join(f"{quote}{part}{quote}" for part in identifier.split("."))


__all__ = [
    "_allowed_token_source_roots",
    "_configure_read_only_connection",
    "_connect_db",
    "_parse_runtime_db_connection",
    "_quote_identifier",
    "_resolve_token_source_path",
]
