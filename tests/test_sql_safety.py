from __future__ import annotations

import types

import pytest

import worldsim.rewards as rewards
from worldsim.rewards import _db_query_match, _validate_read_only_query
from worldsim.seeding import _validate_seed_sql


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO products (id, name) VALUES (1, 'demo')",
        "UPDATE products SET description = 'demo' WHERE product_id = 1",
    ],
)
def test_seed_sql_allows_targeted_insert_and_update(statement):
    _validate_seed_sql(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE products",
        "UPDATE products SET description = 'demo'",
        "INSERT INTO products VALUES (1); DELETE FROM products",
    ],
)
def test_seed_sql_rejects_destructive_or_broad_statements(statement):
    with pytest.raises(ValueError):
        _validate_seed_sql(statement)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "WITH counts AS (SELECT 1) SELECT * FROM counts",
    ],
)
def test_read_only_query_allows_selects(query):
    _validate_read_only_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM products",
        "SELECT 1; UPDATE products SET name = 'oops'",
        "SET search_path TO public",
        "SELECT * INTO backup_table FROM products",
        "SELECT * FROM products INTO OUTFILE '/tmp/leak'",
        "SELECT * FROM products FOR UPDATE",
    ],
)
def test_read_only_query_rejects_write_capable_sql(query):
    with pytest.raises(ValueError):
        _validate_read_only_query(query)


def test_db_query_match_fails_when_read_only_guard_cannot_be_enabled(monkeypatch):
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query):
            return None

        def fetchall(self):
            return [(1,)]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        rewards,
        "_configure_read_only_connection",
        lambda conn, scheme: (_ for _ in ()).throw(RuntimeError("guard failed")),
    )
    monkeypatch.setattr(
        rewards,
        "pymysql",
        types.SimpleNamespace(connect=lambda **kwargs: FakeConnection()),
        raising=False,
    )
    monkeypatch.setitem(__import__("sys").modules, "pymysql", types.SimpleNamespace(connect=lambda **kwargs: FakeConnection()))

    passed, message = _db_query_match(
        {"query": "SELECT 1", "expected": None},
        {"db_connection": "mysql://user:pass@localhost:3306/db"},
    )

    assert passed is False
    assert "guard failed" in message
