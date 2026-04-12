from __future__ import annotations

import pytest

from worldsim.rewards import _validate_read_only_query
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
    ],
)
def test_read_only_query_rejects_write_capable_sql(query):
    with pytest.raises(ValueError):
        _validate_read_only_query(query)
