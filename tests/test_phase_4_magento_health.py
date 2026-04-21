"""Unit tests for worldsim.phase_4.magento_health."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from worldsim.phase_4 import magento_health
from worldsim.phase_4.magento_health import (
    MAGENTO_SITES,
    BaseUrlMismatch,
    assert_base_url,
    check_magento_instances,
    expected_base_url_for_instance,
    probe_base_url,
)


class _StubResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _StubSession:
    def __init__(self, response: _StubResponse | Exception) -> None:
        self._response = response
        self.last_url: str | None = None

    def get(self, url: str, **kwargs):
        self.last_url = url
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


# ---------------------------------------------------------------------------
# probe_base_url
# ---------------------------------------------------------------------------


def test_probe_base_url_parses_single_quoted_magento_declaration():
    html = """<html><head>
    <script>
    var BASE_URL = 'http://3.12.221.9:17770/';
    var require = {"baseUrl":"..."};
    </script></head></html>"""
    session = _StubSession(_StubResponse(200, html))
    assert probe_base_url("http://3.12.221.9:7770/", session=session) == (
        "http://3.12.221.9:17770/"
    )


def test_probe_base_url_returns_none_when_magento_global_absent():
    # 200 OK but no themed Magento response (e.g. a plain HTML admin login).
    session = _StubSession(_StubResponse(200, "<html><body>hi</body></html>"))
    assert probe_base_url("http://3.12.221.9:7770/", session=session) is None


def test_probe_base_url_raises_on_network_error():
    session = _StubSession(requests.ConnectionError("conn refused"))
    with pytest.raises(BaseUrlMismatch) as excinfo:
        probe_base_url("http://3.12.221.9:7770/", session=session)
    assert "failed to fetch" in excinfo.value.args[0]


def test_probe_base_url_raises_on_5xx():
    session = _StubSession(_StubResponse(503, "Service Unavailable"))
    with pytest.raises(BaseUrlMismatch) as excinfo:
        probe_base_url("http://3.12.221.9:7770/", session=session)
    assert "HTTP 503" in excinfo.value.args[0]


def test_probe_base_url_decodes_unicode_escapes():
    # Magento's Luma theme emits the URL with JS ``\uXXXX`` escapes for
    # ``:`` (\u003A) and ``/`` (\u002F). Comparison against the plain-text
    # proxy origin requires decoding.
    html = r"<script>var BASE_URL = 'http\u003A\u002F\u002F3.12.221.9\u003A17770\u002F';</script>"
    session = _StubSession(_StubResponse(200, html))
    assert probe_base_url("http://3.12.221.9:7770/", session=session) == (
        "http://3.12.221.9:17770/"
    )


def test_probe_base_url_accepts_trailing_slash_variation():
    # The probe's target URL normalisation must not double-slash.
    html = "<script>var BASE_URL = 'http://x:17770/';</script>"
    session = _StubSession(_StubResponse(200, html))
    probe_base_url("http://3.12.221.9:7770", session=session)  # no trailing slash
    assert session.last_url == "http://3.12.221.9:7770/"


# ---------------------------------------------------------------------------
# assert_base_url
# ---------------------------------------------------------------------------


def test_assert_base_url_passes_when_actual_equals_expected():
    html = "<script>var BASE_URL = 'http://3.12.221.9:17770/';</script>"
    session = _StubSession(_StubResponse(200, html))
    actual = assert_base_url(
        "shopping",
        "http://3.12.221.9:7770/",
        "http://3.12.221.9:17770/",
        session=session,
    )
    assert actual == "http://3.12.221.9:17770/"


def test_assert_base_url_passes_when_only_trailing_slash_differs():
    html = "<script>var BASE_URL = 'http://3.12.221.9:17770/';</script>"
    session = _StubSession(_StubResponse(200, html))
    # Expected has no trailing slash; normalisation drops it on both sides.
    assert_base_url(
        "shopping",
        "http://3.12.221.9:7770/",
        "http://3.12.221.9:17770",
        session=session,
    )


def test_assert_base_url_raises_on_drift():
    html = "<script>var BASE_URL = 'http://3.12.221.9:7770/';</script>"
    session = _StubSession(_StubResponse(200, html))
    with pytest.raises(BaseUrlMismatch) as excinfo:
        assert_base_url(
            "shopping",
            "http://3.12.221.9:7770/",
            "http://3.12.221.9:17770/",
            session=session,
        )
    err = excinfo.value
    assert err.site_name == "shopping"
    assert err.expected == "http://3.12.221.9:17770/"
    assert err.actual == "http://3.12.221.9:7770/"
    assert "fix_magento_base_url.sh" in str(err)


def test_assert_base_url_raises_when_no_magento_global():
    session = _StubSession(_StubResponse(200, "<html><body>not magento</body></html>"))
    with pytest.raises(BaseUrlMismatch) as excinfo:
        assert_base_url(
            "shopping",
            "http://3.12.221.9:7770/",
            "http://3.12.221.9:17770/",
            session=session,
        )
    assert excinfo.value.actual is None
    assert "could not extract" in str(excinfo.value)


# ---------------------------------------------------------------------------
# expected_base_url_for_instance
# ---------------------------------------------------------------------------


def test_expected_base_url_with_proxy_offsets_port():
    instance = SimpleNamespace(site_url="http://3.12.221.9:7770/")
    proxy = SimpleNamespace(port_offset=10000, scheme="http")
    assert expected_base_url_for_instance(instance, proxy) == "http://3.12.221.9:17770/"


def test_expected_base_url_without_proxy_uses_site_url_origin():
    instance = SimpleNamespace(site_url="http://3.12.221.9:7770/path/ignored")
    assert expected_base_url_for_instance(instance, None) == "http://3.12.221.9:7770/"


def test_expected_base_url_accepts_dict_instance():
    instance = {"site_url": "http://3.12.221.9:7780/"}
    proxy = SimpleNamespace(port_offset=10000, scheme="http")
    assert expected_base_url_for_instance(instance, proxy) == "http://3.12.221.9:17780/"


def test_expected_base_url_zero_port_offset_disables_rewrite():
    instance = SimpleNamespace(site_url="http://3.12.221.9:7770/")
    proxy = SimpleNamespace(port_offset=0, scheme="http")
    assert expected_base_url_for_instance(instance, proxy) == "http://3.12.221.9:7770/"


def test_expected_base_url_empty_site_url_raises():
    with pytest.raises(ValueError):
        expected_base_url_for_instance(SimpleNamespace(site_url=""), None)


def test_expected_base_url_without_port_in_site_url_uses_site_url_origin():
    # Edge case: site_url already on default port, proxy ignored.
    instance = SimpleNamespace(site_url="http://example.com/")
    proxy = SimpleNamespace(port_offset=10000, scheme="http")
    assert expected_base_url_for_instance(instance, proxy) == "http://example.com/"


# ---------------------------------------------------------------------------
# check_magento_instances — only probes magento sites, collects mismatches
# ---------------------------------------------------------------------------


def test_check_magento_instances_skips_non_magento_sites():
    instances = [
        SimpleNamespace(site_name="gitlab", site_url="http://x:8023/"),
        SimpleNamespace(site_name="reddit", site_url="http://x:9999/"),
    ]
    # No sessions constructed because no magento sites to probe.
    mismatches = check_magento_instances(instances, None)
    assert mismatches == []


def test_check_magento_instances_collects_shopping_mismatch():
    proxy = SimpleNamespace(port_offset=10000, scheme="http")
    instances = [
        SimpleNamespace(site_name="shopping", site_url="http://3.12.221.9:7770/"),
    ]
    # Homepage still reports the raw port — simulates a reverted DB.
    html = "<script>var BASE_URL = 'http://3.12.221.9:7770/';</script>"
    session = _StubSession(_StubResponse(200, html))
    mismatches = check_magento_instances(instances, proxy, session=session)
    assert len(mismatches) == 1
    assert mismatches[0].site_name == "shopping"
    assert mismatches[0].expected == "http://3.12.221.9:17770/"


def test_check_magento_instances_passes_when_all_correct():
    proxy = SimpleNamespace(port_offset=10000, scheme="http")
    instances = [
        SimpleNamespace(site_name="shopping", site_url="http://3.12.221.9:7770/"),
        SimpleNamespace(site_name="shopping_admin", site_url="http://3.12.221.9:7780/"),
    ]
    # Single session returns the same HTML for both probes. Because the
    # stub distinguishes by call order only, we patch probe_base_url
    # directly to return the expected origin per call.
    expected_by_host_port = {
        "http://3.12.221.9:7770/": "http://3.12.221.9:17770/",
        "http://3.12.221.9:7780/": "http://3.12.221.9:17780/",
    }

    def _fake_probe(site_url, session=None, timeout=10.0, probe_path="/"):
        return expected_by_host_port[site_url]

    with patch.object(magento_health, "probe_base_url", side_effect=_fake_probe):
        mismatches = check_magento_instances(instances, proxy)
    assert mismatches == []


def test_magento_sites_frozen_set_membership():
    assert "shopping" in MAGENTO_SITES
    assert "shopping_admin" in MAGENTO_SITES
    assert "gitlab" not in MAGENTO_SITES


# ---------------------------------------------------------------------------
# Pending-review backstop (Layer 3)
# ---------------------------------------------------------------------------


from worldsim.phase_4.magento_health import (
    PendingSeedReviewsError,
    check_pending_seed_reviews,
    check_pending_seed_reviews_mysql,
)


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.last_query: str | None = None
        self.last_params: tuple | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params):
        self.last_query = query
        self.last_params = params

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, rows: list[tuple]) -> None:
        self._cursor = _FakeCursor(rows)
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def test_check_pending_clean_returns_none(monkeypatch):
    fake_conn = _FakeConnection(rows=[])
    monkeypatch.setattr(magento_health, "_mysql_connect_from_dsn", lambda dsn: fake_conn)

    result = check_pending_seed_reviews_mysql("shopping", "mysql://u:p@h:3306/db")

    assert result is None
    assert fake_conn.closed
    # Spot-check the query asserts what we expect: the WHERE clause filters
    # status_id=2 (Pending) AND nickname LIKE the seed prefix.
    assert "status_id" in fake_conn._cursor.last_query
    assert "LIKE" in fake_conn._cursor.last_query
    assert fake_conn._cursor.last_params == (2, "SeedNickAdv%")


def test_check_pending_with_rows_returns_error_with_sample_ids(monkeypatch):
    fake_conn = _FakeConnection(
        rows=[
            (101, "SeedNickAdv001"),
            (102, "SeedNickAdv002"),
            (103, "SeedNickAdv003"),
            (104, "SeedNickAdv004"),
            (105, "SeedNickAdv005"),
            (106, "SeedNickAdv006"),
            (107, "SeedNickAdv007"),
        ]
    )
    monkeypatch.setattr(magento_health, "_mysql_connect_from_dsn", lambda dsn: fake_conn)

    result = check_pending_seed_reviews_mysql("shopping", "mysql://u:p@h:3306/db")

    assert isinstance(result, PendingSeedReviewsError)
    assert result.site_name == "shopping"
    assert result.count == 7
    # First 5 review IDs surfaced in the error detail.
    assert result.sample_ids == [101, 102, 103, 104, 105]
    detail = str(result)
    assert "7 pending-approval review" in detail
    assert "SeedNickAdv%" in detail
    assert "[101, 102, 103, 104, 105]" in detail
    assert "force-reverify" in detail


def test_check_pending_propagates_db_errors(monkeypatch):
    """A pymysql connect failure must NOT be silently swallowed — Phase 4
    should fail-loud when the backstop probe itself can't run."""

    class _ConnRefused(Exception):
        pass

    def _fail(_dsn):
        raise _ConnRefused("connection refused")

    monkeypatch.setattr(magento_health, "_mysql_connect_from_dsn", _fail)

    with pytest.raises(_ConnRefused):
        check_pending_seed_reviews_mysql("shopping", "mysql://u:p@h:3306/db")


def test_check_pending_skips_non_magento_sites(monkeypatch):
    called: list[str] = []

    def _fail_if_called(_dsn):
        called.append(_dsn)
        raise AssertionError("non-magento site should never invoke DB connect")

    monkeypatch.setattr(magento_health, "_mysql_connect_from_dsn", _fail_if_called)

    instances = [
        SimpleNamespace(site_name="gitlab", db_connection="postgresql://x"),
        SimpleNamespace(site_name="reddit", db_connection="postgresql://y"),
        SimpleNamespace(site_name="map", db_connection="postgresql://z"),
    ]
    errors = check_pending_seed_reviews(instances)

    assert errors == []
    assert called == []


def test_check_pending_skips_magento_without_db_connection(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(
        magento_health,
        "_mysql_connect_from_dsn",
        lambda _dsn: (_ for _ in ()).throw(AssertionError("should not connect")),
    )

    instances = [SimpleNamespace(site_name="shopping", db_connection=None)]
    with caplog.at_level(logging.WARNING, logger="worldsim.phase_4.magento_health"):
        errors = check_pending_seed_reviews(instances)

    # Backstop is defense-in-depth; missing db_connection is logged but does
    # not block Phase 4 (Layer 2 render check is the primary gate).
    assert errors == []
    assert any("no db_connection" in rec.message for rec in caplog.records)


def test_check_pending_aggregates_errors_across_magento_instances(monkeypatch):
    fake_conn_with_rows = _FakeConnection(rows=[(50, "SeedNickAdv001")])
    fake_conn_clean = _FakeConnection(rows=[])

    conns = iter([fake_conn_with_rows, fake_conn_clean])

    monkeypatch.setattr(magento_health, "_mysql_connect_from_dsn", lambda _dsn: next(conns))

    instances = [
        SimpleNamespace(site_name="shopping", db_connection="mysql://u:p@h:3306/d1"),
        SimpleNamespace(site_name="shopping_admin", db_connection="mysql://u:p@h:3307/d2"),
    ]
    errors = check_pending_seed_reviews(instances)

    assert len(errors) == 1
    assert errors[0].site_name == "shopping"
    assert errors[0].count == 1


def test_check_pending_works_with_dict_instances(monkeypatch):
    """``check_pending_seed_reviews`` must accept dict-shaped instances —
    Phase 4 calls this with the dict form during preflight."""
    fake_conn = _FakeConnection(rows=[])
    monkeypatch.setattr(magento_health, "_mysql_connect_from_dsn", lambda _dsn: fake_conn)

    errors = check_pending_seed_reviews(
        [{"site_name": "shopping", "db_connection": "mysql://u:p@h:3306/d"}]
    )

    assert errors == []
