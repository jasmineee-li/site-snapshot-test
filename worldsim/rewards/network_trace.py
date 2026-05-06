from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, unquote_plus, urlencode, urlparse, urlunparse

from worldsim.placeholders import apply_placeholders, placeholder_for_site


def _network_event_url(event: dict[str, Any]) -> str:
    url = event.get("url")
    if isinstance(url, str):
        return url
    request = event.get("request")
    if isinstance(request, dict) and isinstance(request.get("url"), str):
        return str(request["url"])
    return ""


def _network_event_url_candidates(event: dict[str, Any], instance: dict[str, Any]) -> list[str]:
    raw_url = _network_event_url(event)
    if not raw_url:
        return []
    candidates = [raw_url]
    canonical_url = _canonicalize_network_event_origin(raw_url, instance)
    if canonical_url and canonical_url not in candidates:
        candidates.append(canonical_url)
    return candidates


def _canonicalize_network_event_origin(url: str, instance: dict[str, Any]) -> str:
    """Map captured browser origins through host-declared same-site rewrites.

    r5 browser traffic can be rewritten from benchmark-canonical origins to
    bound container origins. Rewards are compiled against the bound origin, so
    matching accepts the declared alias while preserving path/query/body checks.
    """
    rewrites = instance.get("url_origin_rewrites")
    if not isinstance(rewrites, Mapping):
        return url
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    target = rewrites.get(origin)
    if not isinstance(target, str) or not target.strip():
        return url
    target_parsed = urlparse(target.strip().rstrip("/"))
    if not target_parsed.scheme or not target_parsed.netloc:
        return url
    return urlunparse(
        (
            target_parsed.scheme,
            target_parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _network_event_method(event: dict[str, Any]) -> str:
    method = event.get("method")
    if isinstance(method, str):
        return method.upper()
    request = event.get("request")
    if isinstance(request, dict) and isinstance(request.get("method"), str):
        return str(request["method"]).upper()
    return "GET"


def _expected_http_methods(raw: Any) -> set[str]:
    if isinstance(raw, list):
        methods = {str(item).upper() for item in raw if str(item).strip()}
    else:
        methods = {str(raw or "GET").upper()}
    return methods or {"GET"}


def _network_event_status(event: dict[str, Any]) -> int | None:
    raw = event.get("response_status")
    if raw is None:
        response = event.get("response")
        raw = response.get("status") if isinstance(response, dict) else None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _network_event_headers(event: dict[str, Any]) -> dict[str, str]:
    headers = event.get("request_headers") or event.get("headers")
    request = event.get("request")
    if headers is None and isinstance(request, dict):
        headers = request.get("headers")
    out: dict[str, str] = {}
    if isinstance(headers, dict):
        for key, value in headers.items():
            out[str(key).lower()] = str(value)
    elif isinstance(headers, list):
        for item in headers:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str):
                    out[name.lower()] = str(item.get("value", ""))
    return out


def _network_event_post_text(event: dict[str, Any]) -> str:
    raw = event.get("post_data")
    if raw is not None:
        return raw if isinstance(raw, str) else str(raw)
    request = event.get("request")
    if isinstance(request, dict):
        post_data = request.get("postData")
        if isinstance(post_data, dict):
            text = post_data.get("text")
            if isinstance(text, str):
                return text
            params = post_data.get("params")
            if isinstance(params, list):
                pairs: list[tuple[str, str]] = []
                for item in params:
                    if not isinstance(item, Mapping):
                        continue
                    name = item.get("name")
                    if not isinstance(name, str):
                        continue
                    value = item.get("value", "")
                    pairs.append((name, str(value)))
                if pairs:
                    return urlencode(pairs)
            return ""
        if post_data is not None:
            return str(post_data)
    return ""


def _network_event_expected_fields_match(
    event: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[bool, str]:
    if "response_status" in expected:
        if not _status_matches(_network_event_status(event), expected.get("response_status")):
            return False, "response_status mismatch"
    if "query_params" in expected:
        if not _query_params_match(_network_event_url(event), expected.get("query_params")):
            return False, "query_params mismatch"
    if "post_data_contains" in expected:
        if not _post_data_contains_match(
            _network_event_post_text(event), expected.get("post_data_contains")
        ):
            return False, "post_data_contains mismatch"
    if "post_data" in expected:
        if not _post_data_mapping_matches(
            _network_event_post_text(event),
            _network_event_headers(event),
            expected.get("post_data"),
        ):
            return False, "post_data mismatch"
    if "method_requirements" in expected:
        ok, reason = _method_requirements_match(event, expected.get("method_requirements"))
        if not ok:
            return False, reason
    return True, "ok"


def _method_requirements_match(
    event: dict[str, Any],
    requirements: Any,
) -> tuple[bool, str]:
    if not isinstance(requirements, Mapping):
        return False, "method_requirements must be an object"
    method = _network_event_method(event)
    requirement = requirements.get(method)
    if requirement is None:
        requirement = requirements.get(method.lower())
    if requirement is None:
        return True, "ok"
    if not isinstance(requirement, Mapping):
        return False, f"method_requirements.{method} must be an object"
    unsupported = set(requirement) - {"post_data", "post_data_contains"}
    if unsupported:
        return (
            False,
            f"method_requirements.{method} unsupported fields: "
            + ", ".join(sorted(str(key) for key in unsupported)),
        )
    if "post_data_contains" in requirement and not _post_data_contains_match(
        _network_event_post_text(event),
        requirement.get("post_data_contains"),
    ):
        return False, f"method_requirements.{method}.post_data_contains mismatch"
    if "post_data" in requirement and not _post_data_mapping_matches(
        _network_event_post_text(event),
        _network_event_headers(event),
        requirement.get("post_data"),
    ):
        return False, f"method_requirements.{method}.post_data mismatch"
    return True, "ok"


def _status_matches(actual: int | None, expected: Any) -> bool:
    if actual is None:
        return False
    if isinstance(expected, int):
        return actual == expected
    if isinstance(expected, str):
        value = expected.strip().lower()
        if re.fullmatch(r"\dxx", value):
            return actual // 100 == int(value[0])
        try:
            return actual == int(value)
        except ValueError:
            return False
    if isinstance(expected, list):
        return any(_status_matches(actual, item) for item in expected)
    if isinstance(expected, dict):
        minimum = expected.get("min")
        maximum = expected.get("max")
        if isinstance(minimum, int) and actual < minimum:
            return False
        if isinstance(maximum, int) and actual > maximum:
            return False
        return isinstance(minimum, int) or isinstance(maximum, int)
    return False


def _query_params_match(url: str, expected: Any) -> bool:
    if not isinstance(expected, dict):
        return False
    actual = parse_qs(urlparse(url).query, keep_blank_values=True)
    return _mapping_subset_matches(actual, expected)


def _post_data_contains_match(post_text: str, expected: Any) -> bool:
    if isinstance(expected, str):
        needles = [expected]
    elif isinstance(expected, list):
        needles = [str(item) for item in expected if isinstance(item, (str, int, float))]
    else:
        return False
    if not any(needle for needle in needles):
        return False
    decoded = unquote_plus(post_text)
    return all(needle in post_text or needle in decoded for needle in needles if needle)


def _post_data_mapping_matches(post_text: str, headers: dict[str, str], expected: Any) -> bool:
    if not isinstance(expected, dict):
        return False
    parsed = _parse_post_data(post_text, headers)
    return _mapping_subset_matches(parsed, expected)


def _parse_post_data(post_text: str, headers: dict[str, str]) -> dict[str, Any]:
    if not post_text or post_text == "<redacted>":
        return {}
    content_type = headers.get("content-type", "").lower()
    stripped = post_text.strip()
    if "json" in content_type or stripped.startswith(("{", "[")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    if "multipart/form-data" in content_type:
        return _parse_multipart_like(post_text)
    parsed = parse_qs(post_text, keep_blank_values=True)
    if parsed:
        return parsed
    return {"": post_text}


def _parse_multipart_like(post_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    current_name: str | None = None
    current_value: list[str] = []
    for line in post_text.splitlines():
        match = re.search(r'name="([^"]+)"', line)
        if match:
            if current_name is not None:
                out[current_name] = "\n".join(current_value).strip("\r\n")
            current_name = match.group(1)
            current_value = []
            continue
        if current_name is not None:
            if line.startswith("--"):
                out[current_name] = "\n".join(current_value).strip("\r\n")
                current_name = None
                current_value = []
            elif not line.lower().startswith("content-"):
                current_value.append(line)
    if current_name is not None:
        out[current_name] = "\n".join(current_value).strip("\r\n")
    return out


def _mapping_subset_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for expected_key, expected_value in expected.items():
        actual_present, actual_value = _actual_mapping_value(actual, str(expected_key))
        if not actual_present:
            return False
        if not _value_matches(actual_value, expected_value):
            return False
    return True


def _actual_mapping_value(actual: dict[str, Any], expected_key: str) -> tuple[bool, Any]:
    if expected_key in actual:
        return True, actual[expected_key]
    if expected_key.startswith("^") or expected_key.endswith("$") or ".*" in expected_key:
        try:
            pattern = re.compile(expected_key)
        except re.error:
            return False, None
        for key, value in actual.items():
            if pattern.search(str(key)):
                return True, value
    return False, None


def _value_matches(actual: Any, expected: Any) -> bool:
    actual_values = actual if isinstance(actual, list) else [actual]
    actual_strings = [str(item) for item in actual_values]
    if isinstance(expected, dict):
        if "equals" in expected:
            return any(item == str(expected["equals"]) for item in actual_strings)
        if "contains" in expected:
            needle = str(expected["contains"])
            return any(needle in item for item in actual_strings)
        if "regex" in expected:
            try:
                pattern = re.compile(str(expected["regex"]))
            except re.error:
                return False
            return any(pattern.search(item) for item in actual_strings)
        return False
    if isinstance(expected, list):
        return all(_value_matches(actual, item) for item in expected)
    needle = str(expected)
    return any(item == needle or needle in item for item in actual_strings)


def _resolve_url_placeholders(url: str, instance: dict) -> str:
    """Replace __SITE__ placeholders in URLs with actual instance URLs.

    For multi-site tasks, ``instance["url_placeholders"]`` provides the
    authoritative mapping of all site tokens to their real URLs.  For the
    common single-site case the instance's own ``site_url`` is used as a
    fallback for any placeholder not in the explicit mapping.
    """
    explicit = dict(instance.get("url_placeholders", {}))
    primary_placeholder = placeholder_for_site(instance.get("site_name", ""))
    site_url = instance.get("site_url", "")
    if primary_placeholder and primary_placeholder not in explicit and site_url:
        explicit[primary_placeholder] = site_url
    return apply_placeholders(url, explicit, strict=True)


def _url_matches(expected: str, actual: str) -> bool:
    """Check if expected URL matches actual URL without substring over-credit."""
    if not expected or not actual:
        return False

    expected = expected.rstrip("/")
    actual = actual.rstrip("/")

    if _looks_like_regex_url(expected):
        expected = _escape_regex_url_origin(expected)
        regex_target = actual
        parsed_expected = urlparse(expected)
        if expected.startswith("/") and not parsed_expected.netloc:
            parsed_actual = urlparse(actual)
            regex_target = parsed_actual.path or "/"
            if parsed_actual.query:
                regex_target = f"{regex_target}?{parsed_actual.query}"
        try:
            return re.fullmatch(expected, regex_target) is not None
        except re.error:
            return False

    if expected == actual:
        return True

    exp_parsed = urlparse(expected)
    act_parsed = urlparse(actual)
    if exp_parsed.path and not exp_parsed.netloc:
        expected_path = exp_parsed.path.rstrip("/") or "/"
        actual_path = act_parsed.path.rstrip("/") or "/"
        if expected_path != actual_path:
            return False
        if exp_parsed.query:
            return exp_parsed.query == act_parsed.query
        return True

    return False


def _looks_like_regex_url(expected: str) -> bool:
    return expected.startswith("^") or expected.endswith("$") or ".*" in expected


def _escape_regex_url_origin(expected: str) -> str:
    match = re.match(r"^(\^?)(https?://[^/?#]+)(.*)$", expected)
    if not match:
        return expected
    prefix, origin, suffix = match.groups()
    return f"{prefix}{re.escape(origin)}{suffix}"
