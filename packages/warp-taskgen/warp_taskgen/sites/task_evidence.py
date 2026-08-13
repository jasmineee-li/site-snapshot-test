"""Task evidence normalization owned by Site Targeting.

These helpers validate and normalize URL-bearing task metadata before a Site
adapter receives it.  They do not choose a target or apply Phase 2 policy.
The historical ``warp_taskgen.sites.catalog`` names are re-exported by the
catalog facade for one compatibility cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from warp_taskgen.placeholders import apply_placeholders


def _strip_regex_anchors(url: str) -> str:
    if not url:
        return ""
    stripped = url.strip()
    if stripped.startswith("^"):
        stripped = stripped[1:]
    if stripped.endswith(".*$"):
        stripped = stripped[:-3]
    elif stripped.endswith("$"):
        stripped = stripped[:-1]
    if stripped.endswith(".*"):
        stripped = stripped[:-2]
    return stripped


def _strip_json_suffix(url: str) -> str:
    return url[: -len(".json")] if url.endswith(".json") else url


def _normalise_url(url: str, placeholders: Mapping[str, str]) -> str | None:
    if not url:
        return None
    stripped = _strip_json_suffix(_strip_regex_anchors(url))
    try:
        return apply_placeholders(stripped, dict(placeholders), strict=True)
    except ValueError:
        return None


def _path_and_query(url: str) -> str:
    if not url:
        return ""
    if "://" not in url:
        return url if url.startswith("/") else "/" + url
    parts = urlsplit(url)
    path = parts.path or "/"
    return f"{path}?{parts.query}" if parts.query else path


def _matches_origin(url: str, origin: str) -> bool:
    """Accept relative evidence or an absolute URL on the bound origin."""

    try:
        candidate = urlsplit(url)
        expected = urlsplit(origin)
    except ValueError:
        return False
    if not candidate.scheme and not candidate.netloc:
        return True
    return (candidate.scheme, candidate.netloc) == (expected.scheme, expected.netloc)


def _url_with_expected_query_params(url: str, expected: Mapping[str, Any]) -> str:
    query_params = expected.get("query_params")
    if not isinstance(query_params, Mapping) or not query_params:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    merged = parse_qs(parts.query, keep_blank_values=True)
    for key, raw in query_params.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(raw, list):
            values = [str(value) for value in raw if value is not None]
        elif raw is None:
            values = []
        else:
            values = [str(raw)]
        if values:
            merged[key] = values
    return urlunsplit(parts._replace(query=urlencode(merged, doseq=True)))


def _iter_eval_urls(task: Mapping[str, Any]) -> list[str]:
    """Return expected URLs with NetworkEvent entries ranked first."""

    reward = task.get("reward_function") or {}
    if not isinstance(reward, Mapping):
        return []
    evals = reward.get("eval") or []
    ranked: list[tuple[int, int, str]] = []
    for sequence, evaluator in enumerate(evals):
        if not isinstance(evaluator, Mapping):
            continue
        name = str(evaluator.get("evaluator") or "")
        priority = 0 if "NetworkEvent" in name else 1
        expected = evaluator.get("expected") or {}
        if not isinstance(expected, Mapping):
            continue
        raw = expected.get("url") or expected.get("reference_url")
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, list):
            candidates = [candidate for candidate in raw if isinstance(candidate, str)]
        else:
            continue
        for candidate in candidates:
            ranked.append(
                (priority, sequence, _url_with_expected_query_params(candidate, expected))
            )
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [url for _, _, url in ranked]


def _iter_start_urls(task: Mapping[str, Any]) -> list[str]:
    start = task.get("start_urls") or []
    if isinstance(start, str):
        return [start]
    if isinstance(start, Sequence):
        return [url for url in start if isinstance(url, str)]
    return []


@dataclass(frozen=True)
class _TaskSiteMetadata:
    """Validated task/delivery identity used by the L1/L2 resolver.

    ``task_site`` is the page/benign-task identity.  ``delivery_site`` is
    deliberately kept separate because a payload may be delivered to another
    Site (for example, an admin task whose mutation is on a storefront).  A
    multi-Site ``sites`` list is only accepted when the explicit ``site`` and
    ``delivery_channel.delivery_site`` explain the additional entry; callers
    must never select the first list item as a guess.
    """

    task_site: str | None
    delivery_site: str | None = None
    failure_reason: str | None = None
    failure_message: str | None = None

    @classmethod
    def failure(cls, reason: str, message: str) -> _TaskSiteMetadata:
        return cls(None, failure_reason=reason, failure_message=message)


_MISSING = object()


def _normalise_site_token(
    value: object,
    *,
    field: str,
    allow_none_token: bool = False,
) -> tuple[str | None, tuple[str, str] | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, ("malformed_site_metadata", f"{field} must be a string")
    token = value.strip().lower()
    if not token:
        return None, ("malformed_site_metadata", f"{field} must not be empty")
    if allow_none_token and token == "none":
        return None, None
    if token == "none":
        return None, ("malformed_site_metadata", f"{field} cannot be 'none'")
    return token, None


def _task_site_metadata(task: Mapping[str, Any]) -> _TaskSiteMetadata:
    """Validate site-bearing task metadata without choosing an arbitrary Site."""

    explicit_site: str | None = None
    if task.get("site", _MISSING) is not _MISSING:
        explicit_site, error = _normalise_site_token(task.get("site"), field="task.site")
        if error is not None:
            return _TaskSiteMetadata.failure(*error)

    sites_value = task.get("sites", _MISSING)
    declared_sites: list[str] = []
    if sites_value is not _MISSING and sites_value is not None:
        if isinstance(sites_value, str):
            sites_iterable: Sequence[object] = (sites_value,)
        elif isinstance(sites_value, Sequence) and not isinstance(sites_value, (bytes, bytearray)):
            sites_iterable = sites_value
        else:
            return _TaskSiteMetadata.failure(
                "malformed_site_metadata", "task.sites must be a string or sequence of strings"
            )
        for index, value in enumerate(sites_iterable):
            token, error = _normalise_site_token(value, field=f"task.sites[{index}]")
            if error is not None:
                return _TaskSiteMetadata.failure(*error)
            if token is not None and token not in declared_sites:
                declared_sites.append(token)

    delivery_site: str | None = None
    delivery = task.get("delivery_channel", _MISSING)
    if delivery is not _MISSING and delivery is not None:
        if not isinstance(delivery, Mapping):
            return _TaskSiteMetadata.failure(
                "malformed_metadata", "task.delivery_channel must be a mapping"
            )
        if "delivery_site" in delivery:
            delivery_site, error = _normalise_site_token(
                delivery.get("delivery_site"),
                field="task.delivery_channel.delivery_site",
                allow_none_token=True,
            )
            if error is not None:
                return _TaskSiteMetadata.failure(*error)

    if explicit_site is not None:
        if declared_sites and explicit_site not in declared_sites:
            return _TaskSiteMetadata.failure(
                "conflicting_site_metadata",
                f"task.site {explicit_site!r} is absent from task.sites {declared_sites!r}",
            )
        extra_sites = set(declared_sites) - {explicit_site}
        if extra_sites and extra_sites != {delivery_site}:
            return _TaskSiteMetadata.failure(
                "ambiguous_site_metadata",
                "task.sites contains additional Sites without a matching delivery_site",
            )
        return _TaskSiteMetadata(explicit_site, delivery_site)

    if len(declared_sites) > 1:
        return _TaskSiteMetadata.failure(
            "ambiguous_site_metadata",
            "task.sites declares multiple Sites but task.site is absent",
        )
    if declared_sites:
        return _TaskSiteMetadata(declared_sites[0], delivery_site)
    if delivery_site is not None:
        return _TaskSiteMetadata.failure(
            "missing_task_site",
            "delivery_site is present but the page/task Site is not declared",
        )
    return _TaskSiteMetadata(None, delivery_site)


def _metadata_failure(task: Mapping[str, Any]) -> tuple[str, str] | None:
    """Validate nested L1/L2 metadata before any adapter can infer a target."""

    site_metadata = _task_site_metadata(task)
    if site_metadata.failure_reason:
        return (
            site_metadata.failure_reason,
            site_metadata.failure_message or "invalid site metadata",
        )

    reward = task.get("reward_function", _MISSING)
    if reward is not _MISSING and reward is not None:
        if not isinstance(reward, Mapping):
            return "malformed_metadata", "task.reward_function must be a mapping"
        evals = reward.get("eval", _MISSING)
        if evals is not _MISSING and evals is not None:
            if not isinstance(evals, Sequence) or isinstance(evals, (str, bytes, bytearray)):
                return "malformed_metadata", "task.reward_function.eval must be a sequence"
            for index, evaluator in enumerate(evals):
                if not isinstance(evaluator, Mapping):
                    return (
                        "malformed_metadata",
                        f"task.reward_function.eval[{index}] must be a mapping",
                    )
                expected = evaluator.get("expected", _MISSING)
                if expected is not _MISSING and expected is not None:
                    if not isinstance(expected, Mapping):
                        return (
                            "malformed_metadata",
                            f"task.reward_function.eval[{index}].expected must be a mapping",
                        )
                    for url_key in ("url", "reference_url"):
                        raw = expected.get(url_key, _MISSING)
                        if raw is _MISSING or raw is None:
                            continue
                        if isinstance(raw, str):
                            continue
                        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                            continue
                        return (
                            "malformed_metadata",
                            f"task.reward_function.eval[{index}].expected.{url_key} "
                            "must be a string or list of strings",
                        )
                    query_params = expected.get("query_params", _MISSING)
                    if (
                        query_params is not _MISSING
                        and query_params is not None
                        and not isinstance(query_params, Mapping)
                    ):
                        return (
                            "malformed_metadata",
                            f"task.reward_function.eval[{index}].expected.query_params "
                            "must be a mapping",
                        )

    start_urls = task.get("start_urls", _MISSING)
    if start_urls is not _MISSING and start_urls is not None:
        if isinstance(start_urls, str):
            pass
        elif isinstance(start_urls, Sequence) and not isinstance(start_urls, (bytes, bytearray)):
            if not all(isinstance(url, str) for url in start_urls):
                return "malformed_metadata", "task.start_urls must contain only strings"
        else:
            return "malformed_metadata", "task.start_urls must be a string or sequence"

    agent_context = task.get("agent_context", _MISSING)
    if agent_context is not _MISSING and agent_context is not None:
        if not isinstance(agent_context, Mapping):
            return "malformed_metadata", "task.agent_context must be a mapping"
        for context_name, site_context in agent_context.items():
            if site_context is None:
                continue
            if not isinstance(site_context, Mapping):
                return (
                    "malformed_metadata",
                    f"task.agent_context.{context_name} must be a mapping",
                )
    return None


def _site_kind_for_task(task: Mapping[str, Any]) -> str | None:
    """Return the explicit page/task Site, never an arbitrary list member."""

    metadata = _task_site_metadata(task)
    return metadata.task_site if metadata.failure_reason is None else None


__all__ = [
    "_iter_eval_urls",
    "_iter_start_urls",
    "_matches_origin",
    "_metadata_failure",
    "_normalise_url",
    "_path_and_query",
    "_site_kind_for_task",
    "_strip_json_suffix",
    "_strip_regex_anchors",
    "_task_site_metadata",
    "_url_with_expected_query_params",
]
