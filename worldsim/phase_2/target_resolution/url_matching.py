"""Phase 2 target resolution url matching."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2.target_resolution._context import install_context
from worldsim.phase_2.target_resolution.types import ResolverContractDriftError

install_context(globals())


def _assert_anchor_contract_conformance(
    record: Mapping[str, Any],
    *,
    benchmark: str = "webarena_verified",
    site: str | None = None,
) -> None:
    kind = record.get("kind")
    if kind is None:
        return  # pending/empty records — nothing to verify yet
    contract = _registry_kind_contract(str(kind), benchmark=benchmark, site=site)
    if not contract.valid_methods:
        raise ResolverContractDriftError(
            f"resolver emitted kind {kind!r} but no editor method addresses "
            f"it in the contract registry. Either add an @editor_method "
            f"with this kind in its `kinds` set, or stop emitting this kind."
        )

def _strip_regex_anchors(url: str) -> str:
    """Normalise an eval URL that may carry `^` / `$` / `.*$` regex anchors."""
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

def _literalize_regex_value(value: str | None) -> str | None:
    """Convert a WebArena-style regex query value into a single literal.

    WebArena URL-match evaluators encode query values as regex alternations
    (e.g. ``^(opened|)$`` for "either ``opened`` or absent"). Forwarding
    those literally to the live API yields HTTP 400 because GitLab/Postmill
    expect concrete enum values. This helper picks a usable literal:

    * Plain literals (no regex metacharacters) pass through unchanged.
    * ``^(a|b|...)$`` (with optional whitespace) → first non-empty literal
      branch (mirrors the URL-match semantics: any branch satisfies the
      evaluator, so we pick one).
    * Anything still containing regex syntax → ``None`` (caller drops it).

    Returning ``None`` for residual regex is deliberate: dashboard listing
    APIs return 400 on unknown enum values, and silently sending a regex
    string masks the bug.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if not _REGEX_META_RE.search(text):
        return text
    inner = text
    if inner.startswith("^"):
        inner = inner[1:]
    if inner.endswith("$"):
        inner = inner[:-1]
    inner = inner.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    candidates = inner.split("|") if "|" in inner else [inner]
    for alt in candidates:
        alt = alt.strip()
        if alt and not _REGEX_META_RE.search(alt):
            return alt
    return None

def _strip_json_suffix(url: str) -> str:
    """Drop a trailing `.json` so UI-form URLs match the HTML-page regex."""
    if url.endswith(".json"):
        return url[: -len(".json")]
    return url

def _normalise_url(url: str, placeholders: Mapping[str, str]) -> str | None:
    """Resolve placeholders, strip regex anchors, drop trailing `.json`.

    Returns None if placeholder expansion would leave unresolved
    ``__FOO__`` tokens — caller decides whether that's L3-pending or an
    outright non-match.
    """
    if not url:
        return None
    stripped = _strip_json_suffix(_strip_regex_anchors(url))
    try:
        return apply_placeholders(stripped, dict(placeholders), strict=True)
    except ValueError:
        return None

def _path_and_query(url: str) -> str:
    """Return just the path+query portion of a URL, so hostname components
    can't leak into ``project_path`` captures via greedy matching."""
    if not url:
        return ""
    if "://" not in url:
        # Bare path (eval URLs sometimes arrive without scheme).
        return url if url.startswith("/") else "/" + url
    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return path

def _is_listing_kind(kind: str) -> bool:
    return kind in {
        "gitlab_search_result",
        "gitlab_dashboard_list",
        "gitlab_snippets_index",
        "gitlab_project_labels",
        "reddit_dashboard_list",
    }

def _disambiguate_root_segment(task: Mapping[str, Any], segment: str) -> str | None:
    """Resolve a bare ``/<segment>`` URL into a gitlab kind.

    Reads ``agent_context.gitlab.{user_handles,group_handles}`` populated
    by Phase 0c handle enrichment (see
    :mod:`worldsim.phases.phase_0c_handle_enrichment`). Returns ``"user"``
    or ``"group"`` for an unambiguous match; ``None`` when the segment is
    in both lists, neither list, or when the agent_context block is
    missing. The resolver does not guess: ambiguous cases fall through to
    ``kind=None`` with a categorized drop reason.
    """
    if not isinstance(segment, str) or not segment:
        return None
    gl = (task.get("agent_context") or {}).get("gitlab")
    if not isinstance(gl, Mapping):
        return None
    raw_users = gl.get("user_handles") or []
    raw_groups = gl.get("group_handles") or []
    users = {str(u).strip() for u in raw_users if isinstance(u, str)}
    groups = {str(g).strip() for g in raw_groups if isinstance(g, str)}
    in_users = segment in users
    in_groups = segment in groups
    if in_users and not in_groups:
        return "user"
    if in_groups and not in_users:
        return "group"
    return None

def _listing_start_url(kind: str, resolved_url: str, fallback_url: str | None) -> str | None:
    if not _is_listing_kind(kind):
        return fallback_url
    path = urlsplit(resolved_url).path or ""
    if path.startswith("/api/"):
        return fallback_url
    return resolved_url

def _match_gitlab(
    url: str,
    task: Mapping[str, Any] | None = None,
) -> tuple[ResourceKind, dict[str, str]] | None:
    path_and_query = _path_and_query(url)
    for kind, pattern in _GITLAB_PATTERNS:
        match = pattern.search(path_and_query)
        if not match:
            continue
        anchors = {k: v for k, v in match.groupdict().items() if v}
        if kind == "gitlab_dashboard_list":
            anchors["dashboard"] = anchors.pop("dash", "")
        if kind == "gitlab_search_result" and "q" in anchors:
            anchors["query"] = anchors.pop("q")
        # The root-segment regex emits "segment"; resolve it to user vs
        # group via the Phase 0c handle lists. Unresolved → continue
        # searching (no other gitlab pattern will match a bare /<word>,
        # so this becomes kind=None).
        if "segment" in anchors:
            segment = anchors.pop("segment")
            resolved = _disambiguate_root_segment(task or {}, segment)
            if resolved == "user":
                return "gitlab_user_profile", {"username": segment}
            if resolved == "group":
                return "gitlab_group", {"group_path": segment}
            continue
        return kind, anchors
    return None

def _match_reddit(url: str) -> tuple[ResourceKind, dict[str, str]] | None:
    path_and_query = _path_and_query(url)
    for kind, pattern in _REDDIT_PATTERNS:
        match = pattern.search(path_and_query)
        if match:
            anchors = {k: v for k, v in match.groupdict().items() if v}
            if kind == "reddit_dashboard_list":
                anchors["dashboard"] = anchors.pop("dash", "")
            return kind, anchors
    return None

def _iter_eval_urls(task: Mapping[str, Any]) -> list[str]:
    """Yield all URLs declared under ``reward_function.eval[].expected``.

    Prefers NetworkEventEvaluator entries first (they name the URL the
    agent must hit); AgentResponseEvaluator entries rarely carry URLs
    but are included as fallback.
    """
    reward = task.get("reward_function") or {}
    evals = reward.get("eval") or []
    ranked: list[tuple[int, str]] = []
    for ev in evals:
        if not isinstance(ev, dict):
            continue
        evaluator = str(ev.get("evaluator") or "")
        priority = 0 if "NetworkEvent" in evaluator else 1
        expected = ev.get("expected") or {}
        raw = expected.get("url") or expected.get("reference_url")
        if raw is None:
            continue
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, list):
            candidates = [c for c in raw if isinstance(c, str)]
        else:
            continue
        for candidate in candidates:
            ranked.append((priority, _url_with_expected_query_params(candidate, expected)))
    ranked.sort(key=lambda pair: pair[0])
    return [url for _, url in ranked]

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
    query = urlencode(merged, doseq=True)
    return urlunsplit(parts._replace(query=query))

def _iter_start_urls(task: Mapping[str, Any]) -> list[str]:
    start = task.get("start_urls") or []
    if isinstance(start, str):
        return [start]
    return [u for u in start if isinstance(u, str)]

def _site_kind_for_task(task: Mapping[str, Any]) -> Literal[gitlab, reddit] | None:
    sites = task.get("sites") or []
    for site in sites:
        if not isinstance(site, str):
            continue
        lower = site.strip().lower()
        if lower == "gitlab":
            return "gitlab"
        if lower == "reddit":
            return "reddit"
    primary = str(task.get("site") or "").strip().lower()
    if primary == "gitlab":
        return "gitlab"
    if primary == "reddit":
        return "reddit"
    return None

def _attach_surfaces_for(
    kind: ResourceKind,
    *,
    benchmark: str = "webarena_verified",
    site: str | None = None,
) -> list[dict[str, Any]]:
    return [
        dict(surface) for surface in _registry_attach_surfaces(kind, benchmark=benchmark, site=site)
    ]

def _normalise_sort_direction(raw: Any) -> str:
    """Map an LLM-emitted sort hint onto GitLab's ``asc|desc`` contract.

    The classifier frequently emits combined values like ``updated_desc``
    or ``created_asc`` modelled on BrowserGym-style sorts; GitLab's API
    splits these into ``order_by=<field>`` + ``sort=<dir>`` and rejects
    the combined form with 400. Preserve the direction only.
    """
    value = str(raw or "").strip().lower()
    if not value:
        return "desc"
    if value in ("asc", "ascending", "ascend"):
        return "asc"
    if value in ("desc", "descending", "descend"):
        return "desc"
    if value.endswith("_asc") or value.startswith("asc_"):
        return "asc"
    if value.endswith("_desc") or value.startswith("desc_"):
        return "desc"
    return "desc"

def _benign_user_handle(task: Mapping[str, Any]) -> str | None:
    agent_ctx = task.get("agent_context") or {}
    auth = agent_ctx.get("authentication") or {}
    creds = auth.get("credentials") or {}
    handle = creds.get("username")
    if isinstance(handle, str) and handle.strip():
        return handle.strip()
    return None

def _reddit_forum_choice_is_ambiguous(task: Mapping[str, Any]) -> bool:
    instruction = str(task.get("instruction") or "").casefold()
    # "Most appropriate forum" tasks ask the agent to choose the forum at
    # runtime. A resolver-selected candidate forum is only a plausible answer,
    # not a forced encounter path for a pre-seeded forum listing row.
    return "most appropriate forum" in instruction

def _gitlab_issue_listing_intent(
    task: Mapping[str, Any],
    *,
    resolved_start: str | None,
    placeholders: Mapping[str, str],
    benchmark: str,
) -> dict[str, Any] | None:
    instruction = str(task.get("instruction") or "")
    lowered = instruction.casefold()
    if "issue" not in lowered or not any(
        phrase in lowered
        for phrase in (
            "issues page",
            "list of open issues",
            "list of closed issues",
            "list of all issues",
            "most recent open issues",
            "opened issues",
            "latest issue",
            "most recent issue",
            "newest issue",
            "seeded issue",
        )
    ):
        return None

    project_path = _project_path_from_gitlab_listing_task(
        instruction,
        resolved_start=resolved_start,
        placeholders=placeholders,
    )
    if not project_path:
        return None

    anchors: dict[str, str] = {"project_path": project_path}
    label_names = _label_names_from_gitlab_issue_listing_instruction(instruction)
    if label_names:
        anchors["label_names"] = label_names
    reconstructed = _reconstruct_start_url_from_anchors(
        "gitlab", "gitlab_search_result", anchors, placeholders
    )
    if reconstructed is None:
        base = (placeholders.get("__GITLAB__") or "").rstrip("/")
        reconstructed = f"{base}/{project_path}/-/issues" if base else resolved_start
    record = {
        "kind": "gitlab_search_result",
        "anchors": anchors,
        "start_url_resolved": reconstructed,
        "attach_surfaces": _attach_surfaces_for(
            "gitlab_search_result", benchmark=benchmark, site="gitlab"
        ),
        "encounter_requirements": _encounter_requirements("gitlab_search_result", task, anchors),
        "layer": "L2",
        "resolver_note": "deterministic_gitlab_issue_listing_intent",
        "skip_l4_expansion": True,
    }
    record.update(_route_evidence_flags("gitlab_search_result", task))
    _assert_anchor_contract_conformance(record, benchmark=benchmark, site="gitlab")
    return record

def _project_path_from_gitlab_listing_task(
    instruction: str,
    *,
    resolved_start: str | None,
    placeholders: Mapping[str, str],
) -> str | None:
    if resolved_start:
        parsed = urlsplit(resolved_start)
        path = parsed.path if parsed.scheme else _path_and_query(resolved_start)
        issue_listing = _ISSUE_LISTING_RE.search(path)
        if issue_listing:
            return _canonicalize_project_path(issue_listing.group("project_path"))
        match = _GITLAB_PROJECT_ROOT_RE.match(path)
        if match and "/-/" not in path:
            return _canonicalize_project_path(match.group("project_path"))
    match = _GITLAB_PROJECT_PATH_IN_TEXT_RE.search(instruction)
    if match:
        return _canonicalize_project_path(match.group("project_path"))
    return None

def _label_names_from_gitlab_issue_listing_instruction(instruction: str) -> str | None:
    lowered = instruction.casefold()
    if "don't have any labels" in lowered or "do not have any labels" in lowered:
        return None
    label_match = re.search(
        r"labels? related to (?P<label>.+?)(?:$|[.])",
        instruction,
        re.IGNORECASE,
    )
    if label_match:
        return label_match.group("label").strip().strip("\"'")
    if "report bugs" in lowered or "opened issues that report bugs" in lowered:
        return "bug"
    if "requesting new features" in lowered:
        return "feature"
    if "opt model related questions" in lowered:
        return "question"
    return None

def _encounter_requirements(
    kind: ResourceKind, task: Mapping[str, Any], anchors: Mapping[str, str]
) -> dict[str, Any]:
    """Populate encounter_requirements per handoff doc §Encounter derivation."""
    requirements: dict[str, Any] = {"viewport_budget_chars": VIEWPORT_BUDGET_CHARS}
    if kind == "gitlab_dashboard_list":
        handle = _benign_user_handle(task)
        if handle:
            requirements["requires_at_mention"] = handle
        requirements["must_appear_on_list"] = True
    elif kind == "gitlab_search_result":
        query = anchors.get("query")
        scope = anchors.get("scope") or "issues"
        if query:
            requirements["requires_search_index"] = {"query": query, "scope": scope}
    elif kind == "reddit_forum":
        requirements["requires_post_sort_order"] = "recent"
        if _reddit_forum_choice_is_ambiguous(task):
            requirements["forum_choice_ambiguous"] = True
        else:
            requirements["must_appear_on_list"] = True
    elif kind == "reddit_dashboard_list":
        handle = _benign_user_handle(task)
        if handle:
            requirements["requires_at_mention"] = handle
    elif kind in ("gitlab_snippets_index", "gitlab_project_labels"):
        # Inline-listing surfaces: the seed's visible artifact must appear
        # on the listing page so the agent encounters it during the
        # benign read.
        requirements["must_appear_on_list"] = True
    return requirements

def _empty_record(reason: str, pending_layer: Literal[L3, L4] | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": None,
        "anchors": {},
        "start_url_resolved": None,
        "attach_surfaces": [],
        "encounter_requirements": {"viewport_budget_chars": VIEWPORT_BUDGET_CHARS},
        "layer": None,
        "reason": reason,
    }
    if pending_layer is not None:
        record["pending_layer"] = pending_layer
    return record

def _clean_project_path(project_path: str) -> str:
    """Strip an L4-prefixed ``localhost:NNNN/`` authority from ``project_path``.

    ``_project_item_to_record`` writes anchors with paths like
    ``localhost:8023/byteblaze/a11y-webring.club`` (the authority from
    the API probe's ``web_url``). For URL reconstruction we want just
    the group-slashed path suffix.
    """
    path = project_path.strip().strip("/")
    if "/" in path and path.split("/", 1)[0].startswith("localhost:"):
        path = path.split("/", 1)[1]
    return path

def _canonicalize_project_path(project_path: str) -> str:
    """Strip host prefix and surrounding slashes from a project_path.

    Returns the bare ``namespace/project`` (or ``namespace/subgroup/project``)
    form expected by GitLab's path-based API endpoints. The L3 LLM
    sometimes emits ``localhost:8023/foo/bar`` because the API probe's
    ``web_url`` puts the authority in the path; ``_anchors_from_gitlab_item``
    propagates that into ``project_path`` anchors. The bare form is what
    ``urllib.parse.quote(path, safe='')`` should percent-encode for
    ``GET /api/v4/projects/:id``.

    Idempotent: already-canonical inputs return unchanged. Empty input
    returns the empty string.
    """
    if not project_path:
        return ""
    path = project_path.strip()
    while True:
        stripped = _HOSTPREFIX_RE.sub("", path, count=1)
        if stripped == path:
            break
        path = stripped
    return path.strip("/")
