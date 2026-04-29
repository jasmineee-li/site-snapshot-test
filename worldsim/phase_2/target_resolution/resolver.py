"""Phase 2 target resolution public L1/L2 resolver."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2.target_resolution._context import install_context

install_context(globals())

def _reconstruct_start_url_from_anchors(
    site_kind: Literal[gitlab, reddit],
    kind: str,
    anchors: Mapping[str, Any],
    placeholders: Mapping[str, str],
) -> str | None:
    """Build a synthetic-host URL pointing at the concrete resource.

    Returns the canonical URL (on the synthetic ``https://gitlab.local``
    / ``https://reddit.local`` origin drawn from ``placeholders``) when
    the anchors carry enough to address a single entity. Returns ``None``
    when they do not — caller falls back to the raw benign ``start_urls[0]``.

    Added to close the Phase 2c anchor-vs-probe mismatch: the benign
    task's raw ``start_urls`` often points at a project root or bare
    host, but the seed attaches to a concrete issue / MR / submission.
    Without reconstruction the 2c reachability probe navigates to the
    wrong page and the seed witnesses never appear.
    """
    if site_kind == "gitlab":
        origin = placeholders.get("__GITLAB__")
    elif site_kind == "reddit":
        origin = placeholders.get("__REDDIT__")
    else:
        return None
    if not origin:
        return None
    base = origin.rstrip("/")

    if kind == "gitlab_issue":
        project_path = anchors.get("project_path")
        iid = anchors.get("issue_iid")
        if project_path and iid:
            return f"{base}/{_clean_project_path(str(project_path))}/-/issues/{iid}"
        return None
    if kind == "gitlab_mr":
        project_path = anchors.get("project_path")
        iid = anchors.get("mr_iid")
        if project_path and iid:
            return f"{base}/{_clean_project_path(str(project_path))}/-/merge_requests/{iid}"
        return None
    if kind == "gitlab_search_result":
        query = anchors.get("query")
        scope = anchors.get("scope") or "issues"
        project_path = anchors.get("project_path")
        if project_path:
            return f"{base}/{_clean_project_path(str(project_path))}/-/{scope}"
        if query:
            # GitLab accepts either `+` or `%20` for spaces; keep `+` to
            # match the raw eval URLs we parse (``...?search=foo+bar``).
            encoded = urlquote(str(query), safe="+")
            return f"{base}/search?search={encoded}&scope={scope}"
        return None
    if kind == "gitlab_dashboard_list":
        dashboard = anchors.get("dashboard")
        if dashboard:
            return f"{base}/dashboard/{dashboard}"
        return None
    if kind == "gitlab_user_profile":
        username = anchors.get("username")
        if username:
            return f"{base}/{username}"
        return None
    if kind == "gitlab_group":
        group_path = anchors.get("group_path")
        if group_path:
            return f"{base}/{group_path}"
        return None
    if kind == "gitlab_snippet":
        snippet_id = anchors.get("snippet_id")
        if snippet_id:
            return f"{base}/-/snippets/{snippet_id}"
        return None
    if kind == "gitlab_snippets_index":
        return f"{base}/-/snippets"
    if kind == "gitlab_project_milestone":
        project_path = anchors.get("project_path")
        iid = anchors.get("milestone_iid")
        if project_path and iid:
            return f"{base}/{_clean_project_path(str(project_path))}/-/milestones/{iid}"
        return None
    if kind == "gitlab_project_labels":
        project_path = anchors.get("project_path")
        if project_path:
            return f"{base}/{_clean_project_path(str(project_path))}/-/labels"
        return None
    if kind == "reddit_submission":
        forum = anchors.get("forum_name")
        submission_id = anchors.get("submission_id")
        if forum and submission_id:
            return f"{base}/f/{forum}/{submission_id}"
        return None
    if kind == "reddit_forum":
        forum = anchors.get("forum_name")
        if forum:
            return f"{base}/f/{forum}"
        return None
    if kind == "reddit_dashboard_list":
        user = anchors.get("user") or anchors.get("username")
        dashboard = anchors.get("dashboard")
        if user and dashboard:
            return f"{base}/user/{user}/{dashboard}"
        return None
    return None

def derive_benign_target_resource(
    task: Mapping[str, Any],
    placeholders: Mapping[str, str],
    *,
    allow_layers: tuple[Literal[L1, L2, L3, L4], ...] = ("L1", "L2"),
    benchmark: str = "webarena_verified",
) -> dict[str, Any]:
    """Resolve the benign target resource for a Phase 1 task (L1/L2 only).

    Returns a dict matching handoff §Benign-target resource extraction:
    ``{kind, anchors, start_url_resolved, attach_surfaces,
    encounter_requirements, layer, ...}``. When L1+L2 cannot classify
    the task, returns an empty record with ``pending_layer`` set so the
    caller can route to L3 via :func:`resolve_l3` in a later pass.

    L3 is async (it calls the Anthropic Messages API and the live
    benchmark instance), so this sync entrypoint refuses to dispatch
    L3/L4 directly — ``allow_layers`` containing either raises.
    """
    site_kind = _site_kind_for_task(task)
    if site_kind is None:
        return _empty_record("task is not gitlab or reddit (out of WASP scope)", None)

    if "L3" in allow_layers or "L4" in allow_layers:
        raise NotImplementedError(
            "L3 and L4 are async; call resolve_l3() / resolve_l4() explicitly."
        )

    start_urls_raw = _iter_start_urls(task)
    resolved_start: str | None = None
    for url in start_urls_raw:
        resolved = _normalise_url(url, placeholders)
        if resolved:
            resolved_start = resolved
            break

    # L1: parse eval URLs (gold source — NetworkEvent ranked before
    # AgentResponse so the "which URL must the agent hit" signal wins).
    if "L1" in allow_layers:
        for raw in _iter_eval_urls(task):
            resolved = _normalise_url(raw, placeholders)
            if not resolved:
                continue
            hit = (
                _match_gitlab(resolved, task) if site_kind == "gitlab" else _match_reddit(resolved)
            )
            if hit is None:
                continue
            kind, anchors = hit
            reconstructed = _reconstruct_start_url_from_anchors(
                site_kind, kind, anchors, placeholders
            )
            start_url = (
                _listing_start_url(kind, resolved, resolved_start)
                if _is_listing_kind(kind)
                else reconstructed or resolved_start
            )
            record = {
                "kind": kind,
                "anchors": dict(anchors),
                "start_url_resolved": start_url,
                "attach_surfaces": _attach_surfaces_for(kind, benchmark=benchmark, site=site_kind),
                "encounter_requirements": _encounter_requirements(kind, task, anchors),
                "layer": "L1",
            }
            record.update(_route_evidence_flags(kind, task))
            _assert_anchor_contract_conformance(record, benchmark=benchmark, site=site_kind)
            return record

    # L2: parse start_urls directly — applies when eval[] lacks a URL
    # (AgentResponseEvaluator-only retrieve tasks).
    if "L2" in allow_layers and resolved_start:
        hit = (
            _match_gitlab(resolved_start, task)
            if site_kind == "gitlab"
            else _match_reddit(resolved_start)
        )
        if hit is not None:
            kind, anchors = hit
            reconstructed = _reconstruct_start_url_from_anchors(
                site_kind, kind, anchors, placeholders
            )
            start_url = (
                _listing_start_url(kind, resolved_start, resolved_start)
                if _is_listing_kind(kind) and resolved_start is not None
                else reconstructed or resolved_start
            )
            record = {
                "kind": kind,
                "anchors": dict(anchors),
                "start_url_resolved": start_url,
                "attach_surfaces": _attach_surfaces_for(kind, benchmark=benchmark, site=site_kind),
                "encounter_requirements": _encounter_requirements(kind, task, anchors),
                "layer": "L2",
            }
            record.update(_route_evidence_flags(kind, task))
            _assert_anchor_contract_conformance(record, benchmark=benchmark, site=site_kind)
            return record

    if "L2" in allow_layers and site_kind == "gitlab":
        listing_record = _gitlab_issue_listing_intent(
            task,
            resolved_start=resolved_start,
            placeholders=placeholders,
            benchmark=benchmark,
        )
        if listing_record is not None:
            return listing_record

    # Fall-through: bare __GITLAB__ / __REDDIT__ or intent-only task.
    # L3 owns these: LLM intent parse + live API probe. Signal pending
    # so the caller (Phase 2a) routes this task's target derivation to
    # the L3 pass once it lands.
    record = _empty_record(
        "L1+L2 found no concrete resource; intent-only task pending L3",
        pending_layer="L3",
    )
    record["start_url_resolved"] = resolved_start
    return record
