"""Phase 2 option_a behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2._context import install_context

install_context(globals())

def _rigorous_option_a_enabled() -> bool:
    env = os.environ.get("WORLDSIM_RIGOROUS_OPTION_A")
    if env is not None:
        return env.strip().lower() in {"true", "1", "yes", "on"}
    return RIGOROUS_OPTION_A_DEFAULT

def _is_option_a_site(task: dict) -> bool:
    """Return True when the task's site falls under the WASP scope.

    Falls back to ``task["site"]`` when ``task["sites"]`` is missing so
    legacy records still classify correctly.
    """
    for key in ("sites", "site"):
        raw = task.get(key)
        if isinstance(raw, str):
            if raw.strip().lower() in _OPTION_A_SITES:
                return True
        elif isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str) and entry.strip().lower() in _OPTION_A_SITES:
                    return True
    return False

def _validate_option_a_placement(plan: dict, task_name: str) -> str | None:
    """Validate Option A placement using the editor-method registry.

    The legacy validator still runs for discrepancy logging, but the
    registry verdict is the production verdict. Placement is no longer a
    prompt-owned contract.
    """
    _normalize_gitlab_project_selector_templates(plan)
    legacy_verdict = _validate_option_a_placement_legacy(plan, task_name)
    new_verdict = _validate_option_a_placement_registry(plan, task_name)
    if legacy_verdict != new_verdict:
        _log_validator_discrepancy(plan, task_name, legacy_verdict, new_verdict)
    return new_verdict

def _normalize_gitlab_project_selector_templates(plan: dict) -> None:
    """Prefer project_path_template when a direct GitLab route lacks project_id."""
    if _site_for_option_a_plan(plan) != "gitlab":
        return
    resource = plan.get("benign_target_resource")
    if not isinstance(resource, dict):
        return
    anchors = resource.get("anchors")
    if not isinstance(anchors, dict) or "project_path" not in anchors:
        return
    benchmark = "webarena_verified"
    try:
        benchmark = _benchmark_for_option_a_plan(plan)
    except ValueError:
        pass
    available = available_tokens_for_kind(
        str(resource.get("kind") or ""),
        anchors,
        benchmark=benchmark,
        site="gitlab",
    )
    has_project_id_token = "{benign_project_id}" in available
    has_project_path_token = "{benign_project_path}" in available
    if not has_project_id_token and not has_project_path_token:
        return
    for seed_key in ("seed_template", "adversarial_data_seed"):
        seed = plan.get(seed_key)
        if not isinstance(seed, dict):
            continue
        editor_calls = seed.get("editor_calls")
        if not isinstance(editor_calls, list):
            continue
        for call in editor_calls:
            if not isinstance(call, dict):
                continue
            method = str(call.get("method") or "")
            if method not in {
                "create_issue",
                "create_issue_description",
                "create_issue_title",
                "create_issue_note",
                "create_mr",
                "create_mr_note",
                "create_repo_file",
            }:
                continue
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            project_id_value = str(args.get("project_id") or "")
            if has_project_id_token and project_id_value.strip().startswith("{"):
                args["project_id"] = "{benign_project_id}"
            elif has_project_path_token and not args.get("project_path_template"):
                args.pop("project_id", None)
                args["project_path_template"] = "{benign_project_path}"
            if method in {"create_issue_note"} and args.get("issue_iid") == "{issue_iid}":
                args["issue_iid"] = "{benign_issue_iid}"
            if method in {"create_mr_note"} and args.get("mr_iid") == "{mr_iid}":
                args["mr_iid"] = "{benign_mr_iid}"

def _validate_option_a_placement_legacy(plan: dict, task_name: str) -> str | None:
    """Legacy Option A validator. Kept during dual-run window; deleted in
    commit 9's post-soak cleanup.

    Rejects plans whose delivery mechanism creates a dangling parent
    artifact (new project/group/forum) or whose child-create method
    (create_issue / create_issue_note / create_mr_note /
    create_submission / create_comment) does not reference the benign
    task's existing resource via a {benign_*} template token.

    Skip-on-reject per the adopted validator semantics: returns a
    structured reason that the caller logs + drops the task.
    """
    resource = plan.get("benign_target_resource")
    if not isinstance(resource, dict) or resource.get("kind") is None:
        return (
            "benign_target_resource is missing or has null kind; no Option A "
            "attach surface exists for this task"
        )

    seed = plan.get("seed_template")
    if not isinstance(seed, dict):
        return "seed_template missing; cannot verify delivery method"
    editor_calls = seed.get("editor_calls")
    if not isinstance(editor_calls, list) or not editor_calls:
        return "seed_template.editor_calls missing or empty"

    for idx, call in enumerate(editor_calls):
        if not isinstance(call, dict):
            return f"seed_template.editor_calls[{idx}] is not an object"
        method = str(call.get("method") or "")
        if method in _OPTION_A_DANGLING_METHODS:
            return (
                f"editor_calls[{idx}].method={method!r} creates a dangling "
                "parent artifact; Option A requires attaching to the existing "
                "benign-task resource"
            )
        if method in _OPTION_A_CHILD_CREATE_METHODS:
            required_arg, required_token = _OPTION_A_CHILD_CREATE_METHODS[method]
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            value = str(args.get(required_arg, ""))
            if not _value_starts_with_token(value, required_token):
                return (
                    f"editor_calls[{idx}].args.{required_arg}={value!r} must "
                    f"reference the benign anchor via a well-formed "
                    f"{required_token} token so the seed attaches to the "
                    "existing resource"
                )
    return None

def _validate_option_a_placement_registry(plan: dict, task_name: str) -> str | None:
    """Contract-driven Option A validator. Reads the editor-method registry.

    Differences vs legacy:

    * Rejects unknown methods — legacy silently passed any method outside
      its two hardcoded sets, letting a typo or invented method leak to
      Phase 2c.
    * Enforces ``SelectorGroup`` OR-logic — at least one project
      identifier arg (``project_id`` | ``project_path_template`` |
      ``project_name_template``) must be populated with a valid token.
      Legacy only enforced the innermost anchor (``issue_iid`` etc.),
      silently accepting a phantom ``{benign_project_id}`` even when
      resolver anchors only carried ``project_path``.
    * Intersects declared tokens with ``available_tokens_for_kind`` —
      rejects tokens the resolver's anchors cannot actually reach. This
      is the "silently empty substitution" failure mode the Phase 2a
      regen is meant to fix.
    """
    resource = plan.get("benign_target_resource")
    if not isinstance(resource, dict) or resource.get("kind") is None:
        return (
            "benign_target_resource is missing or has null kind; no Option A "
            "attach surface exists for this task"
        )

    kind = str(resource.get("kind") or "")
    site = _site_for_option_a_plan(plan)
    try:
        benchmark = _benchmark_for_option_a_plan(plan)
    except ValueError as exc:
        return f"benchmark metadata is invalid for Option A placement: {exc}"
    contract = kind_contract(kind, benchmark=benchmark, site=site)
    if not contract.valid_methods:
        return (
            f"kind={kind!r} is not addressable by any registered editor method "
            f"(kind_not_registered)"
        )

    anchors_raw = resource.get("anchors")
    anchors = anchors_raw if isinstance(anchors_raw, dict) else {}
    available = available_tokens_for_kind(kind, anchors, benchmark=benchmark, site=site)

    seed = plan.get("seed_template")
    if not isinstance(seed, dict):
        return "seed_template missing; cannot verify delivery method"
    editor_calls = seed.get("editor_calls")
    if not isinstance(editor_calls, list) or not editor_calls:
        return "seed_template.editor_calls missing or empty"

    for idx, call in enumerate(editor_calls):
        if not isinstance(call, dict):
            return f"seed_template.editor_calls[{idx}] is not an object"
        method = str(call.get("method") or "")
        if method not in contract.valid_methods:
            return (
                f"editor_calls[{idx}].method={method!r} is not a valid Option A "
                f"attach for kind={kind!r} (valid: {sorted(contract.valid_methods)})"
            )

        try:
            spec = method_spec(site, method, benchmark=benchmark)
        except KeyError:
            return (
                f"editor_calls[{idx}].method={method!r} is not registered on "
                f"benchmark={benchmark!r}, site={site!r}"
            )

        args_raw = call.get("args")
        args = args_raw if isinstance(args_raw, dict) else {}
        violation = _check_spec_bindings(idx, spec, args, available)
        if violation is not None:
            return violation

    return None

def _site_for_option_a_plan(plan: dict) -> str:
    for key in ("sites", "site"):
        raw = plan.get(key)
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in _OPTION_A_SITES:
                return s
        elif isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str):
                    s = entry.strip().lower()
                    if s in _OPTION_A_SITES:
                        return s
    return ""

def _benchmark_for_option_a_plan(plan: dict) -> str:
    try:
        benchmark = infer_benchmark_name(_benchmark_values_from_record(plan))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return benchmark or "webarena_verified"

def _check_spec_bindings(
    idx: int,
    spec: Any,
    args: dict,
    available: frozenset[str],
) -> str | None:
    # Group selector bindings by their selector_group name.
    groups: dict[str, list[tuple[str, BindingSpec]]] = {}
    for arg, binding in spec.bindings.items():
        if binding.kind == "selector":
            groups.setdefault(binding.selector_group or "", []).append((arg, binding))

    # Each selector group: require ≥1 populated member whose value starts
    # with one of the usable tokens (declared ∩ available). Unpopulated
    # members of a group are fine; that's the whole point of OR-logic.
    for group_name, members in groups.items():
        any_required = any(b.required for _, b in members)
        if not any_required:
            continue
        if not _selector_group_satisfied(members, args, available):
            names = sorted(a for a, _ in members)
            return (
                f"editor_calls[{idx}] selector group {group_name!r} unsatisfied: "
                f"at least one of {names} must be populated with a valid "
                f"{{benign_*}} token reachable via anchors "
                f"(available: {sorted(available)})"
            )

    # Standalone (non-grouped) Token bindings.
    for arg, binding in spec.bindings.items():
        if binding.kind != "token" or binding.selector_group is not None:
            continue
        value = str(args.get(arg, ""))
        if binding.required and not value:
            return f"editor_calls[{idx}] missing required arg {arg!r}"
        if not value or not binding.tokens:
            continue
        usable = binding.tokens & available
        if not any(_value_starts_with_token(value, tok) for tok in usable):
            return (
                f"editor_calls[{idx}].args.{arg}={value!r} must start with one "
                f"of {sorted(binding.tokens)} and that token must be reachable "
                f"via anchors (available: {sorted(available)})"
            )

    return None

def _selector_group_satisfied(
    members: list[tuple[str, BindingSpec]],
    args: dict,
    available: frozenset[str],
) -> bool:
    for arg, binding in members:
        raw = args.get(arg)
        if raw is None or str(raw).strip() == "":
            continue
        value = str(raw)
        if not binding.tokens:
            # Free-text selector member — being populated satisfies the group.
            return True
        usable = binding.tokens & available
        if any(_value_starts_with_token(value, tok) for tok in usable):
            return True
    return False

def _value_starts_with_token(value: str, token: str) -> bool:
    """Check that ``value`` begins with the closed, well-formed form of
    ``token`` (``{benign_<name>}``).

    Permits trailing content (``"{benign_issue_iid}/extra"`` still passes
    when ``token`` is ``"{benign_issue_iid}"`` or the legacy brace-less
    ``"{benign_issue_iid"``) but rejects values that omit the closing
    brace entirely. Rejecting unclosed tokens is necessary because
    seeding.py's ``_FORMAT_TOKEN_PATTERN`` requires the closing brace
    and leaves malformed tokens un-substituted — the literal token
    string then leaks into the seeded payload and breaks Phase 2c
    reachability (observed on 3 reddit tasks in the 0/107 feasibility
    report: ``"{benign_submission_id"`` without ``}``).
    """
    expected = token if token.endswith("}") else token + "}"
    match = _WELL_FORMED_BENIGN_TOKEN_RE.match(value)
    if match is None:
        return False
    return match.group(0) == expected

def _log_validator_discrepancy(
    plan: dict,
    task_name: str,
    legacy_verdict: str | None,
    new_verdict: str | None,
) -> None:
    state_dir = Path(os.environ.get("WORLDSIM_STATE_DIR", "logs"))
    path = state_dir / "phase_2" / "option_a_validator_discrepancy.ndjson"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        seed = plan.get("seed_template") if isinstance(plan.get("seed_template"), dict) else {}
        calls = seed.get("editor_calls") if isinstance(seed.get("editor_calls"), list) else []
        methods = [
            c.get("method")
            for c in calls
            if isinstance(c, dict) and isinstance(c.get("method"), str)
        ]
        resource = plan.get("benign_target_resource")
        kind = resource.get("kind") if isinstance(resource, dict) else None
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "task_name": task_name,
            "legacy_verdict": legacy_verdict,
            "new_verdict": new_verdict,
            "plan_summary": {
                "benign_task_id": plan.get("benign_task_id"),
                "kind": kind,
                "methods": methods,
            },
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.exception("failed to write option_a validator discrepancy NDJSON")

