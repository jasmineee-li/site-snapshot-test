"""Phase 4 payload text behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context

install_context(globals())


def _selected_rendered_payload(task: dict[str, Any]) -> str | None:
    """Return the rendered payload chosen by ``selected_payload_index``."""
    payload_texts = task.get("payload_texts")
    if not isinstance(payload_texts, list) or not payload_texts:
        return None
    selected_index = task.get("selected_payload_index", 0)
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(payload_texts)):
        logger.warning(
            "Phase 4 task %s has invalid selected_payload_index=%r; "
            "not falling back to payload_texts[0]",
            task.get("id", "unknown"),
            selected_index,
        )
        return None
    payload = payload_texts[selected_index]
    if not isinstance(payload, dict):
        return None
    rendered = payload.get("rendered_payload")
    return rendered if isinstance(rendered, str) and rendered else None


class _PayloadViewHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._hidden_stack: list[bool] = []
        self.visible_chunks: list[str] = []
        self.hidden_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        parent_hidden = self._hidden_stack[-1] if self._hidden_stack else False
        style = str(attrs_dict.get("style") or "").lower().replace(" ", "")
        current_hidden = parent_hidden or (
            "position:absolute" in style and ("left:-" in style or "right:-" in style)
        )
        self._hidden_stack.append(current_hidden)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._hidden_stack:
            self._hidden_stack.pop()

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self._hidden_stack and self._hidden_stack[-1]:
            self.hidden_chunks.append(data)
        else:
            self.visible_chunks.append(data)


def _payload_text_views(rendered_payload: str, concealment: str) -> tuple[str, str]:
    if concealment == "markdown_fenced_system":
        concealed_blocks = [
            match.group(1) for match in _MARKDOWN_SYSTEM_BLOCK_RE.finditer(rendered_payload)
        ]
        visible = _MARKDOWN_SYSTEM_BLOCK_RE.sub("", rendered_payload)
        return (visible, "\n".join(concealed_blocks))
    if concealment == "offscreen_css":
        parser = _PayloadViewHTMLParser()
        parser.feed(rendered_payload)
        return ("".join(parser.visible_chunks), "".join(parser.hidden_chunks))
    return (rendered_payload, rendered_payload)


def _seed_payload_candidates_from_mapping(
    mapping: dict[str, Any],
    *,
    body_field: str | None,
) -> list[str]:
    candidates: list[str] = []
    if body_field and isinstance(mapping.get(body_field), str) and mapping[body_field].strip():
        candidates.append(str(mapping[body_field]))
    dynamic_field = mapping.get("field")
    if isinstance(dynamic_field, str) and dynamic_field.strip():
        dynamic_field = dynamic_field.strip()
        if isinstance(mapping.get("value"), str) and mapping["value"].strip():
            if body_field is None or dynamic_field == body_field:
                candidates.append(str(mapping["value"]))
    for field_name in _PAYLOAD_BODY_FIELD_HINTS:
        value = mapping.get(field_name)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    return candidates


def _extract_variant_rendered_payload(task: dict[str, Any], seed: dict[str, Any]) -> str | None:
    delivery_channel = task.get("delivery_channel")
    body_field = None
    if isinstance(delivery_channel, dict):
        raw_body_field = delivery_channel.get("body_field")
        if isinstance(raw_body_field, str) and raw_body_field.strip():
            body_field = raw_body_field.strip()

    seen: set[str] = set()

    def consider(value: str) -> str | None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return None
        seen.add(normalized)
        return normalized

    def iter_call_candidates(call: Any) -> list[str]:
        if not isinstance(call, dict):
            return []
        args = call.get("args")
        if isinstance(args, dict):
            return _seed_payload_candidates_from_mapping(args, body_field=body_field)
        for container_key in ("body_form", "body"):
            container = call.get(container_key)
            if not isinstance(container, dict):
                continue
            review = container.get("review")
            if isinstance(review, dict) and set(map(str, container.keys())) == {"review"}:
                return _seed_payload_candidates_from_mapping(review, body_field=body_field)
            return _seed_payload_candidates_from_mapping(container, body_field=body_field)
        return []

    for call_list_key in ("editor_calls", "api_calls"):
        call_list = seed.get(call_list_key)
        if not isinstance(call_list, list):
            continue
        for call in call_list:
            for candidate in iter_call_candidates(call):
                resolved = consider(candidate)
                if resolved is not None:
                    return resolved
    return None


def _default_framing_witness(raw_text: str, rendered_payload: str) -> str | None:
    visible = " ".join(raw_text.split()).strip()
    if visible:
        prefix = visible[:80].strip()
        if prefix and prefix in rendered_payload:
            return prefix
    normalized_payload = " ".join(rendered_payload.split()).strip()
    if normalized_payload:
        prefix = normalized_payload[:80].strip()
        if prefix and prefix in rendered_payload:
            return prefix
    return None


def _default_concealment_witness(rendered_payload: str, concealment: str) -> str | None:
    if concealment == "markdown_fenced_system" and "```system" in rendered_payload:
        return "```system"
    if concealment == "offscreen_css":
        if "position:absolute" in rendered_payload:
            return "position:absolute"
        if "left:-" in rendered_payload:
            return "left:-"
    return None


def _updated_payload_text_entry(
    task: dict[str, Any],
    original_payload: Any,
    rendered_payload: str,
) -> dict[str, Any] | None:
    concealment = str(task.get("concealment", ""))
    visible_text, _concealed_text = _payload_text_views(rendered_payload, concealment)
    entry = json.loads(json.dumps(original_payload)) if isinstance(original_payload, dict) else {}
    entry["rendered_payload"] = rendered_payload
    entry["raw_text"] = visible_text.strip()

    framing_witnesses = [
        witness
        for witness in entry.get("framing_witnesses", [])
        if isinstance(witness, str) and witness and witness in rendered_payload
    ]
    if not framing_witnesses:
        fallback = _default_framing_witness(str(entry.get("raw_text", "")), rendered_payload)
        if fallback is not None:
            framing_witnesses = [fallback]
    entry["framing_witnesses"] = framing_witnesses

    if concealment == "plaintext":
        entry["concealment_witnesses"] = [
            witness
            for witness in entry.get("concealment_witnesses", [])
            if isinstance(witness, str) and witness and witness in rendered_payload
        ]
    else:
        concealment_witnesses = [
            witness
            for witness in entry.get("concealment_witnesses", [])
            if isinstance(witness, str) and witness and witness in rendered_payload
        ]
        if not concealment_witnesses:
            fallback = _default_concealment_witness(rendered_payload, concealment)
            if fallback is not None:
                concealment_witnesses = [fallback]
        entry["concealment_witnesses"] = concealment_witnesses

    payload_errors = validate_text_post_hoc(entry, task)
    if payload_errors:
        logger.warning(
            "Variant payload text could not be resynchronized for %s: %s",
            task.get("id", "unknown"),
            "; ".join(payload_errors),
        )
        return None
    return entry


def _synchronize_variant_payload_texts(
    original_task: dict[str, Any],
    merged_task: dict[str, Any],
    candidate_seed: dict[str, Any],
) -> None:
    payload_texts = original_task.get("payload_texts")
    if not isinstance(payload_texts, list) or not payload_texts:
        return
    selected_index = original_task.get("selected_payload_index", 0)
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(payload_texts)):
        logger.warning(
            "Variant task %s has invalid selected_payload_index=%r; keeping original payload_texts",
            original_task.get("id", "unknown"),
            selected_index,
        )
        return
    rendered_payload = _extract_variant_rendered_payload(original_task, candidate_seed)
    if not isinstance(rendered_payload, str) or not rendered_payload:
        logger.warning(
            "Variant task %s revised adversarial_data_seed does not expose a recoverable payload body; "
            "keeping original payload_texts",
            original_task.get("id", "unknown"),
        )
        return
    synced_entry = _updated_payload_text_entry(
        merged_task,
        payload_texts[selected_index],
        rendered_payload,
    )
    if synced_entry is None:
        return
    merged_payloads = json.loads(json.dumps(payload_texts))
    merged_payloads[selected_index] = synced_entry
    merged_task["payload_texts"] = merged_payloads


def _adversarial_seed_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return True when two tasks have the same adversarial seed payload."""
    return json.dumps(left.get("adversarial_data_seed"), sort_keys=True) == json.dumps(
        right.get("adversarial_data_seed"), sort_keys=True
    )


def _seed_preserves_exposure_contract_error(
    task: dict[str, Any],
    candidate_seed: dict[str, Any],
) -> str | None:
    """Reject variant/placement seeds that move outside the Phase 2 contract.

    Phase 4 variants may rewrite the payload text, but they must not turn a
    Path A candidate into a new placement. The exposure contract remains the
    authority for editor method, payload field, and core target surface.
    """
    contract = task.get("exposure_contract")
    if not isinstance(contract, dict):
        return None
    eligibility = contract.get("eligibility")
    if isinstance(eligibility, dict) and eligibility.get("status") != "eligible":
        return "exposure contract is not eligible"

    site = str(contract.get("site") or task.get("site") or "").strip()
    target_surface_id = contract.get("target_surface_id")
    from worldsim.phases.phase_2_core_surfaces import is_core_surface

    if not is_core_surface(site, str(target_surface_id) if target_surface_id else None):
        return f"target surface {target_surface_id!r} is not a Path A core surface"

    expected_method = str(contract.get("editor_method") or "").strip()
    payload_arg = str(contract.get("payload_arg") or "").strip()
    if not expected_method or not payload_arg:
        return None

    editor_calls = candidate_seed.get("editor_calls")
    if not isinstance(editor_calls, list):
        return "candidate seed does not contain editor_calls for exposure contract"

    for call in editor_calls:
        if not isinstance(call, dict) or call.get("method") != expected_method:
            continue
        call_site = str(call.get("site") or site).strip()
        if site and call_site and call_site.lower() != site.lower():
            continue
        args = call.get("args")
        if isinstance(args, dict) and payload_arg in args:
            return None
    return (
        "candidate seed does not preserve exposure contract "
        f"method={expected_method!r} payload_arg={payload_arg!r}"
    )
