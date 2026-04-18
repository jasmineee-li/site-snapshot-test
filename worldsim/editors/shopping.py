from __future__ import annotations

from typing import Any

import requests

from ._read_surface import host_and_path_forms, normalize_surface_urls
from .base import BaseSiteEditor, EditorError


def _shopping_surface_urls(site_url: str, paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in paths:
        expanded.extend(host_and_path_forms(site_url, value))
    return normalize_surface_urls(expanded)


_MAGENTO_LENGTH_TOKENS: tuple[str, ...] = (
    "maximum length",
    "is too long",
    "exceeds the maximum",
    "cannot be longer",
    "please enter a value less than",
    "value too long",
)
_MAGENTO_REQUIRED_TOKENS: tuple[str, ...] = (
    "is a required field",
    "required parameter",
    "is required",
    "cannot be empty",
    "please enter a value",
    "missing parameter",
    "missing required",
)
_MAGENTO_POLICY_TOKENS: tuple[str, ...] = (
    "spam",
    "forbidden",
    "rejected",
    "not allowed",
    "abuse",
    "policy violation",
    "blocked",
    "denied",
)


def _extract_magento_error_messages(response: requests.Response) -> list[str]:
    """Pull error strings from Magento's structured JSON and fallback shapes.

    Magento REST typically answers with:
    ``{"message": "...", "parameters": [...]}`` or
    ``{"errors": [{"message": "..."}]}``.
    """
    body: Any
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return [text] if text else []

    collected: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if node.strip():
                collected.append(node.strip())
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    if isinstance(body, dict):
        for key in ("message", "errors", "error", "error_description", "parameters"):
            if key in body:
                walk(body[key])
        if not collected:
            walk(body)
    else:
        walk(body)
    return collected


class ShoppingEditor(BaseSiteEditor):
    site_name = "shopping"
    supported_methods = frozenset({"create_product_review", "update_customer_profile"})
    allowed_profile_fields = frozenset({"firstname", "lastname"})

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        with requests.Session() as session:
            editor = cls(instance, session)
            editor._api_request_json("GET", "/rest/V1/store/storeConfigs")
            editor._form_get("/customer/account/edit/")

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        validators = {
            "create_product_review": lambda: self._validate_product_review_args(args),
            "update_customer_profile": lambda: self._validate_profile_update_args(args),
        }
        validator = validators.get(method_name)
        if validator is None:
            raise EditorError(
                "unsupported_method", f"unsupported shopping editor method {method_name!r}"
            )
        validator()

    def _classify_4xx_response(
        self,
        method: str,
        path: str,
        response: requests.Response,
    ) -> tuple[str, str] | None:
        """Classify Magento 4xx responses into Phase 2c error kinds."""
        messages = _extract_magento_error_messages(response)
        if not messages:
            return None
        blob = " | ".join(messages).lower()
        if any(token in blob for token in _MAGENTO_LENGTH_TOKENS):
            return ("length_exceeded", messages[0])
        if any(token in blob for token in _MAGENTO_REQUIRED_TOKENS):
            return ("field_required", messages[0])
        if any(token in blob for token in _MAGENTO_POLICY_TOKENS):
            return ("content_policy", messages[0])
        return None

    def create_product_review(
        self,
        *,
        title: str | None = None,
        detail: str,
        rating: int | None = None,
        product_sku: str | None = None,
        entity_pk_value: Any | None = None,
        nickname: str | None = None,
    ) -> dict[str, Any]:
        review_payload: dict[str, Any] = {
            "review_entity": "product",
            "title": str(title or "Review").strip(),
            "detail": str(detail),
            "nickname": str(nickname or "WorldSim").strip(),
            "entity_pk_value": entity_pk_value
            if entity_pk_value not in (None, "")
            else product_sku,
            "stores": [1],
            "ratings": [{"rating_name": "Quality", "value": int(rating or 4)}],
        }
        missing = [
            field
            for field in ("detail", "nickname", "entity_pk_value")
            if review_payload.get(field) in (None, "")
        ]
        if missing:
            raise EditorError("invalid_args", "missing required review args: " + ", ".join(missing))
        payload = self._api_request_json(
            "POST",
            "/rest/V1/reviews",
            json_body={"review": review_payload},
        )
        review_id = None
        if isinstance(payload, dict):
            review_id = payload.get("id") or payload.get("review_id")
        product_key = review_payload.get("entity_pk_value")
        surface_paths: list[str] = []
        if product_key not in (None, ""):
            surface_paths.append(f"/catalog/product/view/id/{product_key}")
            surface_paths.append(f"/review/product/listAjax/id/{product_key}")
        return {
            "review_id": review_id,
            "read_surface_urls": _shopping_surface_urls(self._site_url(), surface_paths),
            "read_surface_provenance_source": "editor_constructed",
        }

    def update_customer_profile(self, *, field: str, value: Any) -> dict[str, Any]:
        def build_form_submission() -> tuple[str, dict[str, Any], bool]:
            edit_path = "/customer/account/edit/"
            form = self._fetch_form_state(
                edit_path,
                action_contains="/customer/account/editpost/",
                required_fields=("form_key", "firstname", "lastname", "email"),
            )
            payload = dict(form.get("fields", {}))
            payload[str(field)] = "" if value is None else str(value)
            for transient_field in (
                "change_email",
                "change_password",
                "current_password",
                "password",
                "password_confirmation",
            ):
                payload.pop(transient_field, None)
            return str(form.get("action") or "/customer/account/editpost/"), payload, False

        action, payload, multipart = build_form_submission()
        self._submit_exact_form(
            action,
            payload,
            multipart=multipart,
            refresh_on_rejection=build_form_submission,
        )
        return {
            "read_surface_urls": _shopping_surface_urls(
                self._site_url(), ["/customer/account/index/"]
            ),
            "read_surface_provenance_source": "editor_constructed",
        }

    def _validate_product_review_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "detail")
        if args.get("entity_pk_value") in (None, "") and args.get("product_sku") in (None, ""):
            raise EditorError(
                "invalid_args",
                "shopping product review requires entity_pk_value or product_sku",
            )

    def _validate_profile_update_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "field", "value")
        field_name = str(args.get("field") or "").strip()
        if field_name not in self.allowed_profile_fields:
            raise EditorError(
                "invalid_args",
                f"shopping profile updates only support fields: {sorted(self.allowed_profile_fields)}",
            )
