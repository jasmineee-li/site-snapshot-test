from __future__ import annotations

from typing import Any

import requests

from .base import EditorError
from .shopping import ShoppingEditor


class ShoppingAdminEditor(ShoppingEditor):
    site_name = "shopping_admin"
    supported_methods = frozenset({"create_product_review", "update_admin_profile"})
    allowed_profile_fields = frozenset({"firstname", "lastname"})

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        with requests.Session() as session:
            editor = cls(instance, session)
            response = session.post(
                f"{editor._site_url()}/rest/V1/reviews",
                headers=editor._build_headers(mechanism="api"),
                json={"review": {}},
                timeout=30,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                raise EditorError(
                    "unexpected_redirect",
                    "shopping_admin editor request for /rest/V1/reviews returned "
                    f"HTTP {response.status_code}",
                )
            if response.status_code in {401, 403}:
                raise EditorError(
                    "auth_missing",
                    "shopping_admin editor request for /rest/V1/reviews returned "
                    f"HTTP {response.status_code}",
                )
            if response.status_code >= 500:
                raise EditorError(
                    "request_failed",
                    "shopping_admin editor request for /rest/V1/reviews returned "
                    f"HTTP {response.status_code}",
                )
            editor._form_get("/admin/admin/user/edit/user_id/1/")

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        if method_name == "update_admin_profile":
            self._validate_profile_update_args(args)
            return
        super().validate_args(method_name, args)

    def update_admin_profile(self, *, field: str, value: Any) -> dict[str, Any]:
        def build_form_submission() -> tuple[str, dict[str, Any], bool]:
            edit_path = "/admin/admin/user/edit/user_id/1/"
            form = self._fetch_form_state(
                edit_path,
                action_contains="/admin/admin/user/save/",
                required_fields=("form_key",),
            )
            payload = dict(form.get("fields", {}))
            payload[str(field)] = "" if value is None else str(value)
            return str(form.get("action") or "/admin/admin/user/save/"), payload, False

        action, payload, multipart = build_form_submission()
        self._submit_exact_form(
            action,
            payload,
            multipart=multipart,
            refresh_on_rejection=build_form_submission,
        )
        return {}
