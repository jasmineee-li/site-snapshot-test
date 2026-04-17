from __future__ import annotations

import requests

import pytest

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.editors.base import EditorError
from worldsim.editors.shopping_admin import ShoppingAdminEditor

pytestmark = pytest.mark.integration


def test_shopping_admin_live_update_admin_profile_and_verify(live_instance, unique_suffix):
    instance = live_instance("shopping_admin")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = ShoppingAdminEditor(instance, session)
        ShoppingAdminEditor.probe_base_state(instance)
        form = editor._fetch_form_state(
            "/admin/admin/user/edit/user_id/1/",
            action_contains="/admin/admin/user/save/",
            required_fields=("form_key",),
        )
        original_firstname = str(form["fields"].get("firstname") or "")
        updated_firstname = f"WorldSim{unique_suffix[:8]}"
        updated = False
        try:
            try:
                editor.update_admin_profile(field="firstname", value=updated_firstname)
            except EditorError as exc:
                if exc.kind == "request_failed":
                    pytest.skip("shopping_admin profile update is not writable on this live host")
                raise
            updated = True
            after = editor._form_get("/admin/admin/user/edit/user_id/1/")
            assert after is not None
            assert updated_firstname in after.text
        finally:
            if updated:
                editor.update_admin_profile(field="firstname", value=original_firstname)


def test_shopping_admin_live_create_product_review(live_instance, unique_suffix):
    instance = live_instance("shopping_admin")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = ShoppingAdminEditor(instance, session)
        ShoppingAdminEditor.probe_base_state(instance)
        try:
            payload = editor.create_product_review(
                title=f"Admin live review {unique_suffix}",
                detail=f"Admin live review detail {unique_suffix}",
                nickname="WorldSimAdmin",
                entity_pk_value=1,
                rating=4,
            )
        except EditorError as exc:
            if exc.kind in {"request_failed", "auth_missing"}:
                pytest.skip("shopping_admin review create is not writable on this live host")
            raise

        assert payload["review_id"] not in (None, "")
