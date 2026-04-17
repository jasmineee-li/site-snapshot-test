from __future__ import annotations

import requests

import pytest

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.editors.shopping import ShoppingEditor

pytestmark = pytest.mark.integration


def test_shopping_live_update_customer_profile_and_verify(live_instance, unique_suffix):
    instance = live_instance("shopping")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = ShoppingEditor(instance, session)
        ShoppingEditor.probe_base_state(instance)
        form = editor._fetch_form_state(
            "/customer/account/edit/",
            action_contains="/customer/account/editpost/",
            required_fields=("firstname", "lastname", "email"),
        )
        original_firstname = str(form["fields"].get("firstname") or "")
        updated_firstname = f"WorldSim{unique_suffix[:8]}"
        updated = False
        try:
            editor.update_customer_profile(field="firstname", value=updated_firstname)
            updated = True
            after = editor._form_get("/customer/account/")
            assert after is not None
            assert updated_firstname in after.text
        finally:
            if updated:
                editor.update_customer_profile(field="firstname", value=original_firstname)


def test_shopping_live_create_product_review(live_instance, unique_suffix):
    instance = live_instance("shopping")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = ShoppingEditor(instance, session)
        ShoppingEditor.probe_base_state(instance)
        payload = editor.create_product_review(
            title=f"Live review {unique_suffix}",
            detail=f"Live review detail {unique_suffix}",
            nickname="WorldSim",
            entity_pk_value=1,
            rating=4,
        )

        assert payload["review_id"] not in (None, "")
