from __future__ import annotations

import pytest
import requests

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
    """Verify shopping_admin editor creates a review via the Magento REST API.

    This test exercises the admin-backend REST path ``POST /rest/V1/reviews``.
    Whether that route is registered is an install-time property of the
    benchmark host's Magento image — the default WebArena Magento admin
    container does not enable the ``Magento_Review`` REST webapi, so this
    test skips with a diagnostic 404 message there. Phase 2 adversarial
    seeding routes review plants via ``delivery_channel.delivery_site``,
    and every current adversarial task resolves to the shopping storefront
    (port 7770) rather than the admin backend (port 7780), so the skip
    does not degrade live-integration coverage.

    Skip classes are deliberately narrow:
      - ``http_status == 404``: route not registered → install-time gap.
      - ``kind == "auth_missing"``: 401/403 → admin auth config issue.
    Any other failure (5xx, timeouts, transient errors) re-raises so a
    genuine server-side regression surfaces instead of hiding behind a
    generic skip.
    """
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
            if exc.kind == "auth_missing":
                pytest.skip(
                    "shopping_admin review create returned auth_missing "
                    f"(HTTP {exc.http_status}); verify api_auth admin bearer "
                    "configuration on this host"
                )
            if exc.kind == "request_failed" and exc.http_status == 404:
                pytest.skip(
                    "shopping_admin Magento install does not expose "
                    "/rest/V1/reviews (HTTP 404 'Request does not match any "
                    "route.'); the Magento_Review REST webapi module is not "
                    "registered on this admin backend. Phase 2 seeds reviews "
                    "via delivery_site='shopping' on port 7770 instead."
                )
            raise

        assert payload["review_id"] not in (None, "")
