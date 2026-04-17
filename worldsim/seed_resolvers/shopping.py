from __future__ import annotations

from typing import Any

from .types import ResolvedCall, ResolverError


def create(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    resource_type = str(target.get("resource_type", "")).strip().lower()
    if resource_type != "product_review":
        raise ResolverError(
            "unsupported_resource_type",
            f"shopping resolver does not support create resource_type={resource_type!r}",
        )

    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ResolverError("missing_site_url", "shopping resolver requires instance['site_url']")

    create_spec = target.get("create")
    if not isinstance(create_spec, dict):
        create_spec = {}
    review_spec = create_spec.get("product_review")
    if not isinstance(review_spec, dict):
        review_spec = {}

    raw_body = target.get("body")
    if not isinstance(raw_body, dict):
        raw_body = {}
    nested_body = raw_body.get("review")
    if isinstance(nested_body, dict) and any(key != "review" for key in raw_body):
        raise ResolverError(
            "conflicting_review_body_shapes",
            "shopping product_review target must use either top-level body fields or body.review, not both",
        )
    review_payload = dict(review_spec)
    if isinstance(nested_body, dict):
        review_payload.update(nested_body)
    else:
        review_payload.update(raw_body)

    review_payload.setdefault("review_entity", "product")
    review_payload.setdefault("stores", [1])
    review_payload.setdefault("ratings", [{"rating_name": "Quality", "value": 4}])

    missing_fields = [
        field
        for field in ("title", "detail", "nickname", "entity_pk_value", "ratings")
        if review_payload.get(field) in (None, "")
    ]
    if missing_fields:
        raise ResolverError(
            "missing_required_fields",
            "shopping product_review target is missing required fields: "
            + ", ".join(missing_fields),
        )

    if _existing_review_seeded(instance, review_payload):
        return ResolvedCall(
            method="GET",
            url=f"{site_url}/",
            headers={},
            context_additions={
                "resolved_review_entity": review_payload.get("review_entity"),
                "resolved_product_id": review_payload.get("entity_pk_value"),
                "review_already_seeded": True,
            },
        )

    return ResolvedCall(
        method=str(target.get("method") or "POST").strip().upper() or "POST",
        url=f"{site_url}/rest/V1/reviews",
        headers={},
        body={"review": review_payload},
        context_additions={
            "resolved_review_entity": review_payload.get("review_entity"),
            "resolved_product_id": review_payload.get("entity_pk_value"),
        },
    )


def update(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    resource_type = str(target.get("resource_type", "")).strip().lower()
    if resource_type != "customer_profile":
        raise ResolverError(
            "unsupported_resource_type",
            f"shopping resolver does not support update resource_type={resource_type!r}",
        )
    return ResolvedCall(
        method="PUT",
        url=_site_url(instance, "/rest/V1/customers/me"),
        headers={},
    )


def _site_url(instance: dict[str, Any], path: str) -> str:
    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ResolverError("missing_site_url", "shopping resolver requires instance['site_url']")
    return f"{site_url}{path}"


def _existing_review_seeded(instance: dict[str, Any], review_payload: dict[str, Any]) -> bool:
    db_connection = instance.get("db_connection")
    if not db_connection:
        return False

    from worldsim import seeding as seeding_module

    try:
        parsed = seeding_module._parse_runtime_db_connection(
            db_connection,
            purpose="shopping resolver review lookup requires instance['db_connection']",
        )
        conn = seeding_module._connect_db(parsed)
    except Exception:
        return False
    try:
        try:
            seeding_module._configure_read_only_connection(conn, parsed.scheme.lower())
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM review_detail WHERE product_id = %s AND body = %s LIMIT 1",
                    [review_payload.get("entity_pk_value"), review_payload.get("detail")],
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()
