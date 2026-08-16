"""Regular-participant Classifieds listing-reply editor contract.

This module is deliberately feature-local.  The editor performs the small
browser form flow needed by issue #149 and returns only evidence that can be
recovered from the ordinary listing page.  It never calls a reset, admin, or
database endpoint and never infers a reply id from ordering or a redirect.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlsplit

from warp_taskgen.editors._method_spec import FreeText, Token, editor_method
from warp_taskgen.editors._registry import EditorMethodSpec
from warp_taskgen.editors.base import BaseSiteEditor, EditorError
from warp_taskgen.sites.classifieds_reply_html import (
    extract_listing_reply_id,
    normalize_reply_body,
)

_ID_RE = re.compile(r"^[1-9][0-9]*$")

# The title is required by the upstream form but is not a WARP carrier.  A
# fixed benign value keeps title/description unavailable to the task contract.
CLASSIFIEDS_FIXED_TITLE = "Additional listing details"
CLASSIFIEDS_DEFAULT_RATING = "0"
CLASSIFIEDS_LISTING_PATH = "/index.php?page=item&id={listing_id}"
_FORM_FIELDS = (
    "action",
    "page",
    "id",
    "replyId",
    "authorName",
    "authorEmail",
    "title",
    "rating",
    "body",
    "octoken",
)
_FORM_REQUIRED_KEYS = (
    "action",
    "page",
    "id",
    "replyId",
    "authorName",
    "authorEmail",
    "title",
    "body",
    "octoken",
)


def _required_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("listing id must be a positive integer")
    text = str(value or "").strip()
    if not _ID_RE.fullmatch(text):
        raise ValueError("listing id must be a positive integer")
    return text


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _listing_path(listing_id: str) -> str:
    return CLASSIFIEDS_LISTING_PATH.format(listing_id=listing_id)


@dataclass(frozen=True)
class ClassifiedsListingReplyForm:
    """Exact regular-participant POST fields for one existing listing.

    ``author_name`` and ``author_email`` are copied from the authenticated
    listing form.  Callers cannot supply a different actor identity to the
    mutation path.
    """

    listing_id: str
    csrf_token: str
    body: str
    author_name: str
    author_email: str
    signature: str | None = None
    title: str = CLASSIFIEDS_FIXED_TITLE
    rating: str = CLASSIFIEDS_DEFAULT_RATING
    method: str = "create_listing_reply"

    def __post_init__(self) -> None:
        listing_id = _required_id(self.listing_id)
        csrf = _required_text(self.csrf_token, "CSRF token")
        body = _required_text(self.body, "reply body")
        signature = self.signature
        if signature is not None:
            signature = _required_text(signature, "reply signature")
        author_name = _required_text(self.author_name, "authenticated author name")
        author_email = _required_text(self.author_email, "authenticated author email")
        if signature is not None and signature not in body:
            raise ValueError("reply body must contain the signature")
        object.__setattr__(self, "listing_id", listing_id)
        object.__setattr__(self, "csrf_token", csrf)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "author_name", author_name)
        object.__setattr__(self, "author_email", author_email)
        # Title is a fixed benign form requirement, never a task-provided
        # carrier.  Ignore any accidental constructor override.
        object.__setattr__(self, "title", CLASSIFIEDS_FIXED_TITLE)
        object.__setattr__(self, "rating", str(self.rating or CLASSIFIEDS_DEFAULT_RATING))

    @property
    def action(self) -> str:
        return _listing_path(self.listing_id)

    @property
    def fields(self) -> Mapping[str, str]:
        # Keep this list explicit: title/description never become task data,
        # and no caller-controlled field can smuggle an endpoint or actor id.
        return MappingProxyType(
            {
                "action": "add_comment",
                "page": "item",
                "id": self.listing_id,
                "replyId": "",
                "authorName": self.author_name,
                "authorEmail": self.author_email,
                "title": self.title,
                "rating": self.rating,
                "body": self.body,
                "octoken": self.csrf_token,
            }
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "body": self.body,
            "signature": self.signature,
            "form_action": self.action,
            "form_fields": dict(self.fields),
            "read_surface_urls": [self.action],
            "read_surface_provenance_source": "classifieds.regular_participant",
        }


ListingReplyForm = ClassifiedsListingReplyForm


def build_listing_reply_form(
    *,
    listing_id: Any,
    csrf_token: Any,
    body: Any,
    author_name: Any,
    author_email: Any,
    signature: Any = None,
    title: Any = CLASSIFIEDS_FIXED_TITLE,
    rating: Any = CLASSIFIEDS_DEFAULT_RATING,
) -> ClassifiedsListingReplyForm:
    """Build the exact form independently of a live HTTP session."""

    return ClassifiedsListingReplyForm(
        listing_id=_required_id(listing_id),
        csrf_token=_required_text(csrf_token, "CSRF token"),
        body=_required_text(body, "reply body"),
        signature=(_required_text(signature, "reply signature") if signature is not None else None),
        author_name=_required_text(author_name, "authenticated author name"),
        author_email=_required_text(author_email, "authenticated author email"),
        title=str(title or CLASSIFIEDS_FIXED_TITLE),
        rating=str(rating or CLASSIFIEDS_DEFAULT_RATING),
    )


def _signature_from_body(body: Any) -> str | None:
    # The host normally supplies a selected signature separately.  This small
    # fallback only supports an explicit marker and never invents one from a
    # title or description field.
    if not isinstance(body, str):
        return None
    match = re.search(r"(?:^|\s)SIGNATURE:([^\s]+)", body)
    return match.group(1) if match else None


def _response_headers(response: Any) -> Mapping[str, Any]:
    headers = getattr(response, "headers", {})
    return headers if isinstance(headers, Mapping) else {}


def _response_text(response: Any) -> str:
    value = getattr(response, "text", "")
    return value if isinstance(value, str) else str(value or "")


class ClassifiedsEditor(BaseSiteEditor):
    """Explicitly composed regular-user editor for ``visualwebarena``."""

    site_name = "classifieds"
    benchmark = "visualwebarena"
    supported_methods = frozenset({"create_listing_reply"})

    def __init__(self, instance: Mapping[str, Any] | None = None, session: Any = None) -> None:
        # Keeping session optional makes contract/spec inspection side-effect
        # free.  The mutation method requires a live injected session and
        # raises ``auth_missing`` when one is absent.
        super().__init__(dict(instance or {}), session)

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        site_url = str(instance.get("site_url", "")).strip()
        if not site_url or urlsplit(site_url).scheme not in {"http", "https"}:
            raise EditorError("missing_site_url", "classifieds editor requires site_url")

    def probe_authenticated(self) -> bool:
        if self.session is None:
            return False
        response = self.session.get(
            f"{self._site_url().rstrip('/')}/index.php?page=user&action=dashboard",
            timeout=10,
            allow_redirects=False,
        )
        status = int(getattr(response, "status_code", 0) or 0)
        if status in {401, 403}:
            return False
        if 300 <= status < 400:
            return False
        if status != 200:
            response.raise_for_status()
            return False
        # The dashboard route alone is not proof: an unauthenticated server
        # may render its login page with HTTP 200. Require the regular-user
        # logout control that the authenticated shell exposes.
        body = _response_text(response).casefold()
        return any(marker in body for marker in ("page=logout", "action=logout", "/logout"))

    def validate_args(self, method_name: str, args: Mapping[str, Any]) -> None:
        if method_name != "create_listing_reply":
            raise EditorError(
                "unsupported_method", f"unsupported Classifieds method {method_name!r}"
            )
        if not isinstance(args, Mapping):
            raise EditorError("invalid_args", "Classifieds editor args must be a mapping")
        try:
            listing_id = _required_id(args.get("listing_id"))
            _required_text(args.get("body"), "reply body")
        except ValueError as exc:
            raise EditorError("invalid_args", str(exc)) from exc
        if self.session is None:
            # Do not require CSRF/actor fields until the GET form is available.
            return
        del listing_id

    @editor_method(
        kinds=frozenset({"listing"}),
        http=("POST", CLASSIFIEDS_LISTING_PATH),
        bindings={
            "listing_id": Token("{benign_listing_id}"),
            "body": FreeText(),
        },
        surface_id_per_kind={"listing": "listing_reply.body"},
        required_editor_args=("listing_id", "body"),
    )
    def create_listing_reply(
        self,
        *,
        listing_id: Any,
        body: str,
        signature: str | None = None,
        csrf_token: str | None = None,
    ) -> dict[str, Any]:
        """POST one top-level reply and recover its exact rendered id.

        The sequence is intentionally fixed: GET the seeded listing form,
        POST only that form's same-origin action with an exact listing id, then
        GET the same listing again as the mutation witness.  A public executor
        must subsequently fetch the emitted surface with an independent reader.
        """

        if self.session is None:
            raise EditorError(
                "auth_missing", "classifieds listing reply requires an injected session"
            )
        try:
            listing_text = _required_id(listing_id)
            body_text = _required_text(body, "reply body")
            signature_candidate = signature or _signature_from_body(body_text)
            signature_text = (
                _required_text(signature_candidate, "reply signature")
                if signature_candidate is not None
                else None
            )
        except ValueError as exc:
            raise EditorError("invalid_args", str(exc)) from exc
        if signature_text is not None and signature_text not in body_text:
            raise EditorError("invalid_args", "reply body must contain the signature")

        path = _listing_path(listing_text)
        form = self._fetch_form_state(path, required_fields=_FORM_REQUIRED_KEYS)
        fields = form.get("fields") if isinstance(form, Mapping) else None
        if not isinstance(fields, Mapping):
            raise EditorError("form_missing", "classifieds listing form did not expose fields")
        normalized_fields = {str(key): str(value or "") for key, value in fields.items()}
        self._validate_form_state(normalized_fields, listing_text)

        csrf = normalized_fields["octoken"].strip()
        if csrf_token is not None and str(csrf_token).strip() != csrf:
            raise EditorError("schema_mismatch", "caller CSRF token disagrees with rendered form")
        actor_name = normalized_fields["authorName"].strip()
        actor_email = normalized_fields["authorEmail"].strip()
        rating = normalized_fields.get("rating") or CLASSIFIEDS_DEFAULT_RATING
        form_action = self._validate_form_action(str(form.get("action") or ""), listing_text)
        payload = dict(
            ClassifiedsListingReplyForm(
                listing_id=listing_text,
                csrf_token=csrf,
                body=body_text,
                signature=signature_text,
                author_name=actor_name,
                author_email=actor_email,
                rating=rating,
            ).fields
        )

        response = self._submit_exact_form(form_action, payload)
        self._validate_submit_response(response, listing_text)

        rendered = self._form_get(path)
        if rendered is None or getattr(rendered, "status_code", 200) != 200:
            raise EditorError(
                "request_failed", "classifieds listing refetch did not return HTTP 200"
            )
        self._validate_rendered_response_url(rendered, listing_text)
        reply_id = extract_listing_reply_id(
            _response_text(rendered),
            actor=actor_name,
            body=body_text,
            signature=signature_text,
            listing_id=listing_text,
        )
        if reply_id is None:
            raise EditorError(
                "schema_mismatch",
                "classifieds listing refetch did not expose one exact rendered reply id",
            )
        surface = f"{self._site_url().rstrip('/')}{path}"
        return {
            "listing_id": listing_text,
            "reply_id": reply_id,
            "actor_name": actor_name,
            "identity_tokens": {
                "listing_id": listing_text,
                "reply_id": reply_id,
                "actor_name": actor_name,
                "reply_body_sha256": hashlib.sha256(
                    normalize_reply_body(body_text).encode("utf-8")
                ).hexdigest(),
            },
            "created_resource": {
                "role": "seed_resource",
                "kind": "listing_reply",
                "id": reply_id,
                "url": surface,
                "parent_url": surface,
                "editor_method": "classifieds.create_listing_reply",
            },
            "read_surface_urls": [surface],
            "read_surface_provenance_source": "classifieds.regular_participant",
        }

    def _validate_form_state(self, fields: Mapping[str, str], listing_id: str) -> None:
        missing_keys = [key for key in _FORM_REQUIRED_KEYS if key not in fields]
        if missing_keys:
            raise EditorError(
                "form_missing", "classifieds form missing: " + ", ".join(missing_keys)
            )
        missing = [
            key
            for key in ("action", "page", "id", "authorName", "authorEmail", "octoken")
            if not fields.get(key, "").strip()
        ]
        if missing:
            raise EditorError("form_missing", "classifieds form missing: " + ", ".join(missing))
        if fields.get("action") != "add_comment" or fields.get("page") != "item":
            raise EditorError(
                "schema_mismatch", "classifieds form is not the add_comment item form"
            )
        try:
            form_listing_id = _required_id(fields.get("id"))
        except ValueError as exc:
            raise EditorError(
                "schema_mismatch", "classifieds form has an invalid listing id"
            ) from exc
        if form_listing_id != listing_id:
            raise EditorError(
                "schema_mismatch", "classifieds form listing id does not match the seed"
            )
        if fields.get("replyId", "").strip() not in {"", "0"}:
            raise EditorError("schema_mismatch", "nested replies are outside listing_reply.body")

    def _validate_form_action(self, action: str, listing_id: str) -> str:
        if not action.strip():
            raise EditorError("form_missing", "classifieds listing form has no action")
        try:
            resolved = urljoin(f"{self._site_url().rstrip('/')}/", action)
            site = urlsplit(self._site_url())
            parsed = urlsplit(resolved)
        except ValueError as exc:
            raise EditorError(
                "cross_origin_form_action", "classifieds form action is malformed"
            ) from exc
        if (parsed.scheme, parsed.netloc) != (site.scheme, site.netloc):
            raise EditorError("cross_origin_form_action", "classifieds form action is foreign")
        if parsed.fragment or parsed.path != "/index.php":
            raise EditorError(
                "cross_origin_form_action", "classifieds form action is not /index.php"
            )
        query = parse_qsl(parsed.query, keep_blank_values=True)
        if query and query != [("page", "item"), ("id", listing_id)]:
            raise EditorError("schema_mismatch", "classifieds form action targets another listing")
        return action.strip()

    def _validate_submit_response(self, response: Any, listing_id: str) -> None:
        status = getattr(response, "status_code", None)
        if status == 200:
            return
        if isinstance(status, int) and 300 <= status < 400:
            location = _response_headers(response).get("Location")
            if not isinstance(location, str) or not location.strip():
                raise EditorError(
                    "unexpected_redirect", "classifieds comment POST had no redirect target"
                )
            expected = urljoin(f"{self._site_url().rstrip('/')}/", _listing_path(listing_id))
            resolved = urljoin(f"{self._site_url().rstrip('/')}/", location.strip())
            if not self._same_origin_listing_url(resolved, listing_id):
                raise EditorError(
                    "unexpected_redirect", "classifieds comment POST redirected off listing"
                )
            if (
                urlsplit(resolved).path != urlsplit(expected).path
                or urlsplit(resolved).query != urlsplit(expected).query
            ):
                raise EditorError(
                    "unexpected_redirect", "classifieds comment POST redirected to another page"
                )
            return
        raise EditorError("request_failed", f"classifieds comment POST returned HTTP {status}")

    def _validate_rendered_response_url(self, response: Any, listing_id: str) -> None:
        raw_url = getattr(response, "url", "")
        if not isinstance(raw_url, str) or not raw_url.strip():
            return
        if not self._same_origin_listing_url(raw_url, listing_id):
            raise EditorError(
                "schema_mismatch", "classifieds refetch URL is not the seeded listing"
            )

    def _same_origin_listing_url(self, url: str, listing_id: str) -> bool:
        try:
            parsed = urlsplit(url)
            site = urlsplit(self._site_url())
            expected = urlsplit(_listing_path(listing_id))
        except ValueError:
            return False
        if (parsed.scheme, parsed.netloc) != (site.scheme, site.netloc):
            return False
        return (
            parsed.path == expected.path and parsed.query == expected.query and not parsed.fragment
        )


ClassifiedsListingReplyEditor = ClassifiedsEditor


def classifieds_editor_specs() -> tuple[EditorMethodSpec, ...]:
    """Return pure registry-shaped specs without global registration."""

    metadata = ClassifiedsEditor.create_listing_reply._editor_method_spec
    return (
        EditorMethodSpec(
            benchmark=ClassifiedsEditor.benchmark,
            site=ClassifiedsEditor.site_name,
            method="create_listing_reply",
            kinds=metadata["kinds"],
            http=metadata["http"],
            bindings=dict(metadata["bindings"]),
            surface_id_per_kind=dict(metadata["surface_id_per_kind"]),
            required_editor_args=metadata["required_editor_args"],
        ),
    )


__all__ = [
    "CLASSIFIEDS_DEFAULT_RATING",
    "CLASSIFIEDS_FIXED_TITLE",
    "ClassifiedsEditor",
    "ClassifiedsListingReplyEditor",
    "ClassifiedsListingReplyForm",
    "ListingReplyForm",
    "build_listing_reply_form",
    "classifieds_editor_specs",
    "extract_listing_reply_id",
]
