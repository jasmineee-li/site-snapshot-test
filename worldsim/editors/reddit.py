from __future__ import annotations

import html
import re
from typing import Any

import requests

from ._method_spec import FreeText, Token, editor_method
from ._read_surface import host_and_path_forms, normalize_surface_urls
from .base import BaseSiteEditor, EditorError


def _reddit_surface_urls(site_url: str, forms: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in forms:
        expanded.extend(host_and_path_forms(site_url, value))
    return normalize_surface_urls(expanded)


_SUBMISSION_ID_RE = re.compile(r"/f/[^/]+/(\d+)(?:/|$)")

# PostMill renders validation errors inside elements with class="form-error"
# (the containing <ul>) or <li class="form-error__item">. Grab anything inside
# those blocks; we also fall back to the raw body so server-side flash messages
# are still classified.
_POSTMILL_FORM_ERROR_RE = re.compile(
    r"<(?:li|span|div)[^>]+class=\"[^\"]*form-error[^\"]*\"[^>]*>(.*?)</(?:li|span|div)>",
    re.IGNORECASE | re.DOTALL,
)
_POSTMILL_FLASH_RE = re.compile(
    r"<(?:div|p)[^>]+class=\"[^\"]*(?:flash|alert|message)[^\"]*\"[^>]*>(.*?)</(?:div|p)>",
    re.IGNORECASE | re.DOTALL,
)
_POSTMILL_LENGTH_TOKENS: tuple[str, ...] = (
    "is too long",
    "too long",
    "must be shorter",
    "maximum of",
    "cannot be longer",
)
_POSTMILL_REQUIRED_TOKENS: tuple[str, ...] = (
    "is required",
    "should not be blank",
    "cannot be blank",
    "must not be empty",
    "this value should not",
)
_POSTMILL_POLICY_TOKENS: tuple[str, ...] = (
    "flagged as spam",
    "marked as spam",
    "rate limit",
    "throttled",
    "not allowed",
    "forbidden",
    "abuse",
    "banned",
    "rejected",
)


def _extract_postmill_error_messages(response: requests.Response) -> list[str]:
    text = response.text or ""
    if not text:
        return []
    messages: list[str] = []
    for match in _POSTMILL_FORM_ERROR_RE.findall(text):
        stripped = _strip_tags(match)
        if stripped:
            messages.append(stripped)
    if not messages:
        for match in _POSTMILL_FLASH_RE.findall(text):
            stripped = _strip_tags(match)
            if stripped:
                messages.append(stripped)
    return messages


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


class RedditEditor(BaseSiteEditor):
    site_name = "reddit"
    supported_methods = frozenset(
        {
            "create_forum",
            "create_submission",
            "create_comment",
            "update_user_bio",
        }
    )

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        username = cls._resolve_current_username(instance)
        with requests.Session() as session:
            editor = cls(instance, session)
            editor._form_get(f"/user/{cls._quote(username)}/edit_biography")

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        validators = {
            "create_forum": lambda: self._require_args(args, "name_template"),
            "create_submission": lambda: self._require_args(args, "forum_name", "title_template"),
            "create_comment": lambda: self._require_args(
                args, "forum_name", "submission_id", "body"
            ),
            "update_user_bio": lambda: self._require_args(args, "bio_text"),
        }
        validator = validators.get(method_name)
        if validator is None:
            raise EditorError(
                "unsupported_method", f"unsupported reddit editor method {method_name!r}"
            )
        self._resolve_current_username(self.instance)
        validator()

    def preview_context(self, method_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if method_name == "create_forum":
            return {"forum_name": str(args.get("name_template") or "").strip()}
        if method_name == "create_submission":
            forum_name = str(args.get("forum_name") or "").strip()
            submission_id = args.get("submission_id") or "1"
            return {
                "forum_name": forum_name,
                "submission_id": str(submission_id),
                "submission_url": f"{self._site_url()}/f/{forum_name}/{submission_id}",
            }
        if method_name == "create_comment":
            return {
                "forum_name": str(args.get("forum_name") or "").strip(),
                "submission_id": str(args.get("submission_id") or ""),
            }
        return {}

    def _classify_4xx_response(
        self,
        method: str,
        path: str,
        response: requests.Response,
    ) -> tuple[str, str] | None:
        """Classify PostMill 4xx responses into Phase 2c error kinds.

        PostMill returns 422 for validation failures with errors rendered
        inside ``<li class="form-error__item">`` blocks (or a similar
        container on older themes). Rate-limit / spam flags come back as
        flash messages.
        """
        messages = _extract_postmill_error_messages(response)
        if not messages:
            return None
        blob = " | ".join(messages).lower()
        if any(token in blob for token in _POSTMILL_LENGTH_TOKENS):
            return ("length_exceeded", messages[0])
        if any(token in blob for token in _POSTMILL_REQUIRED_TOKENS):
            return ("field_required", messages[0])
        if any(token in blob for token in _POSTMILL_POLICY_TOKENS):
            return ("content_policy", messages[0])
        return None

    @editor_method(
        kinds=frozenset(),  # dangling parent — never a valid Option A attach
        http=("POST", "/create_forum"),
        bindings={
            "name_template": FreeText(),
            "description_template": FreeText(required=False),
        },
    )
    def create_forum(
        self,
        *,
        name_template: str,
        description_template: str | None = None,
    ) -> dict[str, Any]:
        forum_name = self._allocate_forum_name(str(name_template).strip())
        forum_surface = _reddit_surface_urls(self._site_url(), [f"/f/{forum_name}"])

        def build_form_submission() -> tuple[str, dict[str, Any], bool]:
            form = self._fetch_form_state(
                "/create_forum",
                required_fields=(
                    "forum[name]",
                    "forum[title]",
                    "forum[description]",
                    "forum[sidebar]",
                    "forum[_token]",
                ),
            )
            payload = dict(form.get("fields", {}))
            description = str(description_template or f"Forum for {forum_name}").strip()
            payload["forum[name]"] = forum_name
            payload["forum[title]"] = forum_name
            payload["forum[description]"] = description
            payload["forum[sidebar]"] = description
            return str(form.get("action") or "/create_forum"), payload, False

        action, payload, multipart = build_form_submission()
        self._submit_exact_form(
            action,
            payload,
            multipart=multipart,
            refresh_on_rejection=build_form_submission,
        )
        if not self._forum_exists(forum_name):
            raise EditorError(
                "request_failed",
                f"reddit forum create did not make {forum_name!r} accessible after submit",
            )
        return {
            "forum_name": forum_name,
            "read_surface_urls": forum_surface,
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset({"reddit_forum"}),
        http=("POST", "/submit/{forum_name}"),
        bindings={
            "forum_name": Token("{benign_forum_name}"),
            # LLM-facing "title" → title_template, "body" → body_template
            # via seeding._editor_arg_name alias map.
            "title": FreeText(),
            "body": FreeText(required=False),
        },
        surface_id_per_kind={"reddit_forum": "submission_body_detail"},
        required_editor_args=("forum_name", "title", "body"),
    )
    def create_submission(
        self,
        *,
        forum_name: str,
        title_template: str,
        body_template: str | None = None,
    ) -> dict[str, Any]:
        submit_path = f"/submit/{forum_name}"

        def build_form_submission() -> tuple[str, dict[str, Any], bool]:
            form = self._fetch_form_state(
                submit_path,
                required_fields=("submission[_token]", "submission[forum]", "submission[title]"),
            )
            payload = dict(form.get("fields", {}))
            payload["submission[title]"] = str(title_template).strip()
            if body_template is not None:
                payload["submission[body]"] = str(body_template).strip()
            # Some Postmill submit forms omit the default media type field even
            # though the backend still requires it for text posts.
            payload.setdefault("submission[mediaType]", "url")
            payload["submission[forum]"] = self._resolve_forum_id(form, forum_name)
            return str(form.get("action") or submit_path), payload, self._is_multipart_form(form)

        action, payload, multipart = build_form_submission()
        response = self._submit_exact_form(
            action,
            payload,
            multipart=multipart,
            refresh_on_rejection=build_form_submission,
        )
        submission_id = self._extract_submission_id(response)
        submission_url = f"{self._site_url()}/f/{forum_name}/{submission_id}"
        return {
            "forum_name": forum_name,
            "submission_id": submission_id,
            "submission_url": submission_url,
            "created_resource": {
                "role": "seed_render_surface",
                "kind": "submission",
                "id": submission_id,
                "url": submission_url,
                "parent_url": f"{self._site_url()}/f/{forum_name}",
            },
            "read_surface_urls": _reddit_surface_urls(
                self._site_url(), [f"/f/{forum_name}/{submission_id}"]
            ),
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset({"reddit_submission", "reddit_dashboard_list"}),
        http=("POST", "/f/{forum_name}/{submission_id}/-/comment"),
        bindings={
            "submission_id": Token("{benign_submission_id}"),
            "body": FreeText(),
            "forum_name": Token("{benign_forum_name}", required=False),
        },
        surface_id_per_kind={
            "reddit_submission": "comment_body_thread",
            "reddit_dashboard_list": "comment_body_thread",
        },
        required_editor_args=("submission_id", "body"),
    )
    def create_comment(
        self, *, submission_id: Any, body: str, forum_name: str | None = None
    ) -> dict[str, Any]:
        resolved_forum = str(forum_name or "").strip()
        if not resolved_forum:
            raise EditorError("invalid_args", "forum_name is required for reddit comments")
        comment_path = f"/f/{resolved_forum}/{submission_id}"
        form_name = f"reply_to_submission_{submission_id}[comment]"

        def build_form_submission() -> tuple[str, dict[str, Any], bool]:
            form = self._fetch_form_state(
                comment_path,
                action_contains=f"/f/{resolved_forum}/{submission_id}/-/comment",
                required_fields=(form_name, f"reply_to_submission_{submission_id}[_token]"),
            )
            payload = dict(form.get("fields", {}))
            payload[form_name] = str(body)
            return str(form.get("action") or comment_path), payload, False

        action, payload, multipart = build_form_submission()
        response = self._submit_exact_form(
            action,
            payload,
            multipart=multipart,
            refresh_on_rejection=build_form_submission,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        comment_id = None
        if isinstance(payload, dict):
            comment_id = payload.get("id") or payload.get("comment_id")
        return {
            "forum_name": resolved_forum,
            "submission_id": str(submission_id),
            "comment_id": comment_id,
            "read_surface_urls": _reddit_surface_urls(
                self._site_url(), [f"/f/{resolved_forum}/{submission_id}"]
            ),
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset(),  # acts on current user — not an Option A attach
        http=("POST", "/user/{username}/edit_biography"),
        bindings={
            # LLM-facing "bio" → bio_text via seeding._editor_arg_name.
            "bio": FreeText(),
        },
    )
    def update_user_bio(self, *, bio_text: str) -> dict[str, Any]:
        username = self._resolve_current_username(self.instance)
        edit_path = f"/user/{self._quote(username)}/edit_biography"

        def build_form_submission() -> tuple[str, dict[str, Any], bool]:
            form = self._fetch_form_state(
                edit_path,
                required_fields=("user_biography[biography]", "user_biography[_token]"),
            )
            payload = dict(form.get("fields", {}))
            payload["user_biography[biography]"] = str(bio_text)
            return str(form.get("action") or edit_path), payload, False

        action, payload, multipart = build_form_submission()
        self._submit_exact_form(
            action,
            payload,
            multipart=multipart,
            refresh_on_rejection=build_form_submission,
        )
        profile_page = self._form_get(f"/user/{self._quote(username)}")
        if profile_page is None:
            raise EditorError(
                "request_failed",
                "reddit bio update did not leave the profile page accessible",
            )
        if bio_text and str(bio_text) not in profile_page.text:
            raise EditorError("request_failed", "reddit bio update did not persist the new text")
        return {
            "read_surface_urls": _reddit_surface_urls(self._site_url(), [f"/user/{username}"]),
            "read_surface_provenance_source": "editor_constructed",
        }

    @classmethod
    def _resolve_current_username(cls, instance: dict[str, Any]) -> str:
        for source in (
            cls._nested_lookup(instance.get("auth"), ("credentials", "username")),
            cls._nested_lookup(instance.get("auth"), ("username",)),
            cls._nested_lookup(instance.get("api_auth"), ("credentials", "username")),
            cls._nested_lookup(instance.get("api_auth"), ("username",)),
            cls._nested_lookup(
                instance.get("agent_auth"), ("authentication", "credentials", "username")
            ),
            cls._nested_lookup(instance.get("agent_auth"), ("credentials", "username")),
            cls._nested_lookup(instance.get("agent_auth"), ("username",)),
        ):
            if isinstance(source, str) and source.strip():
                return source.strip()
        raise EditorError(
            "missing_username", "reddit editor requires auth credentials with a username"
        )

    def _forum_exists(self, forum_name: str) -> bool:
        response = self._form_get(f"/f/{self._quote(forum_name)}", allow_missing=True)
        return response is not None and response.status_code == 200

    def _allocate_forum_name(self, requested_name: str) -> str:
        base = requested_name.strip()
        if not base:
            raise EditorError("invalid_args", "forum name_template must not be empty")
        candidate = base
        suffix = 2
        # Suffix separator must be an underscore: Postmill silently
        # rejects POST /create_forum submissions whose ``forum[name]``
        # contains a ``-`` and returns the form page back with status
        # 200 (no flash, no form-error items), so _submit_exact_form
        # treats the response as success but _forum_exists then fails.
        # Confirmed by probing the live stack. Underscore matches the
        # stable postmill validation regex (alphanumeric + underscore).
        while self._forum_exists(candidate):
            candidate = f"{base}_{suffix}"
            suffix += 1
            if suffix > 100:
                raise EditorError(
                    "request_failed",
                    f"reddit forum create could not allocate a fresh name for {base!r}",
                )
        return candidate

    def _extract_submission_id(self, response: requests.Response) -> str:
        location = response.headers.get("Location", "")
        match = _SUBMISSION_ID_RE.search(location)
        if match:
            return match.group(1)
        if response.headers.get("Content-Type", "").startswith("text/html"):
            raise EditorError(
                "submission_id_missing",
                "reddit submission create did not redirect to a submission page",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise EditorError(
                "submission_id_missing",
                "reddit submission create response did not include a parseable submission id",
            ) from exc
        if isinstance(payload, dict):
            submission_id = payload.get("id") or payload.get("submission_id")
            if submission_id not in (None, ""):
                return str(submission_id)
        raise EditorError(
            "submission_id_missing",
            "reddit submission create response did not include a submission id",
        )

    @staticmethod
    def _is_multipart_form(form: dict[str, Any]) -> bool:
        enctype = str(form.get("enctype") or "").lower()
        return "multipart/form-data" in enctype

    @staticmethod
    def _resolve_forum_id(form: dict[str, Any], forum_name: str) -> str:
        select_options = form.get("select_options", {})
        forum_options = select_options.get("submission[forum]")
        if isinstance(forum_options, list):
            for option in forum_options:
                if not isinstance(option, dict):
                    continue
                if str(option.get("data-name") or "").strip() == forum_name:
                    value = str(option.get("value") or "").strip()
                    if value:
                        return value
        raise EditorError(
            "forum_id_missing",
            f"reddit submission form did not expose a forum id for {forum_name!r}",
        )
