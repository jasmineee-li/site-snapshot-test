from __future__ import annotations

import logging
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests

logger = logging.getLogger(__name__)


class EditorError(RuntimeError):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[dict[str, Any]] = []
        self._current_form: dict[str, Any] | None = None
        self._current_textarea_name: str | None = None
        self._textarea_chunks: list[str] = []
        self._current_select_name: str | None = None
        self._current_option_attrs: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value for key, value in attrs}
        if tag == "form":
            self._current_form = {
                "action": attrs_dict.get("action") or "",
                "method": attrs_dict.get("method") or "post",
                "enctype": attrs_dict.get("enctype") or "",
                "fields": {},
                "select_options": {},
            }
            self.forms.append(self._current_form)
            return
        if self._current_form is None:
            return
        if tag == "input":
            self._handle_input(attrs_dict)
            return
        if tag == "textarea":
            name = attrs_dict.get("name")
            if name:
                self._current_textarea_name = name
                self._textarea_chunks = []
            return
        if tag == "select":
            name = attrs_dict.get("name")
            if name:
                self._current_select_name = name
                self._current_form["select_options"].setdefault(name, [])
            return
        if tag == "option" and self._current_select_name:
            option_attrs = {key: value or "" for key, value in attrs}
            self._current_form["select_options"][self._current_select_name].append(option_attrs)
            self._current_option_attrs = option_attrs
            # Mirror browser behavior: an explicitly selected option becomes
            # the submitted value. Without this, single-option selects (e.g.
            # postmill's userFlag) yield null on POST and blow up server-side.
            if "selected" in option_attrs:
                self._current_form["fields"][self._current_select_name] = option_attrs.get(
                    "value", ""
                )

    def handle_endtag(self, tag: str) -> None:
        if tag == "textarea" and self._current_form is not None and self._current_textarea_name:
            self._current_form["fields"][self._current_textarea_name] = "".join(
                self._textarea_chunks
            )
            self._current_textarea_name = None
            self._textarea_chunks = []
            return
        if tag == "select":
            # Per HTML spec, a single-value <select> without an explicitly
            # selected <option> submits the first option's value. Apply that
            # fallback when nothing was marked selected inside the select.
            select_name = self._current_select_name
            if (
                select_name
                and self._current_form is not None
                and select_name not in self._current_form["fields"]
            ):
                options = self._current_form["select_options"].get(select_name, [])
                if options:
                    self._current_form["fields"][select_name] = options[0].get("value", "")
            self._current_select_name = None
            self._current_option_attrs = None
            return
        if tag == "option":
            self._current_option_attrs = None

    def handle_data(self, data: str) -> None:
        if self._current_textarea_name is not None:
            self._textarea_chunks.append(data)
            return
        if self._current_option_attrs is not None:
            label = self._current_option_attrs.get("label", "")
            self._current_option_attrs["label"] = f"{label}{data}".strip()

    def _handle_input(self, attrs_dict: dict[str, str | None]) -> None:
        if self._current_form is None:
            return
        name = attrs_dict.get("name")
        if not name:
            return
        input_type = (attrs_dict.get("type") or "text").lower()
        if input_type == "file":
            return
        if input_type in {"checkbox", "radio"} and "checked" not in attrs_dict:
            return
        value = attrs_dict.get("value") or ""
        fields = self._current_form["fields"]
        if name not in fields or "checked" in attrs_dict:
            fields[name] = value


class BaseSiteEditor:
    site_name: str = ""
    benchmark: str = "webarena_verified"
    supported_methods: frozenset[str] = frozenset()

    def __init__(self, instance: dict[str, Any], session: requests.Session) -> None:
        self.instance = instance
        self.session = session
        self._cleanup_stack: list[Callable[[], None]] = []

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        raise NotImplementedError

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        raise NotImplementedError

    def preview_context(self, method_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {}

    def cleanup(self) -> None:
        failures: list[str] = []
        for fn in reversed(self._cleanup_stack):
            try:
                fn()
            except Exception as exc:
                logger.exception("editor cleanup op failed for %s", self.site_name)
                failures.append(str(exc) or exc.__class__.__name__)
        self._cleanup_stack.clear()
        if failures:
            raise EditorError(
                "cleanup_failed",
                f"{self.site_name} editor cleanup failed: {'; '.join(failures)}",
            )

    def _push_cleanup(self, fn: Callable[[], None]) -> None:
        self._cleanup_stack.append(fn)

    def _site_url(self) -> str:
        site_url = str(self.instance.get("site_url", "")).rstrip("/")
        if not site_url:
            raise EditorError("missing_site_url", f"{self.site_name} editor requires site_url")
        return site_url

    def _build_headers(self, *, mechanism: str) -> dict[str, str]:
        from worldsim import seeding as seeding_module

        return seeding_module._build_request_headers(self.instance, {}, mechanism=mechanism)

    def _ensure_form_login(self) -> None:
        from worldsim import seeding as seeding_module

        seeding_module._perform_web_login_if_needed(self.session, self.instance, "form")

    def _api_request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_missing: bool = False,
    ) -> Any:
        response = self.session.request(
            method.strip().upper(),
            f"{self._site_url()}{path}",
            headers=self._build_headers(mechanism="api"),
            json=json_body,
            params=params,
            timeout=30,
            allow_redirects=False,
        )
        if allow_missing and response.status_code == 404:
            return None
        if 300 <= response.status_code < 400:
            raise EditorError(
                "unexpected_redirect",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
            )
        if response.status_code in {401, 403}:
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
            ) from exc
        if not response.text:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    def _api_request_response(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_missing: bool = False,
    ) -> requests.Response | None:
        response = self.session.request(
            method.strip().upper(),
            f"{self._site_url()}{path}",
            headers=self._build_headers(mechanism="api"),
            json=json_body,
            params=params,
            timeout=30,
            allow_redirects=False,
        )
        if allow_missing and response.status_code == 404:
            return None
        if 300 <= response.status_code < 400:
            raise EditorError(
                "unexpected_redirect",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
            )
        if response.status_code in {401, 403}:
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
            ) from exc
        return response

    def _form_get(self, path: str, *, allow_missing: bool = False) -> requests.Response | None:
        self._ensure_form_login()
        response = self.session.get(
            f"{self._site_url()}{path}",
            headers=self._build_headers(mechanism="form"),
            timeout=30,
            allow_redirects=False,
        )
        if allow_missing and response.status_code == 404:
            return None
        if 300 <= response.status_code < 400:
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor form GET {path} returned HTTP {response.status_code}",
            )
        if response.status_code in {401, 403}:
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor form GET {path} returned HTTP {response.status_code}",
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor form GET {path} returned HTTP {response.status_code}",
            ) from exc
        return response

    def _submit_form(self, path: str, body_form: dict[str, Any]) -> requests.Response:
        from worldsim import seeding as seeding_module

        self._ensure_form_login()
        url = f"{self._site_url()}{path}"
        headers = self._build_headers(mechanism="form")
        form_body = seeding_module._prepare_form_body(
            "POST",
            url,
            headers,
            body_form,
            self.instance,
            self.session,
        )
        response = self.session.request(
            "POST",
            url,
            headers=headers,
            data=form_body,
            timeout=30,
            allow_redirects=False,
        )
        if response.status_code in {401, 403}:
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor form POST {path} returned HTTP {response.status_code}",
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor form POST {path} returned HTTP {response.status_code}",
            ) from exc
        return response

    def _submit_exact_form(
        self,
        action_path: str,
        form_fields: dict[str, Any],
        *,
        multipart: bool = False,
        refresh_on_rejection: Callable[[], tuple[str, dict[str, Any], bool]] | None = None,
    ) -> requests.Response:
        self._ensure_form_login()
        attempts = 0
        while True:
            url = self._resolve_form_action(action_path)
            headers = self._build_headers(mechanism="form")
            request_kwargs: dict[str, Any]
            if multipart:
                request_kwargs = {
                    "files": [
                        (str(key), (None, "" if value is None else str(value)))
                        for key, value in form_fields.items()
                    ]
                }
            else:
                request_kwargs = {"data": form_fields}
            response = self.session.request(
                "POST",
                url,
                headers=headers,
                timeout=30,
                allow_redirects=False,
                **request_kwargs,
            )
            if (
                response.status_code in {403, 419, 422}
                and refresh_on_rejection is not None
                and attempts == 0
            ):
                attempts += 1
                action_path, form_fields, multipart = refresh_on_rejection()
                continue
            break
        if response.status_code in {401, 403}:
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor form POST {action_path} returned HTTP {response.status_code}",
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor form POST {action_path} returned HTTP {response.status_code}",
            ) from exc
        return response

    def _fetch_form_state(
        self,
        path: str,
        *,
        action_contains: str | None = None,
        required_fields: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        response = self._form_get(path)
        parser = _FormParser()
        parser.feed(response.text)
        action_filter = (action_contains or "").lower()
        for form in parser.forms:
            action = str(form.get("action") or "")
            if action_filter and action_filter not in action.lower():
                continue
            available = set(form.get("fields", {})) | set(form.get("select_options", {}))
            if all(field in available for field in required_fields):
                return form
        raise EditorError(
            "form_missing",
            f"{self.site_name} editor could not find expected form for {path}",
        )

    def _resolve_form_action(self, action_path: str) -> str:
        site_url = self._site_url()
        resolved = urljoin(f"{site_url}/", action_path or "")
        site_origin = urlparse(site_url)
        action_origin = urlparse(resolved)
        if (site_origin.scheme, site_origin.netloc) != (action_origin.scheme, action_origin.netloc):
            raise EditorError(
                "cross_origin_form_action",
                f"{self.site_name} editor refused cross-origin form action {resolved!r}",
            )
        if action_origin.path:
            suffix = action_origin.path
            if action_origin.query:
                suffix += f"?{action_origin.query}"
            return f"{site_url}{suffix}"
        return site_url

    @staticmethod
    def _require_args(args: dict[str, Any], *required: str) -> None:
        missing = [
            key
            for key in required
            if args.get(key) in (None, "") and not isinstance(args.get(key), list)
        ]
        if missing:
            raise EditorError("invalid_args", "missing required args: " + ", ".join(missing))

    @staticmethod
    def _nested_lookup(value: Any, path: tuple[str, ...]) -> Any:
        current = value
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @classmethod
    def current_username_preview(cls, instance: dict[str, Any]) -> str:
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
        return "current-user"

    @staticmethod
    def _quote(value: Any) -> str:
        return quote(str(value), safe="")
