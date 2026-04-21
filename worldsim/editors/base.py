from __future__ import annotations

import logging
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests

from worldsim.http_proxy import ProxyInfo, proxy_info_from_session

logger = logging.getLogger(__name__)

_RESPONSE_SNIPPET_MAX_CHARS = 512


def _redirect_location_path(response: requests.Response) -> str:
    location = response.headers.get("Location")
    if not isinstance(location, str) or not location.strip():
        return ""
    return (urlparse(location).path or "").strip().lower()


def _looks_like_login_redirect(response: requests.Response) -> bool:
    path = _redirect_location_path(response)
    if not path:
        return False
    return any(
        token in path
        for token in (
            "/login",
            "/sign_in",
            "/users/sign_in",
            "/session",
        )
    )


def _same_logical_origin(
    site: Any,
    action: Any,
    *,
    proxy_info: ProxyInfo | None = None,
) -> bool:
    """Treat proxy-rewritten origins as equivalent to their raw origin.

    When a verification proxy is installed (see ``worldsim.http_proxy``),
    an upstream backend may bake its own proxy-origin URL into HTML form
    actions — e.g. Magento's ``web/{unsecure,secure}/base_url`` points at
    ``http://<ip>:17770/`` even though ``site_url`` in the instance config
    is ``http://<ip>:7770/``. The two URLs name the same logical origin;
    only the port differs by ``port_offset``. Without this relaxation, the
    editor's cross-origin check rejects every form on a Magento page.

    We consider origins equivalent when scheme matches, hostname matches,
    and the action port is one of: the site port, site_port+port_offset,
    site_port-port_offset. The last two cover both directions (raw→proxy
    and proxy→raw). We only relax when ``proxy_info`` is supplied AND the
    raw port is on its ``site_ports`` list — so an arbitrary cross-port
    form action (e.g. an exfil attempt to a different service) is still
    rejected.

    The ``proxy_info`` parameter is threaded in explicitly rather than read
    from a module-level global. Callers derive it from the ``requests.Session``
    they already hold via ``http_proxy.proxy_info_from_session(session, url)``;
    if that returns ``None``, pass ``None`` and strict same-origin semantics
    apply. This keeps the cross-origin decision a function of the session
    the editor was constructed with, not of hidden process state.
    """
    if site.scheme != action.scheme:
        return False
    if (site.hostname or "") != (action.hostname or ""):
        return False
    site_port = site.port
    action_port = action.port
    if site_port == action_port:
        return True
    if site_port is None or action_port is None:
        return False
    if proxy_info is None:
        return False
    if site_port not in proxy_info.site_ports:
        return False
    return action_port in {
        site_port + proxy_info.port_offset,
        site_port - proxy_info.port_offset,
    }


def _truncate_snippet(text: str | None) -> str | None:
    if not text:
        return None
    if len(text) <= _RESPONSE_SNIPPET_MAX_CHARS:
        return text
    return text[:_RESPONSE_SNIPPET_MAX_CHARS] + "…"


class EditorError(RuntimeError):
    def __init__(
        self,
        kind: str,
        detail: str,
        *,
        http_status: int | None = None,
        response_snippet: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.http_status = http_status
        self.response_snippet = response_snippet


# Informational: the canonical taxonomy raised by editors + seeding. No code
# currently validates against a closed set; this is a human-reference list.
EDITOR_ERROR_KINDS: frozenset[str] = frozenset(
    {
        "auth_missing",
        "cleanup_failed",
        "content_policy",
        "cross_origin_form_action",
        "empty_seed",
        "field_required",
        "form_missing",
        "invalid_args",
        "length_exceeded",
        "missing_site_url",
        "request_failed",
        "schema_mismatch",
        "site_mismatch",
        "unexpected_redirect",
        "unreachable",
        "unsupported_method",
        "unsupported_site",
    }
)


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

    def _classify_4xx_response(
        self,
        method: str,
        path: str,
        response: requests.Response,
    ) -> tuple[str, str] | None:
        """Classify a 4xx response beyond the generic ``request_failed`` kind.

        Subclasses override this to detect platform-level rejection patterns
        (``length_exceeded``, ``field_required``, ``content_policy``). The
        base implementation has no opinion and returns ``None`` so callers
        fall through to the existing generic ``request_failed`` raise.

        Called only for 4xx responses that are not 401 or 403 (auth errors
        are already classified separately).
        """
        return None

    def _raise_classified_4xx(
        self,
        method: str,
        path: str,
        response: requests.Response,
    ) -> None:
        """Raise a classified ``EditorError`` for a 4xx response, if possible.

        No-op for non-4xx, 401, 403, or when the subclass hook returns
        ``None`` (the caller then proceeds to its existing
        ``raise_for_status``/``request_failed`` branch).
        """
        status = response.status_code
        if not (400 <= status < 500) or status in {401, 403}:
            return
        try:
            classified = self._classify_4xx_response(method, path, response)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "%s editor _classify_4xx_response raised; falling back to request_failed",
                self.site_name,
            )
            return
        if classified is None:
            return
        kind, detail = classified
        raise EditorError(
            kind,
            detail,
            http_status=status,
            response_snippet=_truncate_snippet(response.text),
        )

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
        self._raise_classified_4xx(method, path, response)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
                http_status=response.status_code,
                response_snippet=_truncate_snippet(response.text),
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
        self._raise_classified_4xx(method, path, response)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor request for {path} returned HTTP {response.status_code}",
                http_status=response.status_code,
                response_snippet=_truncate_snippet(response.text),
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
        self._raise_classified_4xx("GET", path, response)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor form GET {path} returned HTTP {response.status_code}",
                http_status=response.status_code,
                response_snippet=_truncate_snippet(response.text),
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
        self._raise_classified_4xx("POST", path, response)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor form POST {path} returned HTTP {response.status_code}",
                http_status=response.status_code,
                response_snippet=_truncate_snippet(response.text),
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
        if 300 <= response.status_code < 400 and _looks_like_login_redirect(response):
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor form POST {action_path} redirected to login "
                f"(HTTP {response.status_code})",
                http_status=response.status_code,
                response_snippet=_truncate_snippet(response.text),
            )
        if response.status_code in {401, 403}:
            raise EditorError(
                "auth_missing",
                f"{self.site_name} editor form POST {action_path} returned HTTP {response.status_code}",
            )
        self._raise_classified_4xx("POST", action_path, response)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"{self.site_name} editor form POST {action_path} returned HTTP {response.status_code}",
                http_status=response.status_code,
                response_snippet=_truncate_snippet(response.text),
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
        proxy_info = proxy_info_from_session(self.session, site_url)
        if not _same_logical_origin(site_origin, action_origin, proxy_info=proxy_info):
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
