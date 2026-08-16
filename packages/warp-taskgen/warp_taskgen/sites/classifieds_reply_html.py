"""Exact ordinary-page reply identity extraction for Classifieds."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlsplit

_ID_RE = re.compile(r"^[1-9][0-9]*$")
_ACTOR_SUFFIX_RE = re.compile(r"(?:^|\s)by\s+(.+)$", re.IGNORECASE)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


def _required_id(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError("listing id must be a positive integer")
    text = str(value or "").strip()
    if not _ID_RE.fullmatch(text):
        raise ValueError("listing id must be a positive integer")
    return text


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def normalize_reply_body(value: str) -> str:
    """Normalize editor and ordinary-reader body text identically."""

    return _normalize_text(value)


def _has_class(value: str, token: str) -> bool:
    return token.casefold() in {part.casefold() for part in (value or "").split()}


@dataclass
class _CommentBlock:
    attrs: dict[str, str]
    body_text: list[str]
    heading_text: list[str]
    actor_text: list[str]
    actor_values: list[str]
    reply_ids: list[str]
    delete_ids: list[str]
    malformed_identity: bool = False


class _RenderedCommentParser(HTMLParser):
    """Parse outer ``div.comment`` blocks and their own identity evidence."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[_CommentBlock] = []
        self._current: _CommentBlock | None = None
        self._depth = 0
        self._open: list[tuple[str, bool, bool, bool]] = []
        self._ignored_open: list[str] = []
        self._heading_depth = 0
        self._body_depth = 0
        self._actor_depth = 0

    @property
    def has_incomplete_comment(self) -> bool:
        """Whether an outer comment opened without a complete closing boundary."""

        return self._current is not None or self._depth != 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {str(key).lower(): str(value or "") for key, value in attrs}
        tag_name = tag.casefold()
        if self._current is None:
            if tag_name == "div" and _has_class(attr_map.get("class", ""), "comment"):
                self._current = _CommentBlock(attr_map, [], [], [], [], [], [])
                self._depth = 1
                self._open = [(tag_name, False, False, False)]
            return

        if self._ignored_open:
            if tag_name in _VOID_TAGS:
                return
            self._ignored_open.append(tag_name)
            self._depth += 1
            return
        if tag_name == "div" and _has_class(attr_map.get("class", ""), "comment"):
            self._ignored_open.append(tag_name)
            self._depth += 1
            self._open.append((tag_name, False, False, False))
            return

        self._depth += 1
        actor_marker = (
            any(key in attr_map for key in ("data-author", "data-username", "author", "username"))
            or _has_class(attr_map.get("class", ""), "author")
            or _has_class(attr_map.get("class", ""), "username")
        )
        heading_marker = tag_name == "h3"
        # In the pinned Sigma theme the reply body is the only direct comment
        # paragraph without a class. Profile-image, rating, delete, and reply
        # action rows are all classed and must not contaminate the body digest.
        body_marker = tag_name == "p" and not attr_map.get("class", "").strip()
        self._open.append((tag_name, actor_marker, heading_marker, body_marker))
        if actor_marker:
            self._actor_depth += 1
            for key in ("data-author", "data-username", "author", "username"):
                value = attr_map.get(key, "").strip()
                if value:
                    self._current.actor_values.append(value)
        if heading_marker:
            self._heading_depth += 1
        if body_marker:
            self._body_depth += 1
        if tag_name == "a":
            classes = attr_map.get("class", "")
            if _has_class(classes, "comment-reply") and "data-id" in attr_map:
                value = attr_map.get("data-id", "").strip()
                if _ID_RE.fullmatch(value):
                    self._current.reply_ids.append(value)
                else:
                    self._current.malformed_identity = True
            identity = _delete_identity(attr_map.get("href", ""))
            if identity is not None:
                self._current.delete_ids.append(identity)
        if tag_name in _VOID_TAGS:
            self._close_entry()

    def _close_entry(self) -> None:
        entry = self._open.pop() if self._open else ("", False, False, False)
        _opened_tag, actor_marker, heading_marker, body_marker = entry
        if actor_marker:
            self._actor_depth = max(0, self._actor_depth - 1)
        if heading_marker:
            self._heading_depth = max(0, self._heading_depth - 1)
        if body_marker:
            self._body_depth = max(0, self._body_depth - 1)
        self._depth -= 1

    def handle_endtag(self, tag: str) -> None:
        if self._current is None or self._depth <= 0:
            return
        if self._ignored_open:
            self._ignored_open.pop()
            self._depth -= 1
            if not self._ignored_open and self._open:
                self._open.pop()
            return
        self._close_entry()
        if self._depth <= 0:
            self.blocks.append(self._current)
            self._current = None
            self._depth = 0
            self._open = []
            self._ignored_open = []
            self._heading_depth = self._body_depth = self._actor_depth = 0

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        # ``handle_starttag`` closes HTML void elements itself. Calling the
        # matching end handler again would close the surrounding comment early
        # for the pinned theme's ``<img ... />`` profile image.
        if tag.casefold() not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._current is None or not data:
            return
        if self._body_depth > 0:
            self._current.body_text.append(data)
        if self._heading_depth > 0:
            self._current.heading_text.append(data)
        if self._actor_depth > 0:
            self._current.actor_text.append(data)


class _ListingSurfaceParser(HTMLParser):
    """Recognize the pinned ordinary listing reply form without executing it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._depth = 0
        self.malformed = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.casefold()
        attr_map = {str(key).casefold(): str(value or "") for key, value in attrs}
        if tag_name == "form":
            if self._current is not None:
                self.malformed = True
                return
            self._current = {
                "__action": attr_map.get("action", "").strip(),
                "__method": attr_map.get("method", "").strip().casefold(),
            }
            self._depth = 1
            return
        if self._current is None:
            return
        if tag_name not in _VOID_TAGS:
            self._depth += 1
        if tag_name == "input":
            name = attr_map.get("name", "").strip()
            if name:
                if name in self._current:
                    self.malformed = True
                self._current[name] = attr_map.get("value", "").strip()
        elif tag_name == "textarea" and attr_map.get("name", "").strip() == "body":
            self._current["__body_control"] = "textarea"

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag.casefold() in _VOID_TAGS:
            return
        self._depth -= 1
        if tag.casefold() == "form":
            if self._depth != 0:
                self.malformed = True
            self.forms.append(self._current)
            self._current = None
            self._depth = 0

    @property
    def incomplete(self) -> bool:
        return self._current is not None or self._depth != 0


def _delete_identity(href: str) -> str | None:
    if not href:
        return None
    try:
        parts = urlsplit(href)
        query = parse_qsl(parts.query, keep_blank_values=True)
    except ValueError:
        return None
    path = parts.path.casefold()
    if "delete" not in path and not any(
        key.casefold() in {"action", "delete", "page"} and "delete" in value.casefold()
        for key, value in query
    ):
        return None
    values = {key.casefold(): value for key, value in query}
    action = values.get("action", "").casefold()
    page = values.get("page", "").casefold()
    # The pinned Classifieds owner route identifies the parent listing as
    # ``id`` and the child reply as ``comment``.  Treating ``id`` as the reply
    # would make the ordinary-owner markup disagree with its data-id anchor.
    if action == "delete_comment":
        value = values.get("comment", "")
        return value if _ID_RE.fullmatch(value) else None
    # Retain the older direct delete page shape as a compatibility read seam.
    if page == "delete_comment":
        value = values.get("comment_id") or values.get("item_comment_id") or values.get("id", "")
        return value if _ID_RE.fullmatch(value) else None
    return None


def _block_data_id(block: _CommentBlock) -> str | None:
    if block.malformed_identity:
        return None
    identities = [value for value in block.reply_ids if _ID_RE.fullmatch(value)]
    identities.extend(value for value in block.delete_ids if _ID_RE.fullmatch(value))
    unique = set(identities)
    return next(iter(unique)) if len(unique) == 1 else None


def _block_actor_matches(block: _CommentBlock, expected: str) -> bool:
    expected_text = _normalize_text(expected)
    candidates = {_normalize_text(value) for value in block.actor_values if value.strip()}
    candidates.update(_normalize_text(value) for value in block.actor_text if value.strip())
    heading = _normalize_text(" ".join(block.heading_text))
    suffix = _ACTOR_SUFFIX_RE.search(heading)
    if suffix is not None:
        rendered_actor = _normalize_text(suffix.group(1)).removesuffix(":")
        return rendered_actor == expected_text
    return expected_text in candidates


def extract_listing_reply_id(
    html: str,
    *,
    actor: str,
    body: str,
    signature: str | None = None,
    listing_id: str | None = None,
) -> str | None:
    """Return one exact ID from one matching outer rendered comment block."""

    if not isinstance(html, str) or not html.strip():
        return None
    try:
        actor_text = _required_text(actor, "actor")
        body_text = _normalize_text(_required_text(body, "reply body"))
        signature_text = (
            _normalize_text(_required_text(signature, "reply signature"))
            if signature is not None
            else None
        )
        listing_text = _required_id(listing_id) if listing_id is not None else None
    except ValueError:
        return None
    if signature_text is not None and signature_text not in body_text:
        return None
    parser = _RenderedCommentParser()
    try:
        parser.feed(html)
        parser.close()
    except (TypeError, ValueError):
        return None

    matches: list[str] = []
    for block in parser.blocks:
        block_id = _block_data_id(block)
        if block_id is None or not _block_actor_matches(block, actor_text):
            continue
        rendered_body = _normalize_text(" ".join(block.body_text))
        if rendered_body != body_text:
            continue
        if signature_text is not None and signature_text not in rendered_body:
            continue
        if listing_text is not None:
            rendered_listing = block.attrs.get("data-listing-id") or block.attrs.get("data-item-id")
            if rendered_listing and rendered_listing.strip() != listing_text:
                continue
        matches.append(block_id)
    return matches[0] if len(matches) == 1 else None


def rendered_listing_reply_id_presence(html: str, reply_id: object) -> bool | None:
    """Return exact reply-ID presence, or ``None`` when absence is unprovable."""

    try:
        expected = _required_id(reply_id)
    except ValueError:
        return None
    if not isinstance(html, str) or not html.strip():
        return None
    identities = rendered_listing_reply_ids(html)
    return None if identities is None else expected in identities


def rendered_listing_reply_ids(html: str) -> frozenset[str] | None:
    """Return every stable rendered reply ID, or ``None`` if the page is ambiguous.

    This is intentionally stricter than a best-effort scrape: a post-submit
    recovery may delete only a single ID in the exact ``post - pre`` delta.
    """

    if not isinstance(html, str) or not html.strip():
        return None
    parser = _RenderedCommentParser()
    try:
        parser.feed(html)
        parser.close()
    except (TypeError, ValueError):
        return None
    if parser.has_incomplete_comment or any(block.malformed_identity for block in parser.blocks):
        return None
    identities: set[str] = set()
    for block in parser.blocks:
        identity = _block_data_id(block)
        # A visible comment with missing or conflicting ordinary-page identity
        # makes exact absence unknowable; it is not evidence that the target ID
        # disappeared.
        if identity is None:
            return None
        identities.add(identity)
    return frozenset(identities)


def rendered_listing_surface_present(html: str, listing_id: object, *, origin: str) -> bool:
    """Return whether HTML proves the exact ordinary listing reply surface."""

    try:
        expected = _required_id(listing_id)
    except ValueError:
        return False
    if not isinstance(html, str) or not html.strip():
        return False
    parser = _ListingSurfaceParser()
    try:
        parser.feed(html)
        parser.close()
    except (TypeError, ValueError):
        return False
    if parser.malformed or parser.incomplete:
        return False
    try:
        expected_origin = urlsplit(origin.rstrip("/"))
    except ValueError:
        return False

    def valid_action(value: str) -> bool:
        try:
            action = urlsplit(value)
        except ValueError:
            return False
        if action.path != "/index.php" or action.query or action.fragment:
            return False
        if not action.scheme and not action.netloc:
            return True
        return (action.scheme, action.netloc) == (expected_origin.scheme, expected_origin.netloc)

    matches = [
        form
        for form in parser.forms
        if form.get("action") == "add_comment"
        and form.get("page") == "item"
        and form.get("id") == expected
        and valid_action(form.get("__action", ""))
        and form.get("__method") == "post"
        and form.get("__body_control") == "textarea"
    ]
    return len(matches) == 1


__all__ = [
    "extract_listing_reply_id",
    "normalize_reply_body",
    "rendered_listing_reply_id_presence",
    "rendered_listing_reply_ids",
    "rendered_listing_surface_present",
]
