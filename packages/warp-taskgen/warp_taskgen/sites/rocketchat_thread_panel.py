"""Measured Rocket.Chat 5.3 thread-panel HTML parser.

The parser extracts exact direct UserMessage rows from the scoped thread
panel. It is deliberately separate from readback interpretation: serialized
HTML establishes DOM identity/body/order only; the render executor must supply
the independent non-zero geometry witness for Painted Visibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from warp_taskgen.sites.readback import identity_token_text

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
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

def _class_tokens(value: object) -> frozenset[str]:
    if not isinstance(value, str):
        return frozenset()
    return frozenset(part for part in value.split() if part)


def _css_attr_value(value: object) -> str | None:
    """Return an attribute value safe to interpolate into a CSS selector."""

    text = identity_token_text(value)
    if text is None or _ID_RE.fullmatch(text) is None:
        return None
    return text


@dataclass
class _ThreadPanelRow:
    message_id: str
    thread_id: str | None
    author: str | None
    body_parts: list[str]

    @property
    def body(self) -> str:
        # ``HTMLParser(convert_charrefs=True)`` has already decoded entities.
        # Keep interior spacing intact: the writer's body digest is over the
        # exact string, and the TAC composer emits one-line message bodies.
        return "".join(self.body_parts).strip()


@dataclass
class _ThreadPanelFrame:
    tag: str
    classes: frozenset[str] = frozenset()
    thread_view: bool = False
    thread_list: bool = False
    row: _ThreadPanelRow | None = None
    body_marker: bool = False


class _RocketChatThreadPanelParser(HTMLParser):
    """Extract only direct UserMessage rows from the measured thread panel."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.frames: list[_ThreadPanelFrame] = []
        self.rows: list[_ThreadPanelRow] = []
        self.malformed = False

    @property
    def _inside_thread_view(self) -> bool:
        return any(frame.thread_view for frame in self.frames)

    @property
    def _inside_thread_list(self) -> bool:
        return any(frame.thread_list for frame in self.frames)

    @property
    def _current_row(self) -> _ThreadPanelRow | None:
        for frame in reversed(self.frames):
            if frame.row is not None:
                return frame.row
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.casefold()
        attr_map = {str(key).casefold(): str(value or "") for key, value in attrs}
        classes = _class_tokens(attr_map.get("class"))
        frame = _ThreadPanelFrame(
            tag=tag_name,
            classes=classes,
            thread_view="rcx-thread-view" in classes and not self._inside_thread_view,
            thread_list=(
                tag_name == "ul"
                and "thread" in classes
                and self._inside_thread_view
                and any("js-scroll-thread" in ancestor.classes for ancestor in self.frames)
            ),
        )
        if (
            tag_name == "li"
            and self._inside_thread_list
            and self.frames
            and self.frames[-1].thread_list
            and attr_map.get("data-qa-id", "").casefold() == "usermessage"
        ):
            message_id = _css_attr_value(attr_map.get("data-id"))
            if message_id is None:
                self.malformed = True
            else:
                thread_id = _css_attr_value(attr_map.get("data-tmid"))
                if attr_map.get("data-tmid", "").strip() and thread_id is None:
                    self.malformed = True
                author = attr_map.get("data-username", "").strip() or None
                frame.row = _ThreadPanelRow(message_id, thread_id, author, [])
        if self._current_row is not None:
            frame.body_marker = attr_map.get("data-qa-type") == "message-body"
            if frame.body_marker and not any(parent.body_marker for parent in self.frames):
                # The body marker must be inside the row currently being
                # collected; nested markers are ignored rather than doubled.
                pass
            for key in ("data-username", "data-author"):
                candidate = attr_map.get(key, "").strip()
                if candidate and self._current_row.author is None:
                    self._current_row.author = candidate
        self.frames.append(frame)
        if tag_name in _VOID_TAGS:
            # ``HTMLParser`` does not emit an end tag for void elements.
            self._close_frame(tag_name)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in _VOID_TAGS and self.frames:
            self._close_frame(tag.casefold())

    def _close_frame(self, tag_name: str) -> None:
        if not self.frames:
            self.malformed = True
            return
        frame = self.frames.pop()
        if frame.tag != tag_name:
            self.malformed = True
        if frame.row is not None:
            if not frame.row.body:
                self.malformed = True
            self.rows.append(frame.row)

    def handle_endtag(self, tag: str) -> None:
        self._close_frame(tag.casefold())

    def handle_data(self, data: str) -> None:
        row = self._current_row
        if row is None or not data:
            return
        if any(frame.body_marker for frame in self.frames):
            row.body_parts.append(data)
