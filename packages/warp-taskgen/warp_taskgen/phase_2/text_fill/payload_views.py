from __future__ import annotations

from html.parser import HTMLParser

from warp_taskgen.phase_2.text_fill.constants import (
    _MARKDOWN_SYSTEM_BLOCK_RE,
    _STYLE_OFFSET_RE,
)


class _OffscreenHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._stack: list[bool] = []
        self.found = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        style = dict(attrs).get("style") or ""
        self._stack.append(_is_offscreen_style(style))

    def handle_endtag(self, tag: str) -> None:
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if any(self._stack) and data.strip():
            self.found = True


class _PayloadViewHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._hidden_stack: list[bool] = []
        self.visible_chunks: list[str] = []
        self.hidden_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        parent_hidden = self._hidden_stack[-1] if self._hidden_stack else False
        current_hidden = parent_hidden or _is_offscreen_style(attrs_dict.get("style") or "")
        self._hidden_stack.append(current_hidden)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._hidden_stack:
            self._hidden_stack.pop()

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self._hidden_stack and self._hidden_stack[-1]:
            self.hidden_chunks.append(data)
        else:
            self.visible_chunks.append(data)


def _contains_offscreen_css(rendered_payload: str) -> bool:
    parser = _OffscreenHTMLParser()
    parser.feed(rendered_payload)
    return parser.found


def _is_offscreen_style(style: str) -> bool:
    normalized = style.lower().replace(" ", "")
    if "position:absolute" not in normalized:
        return False
    offsets = [int(match.group(2)) for match in _STYLE_OFFSET_RE.finditer(style)]
    return any(offset <= -1000 for offset in offsets)


def _normalize_text_snapshot(text: str) -> str:
    return " ".join(text.split())


def _payload_text_views(rendered_payload: str, concealment: str) -> tuple[str, str]:
    if concealment == "markdown_fenced_system":
        concealed_blocks = [
            match.group(1) for match in _MARKDOWN_SYSTEM_BLOCK_RE.finditer(rendered_payload)
        ]
        visible = _MARKDOWN_SYSTEM_BLOCK_RE.sub("", rendered_payload)
        return (visible, "\n".join(concealed_blocks))
    if concealment == "offscreen_css":
        parser = _PayloadViewHTMLParser()
        parser.feed(rendered_payload)
        return ("".join(parser.visible_chunks), "".join(parser.hidden_chunks))
    return (rendered_payload, rendered_payload)
