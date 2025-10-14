import time
import base64
from typing import List, Dict, Literal, Optional
from playwright.sync_api import sync_playwright, Browser, Page
from utils import check_blocklisted_url

# Optional: key mapping if your model uses "CUA" style keys
CUA_KEY_TO_PLAYWRIGHT_KEY = {
    "/": "Divide",
    "\\": "Backslash",
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "backspace": "Backspace",
    "capslock": "CapsLock",
    "cmd": "Meta",
    "ctrl": "Control",
    "delete": "Delete",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "shift": "Shift",
    "space": " ",
    "super": "Meta",
    "tab": "Tab",
    "win": "Meta",
}


class BasePlaywrightComputer:
    """
    Abstract base for Playwright-based computers:

      - Subclasses override `_get_browser_and_page()` to do local or remote connection,
        returning (Browser, Page).
      - This base class handles context creation (`__enter__`/`__exit__`),
        plus standard "Computer" actions like click, scroll, etc.
      - We also have extra browser actions: `goto(url)` and `back()`.
    """

    def get_environment(self):
        return "browser"

    def get_dimensions(self):
        return (1024, 768)

    def __init__(self):
        self._playwright = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def __enter__(self):
        # Start Playwright and call the subclass hook for getting browser/page
        self._playwright = sync_playwright().start()
        self._browser, self._page = self._get_browser_and_page()

        # Set up network interception to flag URLs matching domains in BLOCKED_DOMAINS
        def handle_route(route, request):

            url = request.url
            if check_blocklisted_url(url):
                print(f"Flagging blocked domain: {url}")
                route.abort()
            else:
                route.continue_()

        self._page.route("**/*", handle_route)

        # Ensure any popup inherits routing and becomes the active page
        def on_popup(popup: Page):
            popup.route("**/*", handle_route)
            popup.wait_for_load_state("domcontentloaded", timeout=10000)
            self._page = popup  # switch focus to the popup

        self._page.on("popup", on_popup)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def get_current_url(self) -> str:
        return self._page.url

    # --- Common "Computer" actions ---
    def screenshot(self) -> str:
        """Capture only the viewport (not full_page)."""
        png_bytes = self._page.screenshot(full_page=False)
        return base64.b64encode(png_bytes).decode("utf-8")

    def click(self, x: int, y: int, button: str = "left") -> None:
        match button:
            case "back":
                self.back()
            case "forward":
                self.forward()
            case "wheel":
                self._page.mouse.wheel(x, y)
            case _:
                button_mapping = {"left": "left", "right": "right"}
                button_type = button_mapping.get(button, "left")
                self._page.mouse.click(x, y, button=button_type)

    def double_click(self, x: int, y: int) -> None:
        self._page.mouse.dblclick(x, y)

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._page.mouse.move(x, y)
        self._page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

    def type(self, text: str) -> None:
        self._page.keyboard.type(text)

    def wait(self, ms: int = 1000) -> None:
        time.sleep(ms / 1000)

    def move(self, x: int, y: int) -> None:
        self._page.mouse.move(x, y)

    def keypress(self, keys: List[str]) -> None:
        mapped_keys = [CUA_KEY_TO_PLAYWRIGHT_KEY.get(key.lower(), key) for key in keys]
        for key in mapped_keys:
            self._page.keyboard.down(key)
        for key in reversed(mapped_keys):
            self._page.keyboard.up(key)

    def drag(self, path: List[Dict[str, int]]) -> None:
        if not path:
            return
        self._page.mouse.move(path[0]["x"], path[0]["y"])
        self._page.mouse.down()
        for point in path[1:]:
            self._page.mouse.move(point["x"], point["y"])
        self._page.mouse.up()

    # --- Extra browser-oriented actions ---
    def goto(self, url: str) -> None:
        try:
            return self._page.goto(url)
        except Exception as e:
            print(f"Error navigating to {url}: {e}")

    def back(self) -> None:
        return self._page.go_back()

    def forward(self) -> None:
        return self._page.go_forward()

    def _resolve_single(self, selector: str, timeout_ms: int = 10_000):
        """Return a Locator that is guaranteed to refer to a single element.
        If the selector matches multiple elements, pick the first visible one."""
        loc = self._page.locator(selector)
        try:
            # If strict mode passes (single match), great.
            loc.wait_for(state="attached", timeout=timeout_ms)
            count = loc.count()
            if count == 1:
                return loc
            elif count > 1:
                # Prefer first visible element
                try:
                    vis = loc.filter(has=None, has_text=None).first
                    vis.wait_for(state="visible", timeout=timeout_ms)
                    return vis
                except Exception:
                    # Fallback: just take .first()
                    return loc.first
            else:
                raise RuntimeError(f"No elements found for selector: {selector!r}")
        except Exception:
            # Fallback via wait_for_selector (returns first)
            handle = self._page.wait_for_selector(
                selector, state="attached", timeout=timeout_ms
            )
            # Convert ElementHandle back to Locator
            return self._page.locator(selector).first

    # --- Selector-based DOM helpers ---
    def dom_click(
        self, selector: str, button: str = "left", timeout_ms: int = 10_000
    ) -> None:
        target = self._resolve_single(selector, timeout_ms)
        try:
            target.wait_for(state="visible", timeout=timeout_ms)
        except Exception:
            try:
                target.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass
        btn = "left" if button not in ("right", "middle") else button
        try:
            with self._page.expect_popup(timeout=1500) as pop:
                target.click(button=btn, timeout=timeout_ms)
            popup = pop.value
            popup.wait_for_load_state("domcontentloaded", timeout=10000)
            self._page = popup
        except Exception:
            target.click(button=btn, timeout=timeout_ms)

    def dom_type(
        self, selector: str, text: str, submit: bool = False, timeout_ms: int = 10_000
    ) -> None:
        target = self._resolve_single(selector, timeout_ms)
        target.wait_for(state="visible", timeout=timeout_ms)
        try:
            target.fill(text, timeout=timeout_ms)
        except Exception:
            target.click(timeout=timeout_ms)
            self._page.keyboard.type(text)
        if submit:
            self._page.keyboard.press("Enter")

    def dom_read_text(self, selector: str, timeout_ms: int = 10_000) -> str:
        target = self._resolve_single(selector, timeout_ms)
        target.wait_for(state="visible", timeout=timeout_ms)
        try:
            txt = target.inner_text(timeout=timeout_ms)
        except Exception:
            handle = target.element_handle()
            txt = handle.text_content() if handle else ""
        return (txt or "").strip()

    def dom_wait_for(
        self,
        selector: str,
        state: Literal["visible", "hidden", "attached", "detached"] = "visible",
        timeout_ms: int = 10_000,
    ) -> bool:
        target = self._resolve_single(selector, timeout_ms)
        target.wait_for(state=state, timeout=timeout_ms)
        return True

    def dom_screenshot(self, selector: str, timeout_ms: int = 10_000) -> Optional[str]:
        target = self._resolve_single(selector, timeout_ms)
        target.wait_for(state="visible", timeout=timeout_ms)
        png = target.screenshot(timeout=timeout_ms)
        return base64.b64encode(png).decode("utf-8")

    def dom_exists(self, selector: str, timeout_ms: int = 0) -> bool:
        try:
            if timeout_ms and timeout_ms > 0:
                # Try to find quickly with a short wait
                self._page.wait_for_selector(
                    selector, timeout=timeout_ms, state="attached"
                )
                return True
            return self._page.query_selector(selector) is not None
        except Exception:
            return False

    # --- Storage helpers ---
    def read_storage(
        self, key: str, scope: Literal["auto", "session", "local"] = "auto"
    ) -> str:
        script = """
            (key, scope) => {
                try {
                    if (scope === 'session') {
                        return window.sessionStorage.getItem(key) || '';
                    }
                    if (scope === 'local') {
                        return window.localStorage.getItem(key) || '';
                    }
                    // auto: prefer sessionStorage, then localStorage
                    const s = window.sessionStorage.getItem(key);
                    if (s !== null) return s;
                    const l = window.localStorage.getItem(key);
                    return l !== null ? l : '';
                } catch (e) {
                    return '';
                }
            }
        """
        try:
            value = self._page.evaluate(script, key, scope)
            return value or ""
        except Exception:
            return ""

    # --- Subclass hook ---
    def _get_browser_and_page(self) -> tuple[Browser, Page]:
        """Subclasses must implement, returning (Browser, Page)."""
        raise NotImplementedError
