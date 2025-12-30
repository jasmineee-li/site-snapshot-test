"""This module contains the PlaywrightToolbox class to be used with an Async Playwright Page."""

import importlib.resources
import base64
import os
from pathlib import Path
from typing import Literal, TypedDict, get_args, Type, cast
from playwright.async_api import Page
from asyncio import sleep
from PIL import Image
import io
import aiohttp
from anthropic.types.beta import (
    BetaToolComputerUse20241022Param,
    BetaToolComputerUse20250124Param,
    BetaToolParam,
    BetaToolResultBlockParam,
    BetaTextBlockParam,
    BetaImageBlockParam,
)
from dataclasses import dataclass

TYPING_DELAY_MS = 12
SCROLL_MULTIPLIER_FACTOR = 500
TYPING_GROUP_SIZE = 50

Action_20241022 = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]

Action_20250124 = (
    Action_20241022
    | Literal[
        "left_mouse_down",
        "left_mouse_up",
        "scroll",
        "hold_key",
        "wait",
        "triple_click",
    ]
)

ScrollDirection = Literal["up", "down", "left", "right"]


class ComputerToolOptions(TypedDict):
    """Options for the computer tool."""

    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    """Split a string into chunks of a specific size."""
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None


class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        """Create a new ToolError."""
        self.message = message


class PlaywrightToolbox:
    """Toolbox for interaction between Claude and Async Playwright Page."""

    def __init__(
        self,
        page: Page,
        use_cursor: bool = True,
        screenshot_wait_until: Literal["load", "domcontentloaded", "networkidle"] | None = None,
        beta_version: Literal["20241022", "20250124", "20251124"] = "20251124",
        save_dir: str = "./downloads",
    ):
        """Create a new PlaywrightToolbox.

        Args:
            page: The Async Playwright page to interact with.
            use_cursor: Whether to display the cursor in the screenshots or not.
            screenshot_wait_until: Optional, wait until the page is in a specific state before taking a screenshot. Default does not wait
            beta_version: The version of the beta to use. Default is the latest version (Claude Opus 4.5)
            save_dir: Directory to save downloaded files and screenshots to.
        """
        self.page = page
        self.beta_version = beta_version
        self.save_dir = save_dir
        computer_tool_map: dict[str, Type[BasePlaywrightComputerTool]] = {
            "20241022": PlaywrightComputerTool20241022,
            "20250124": PlaywrightComputerTool20250124,
            "20251124": PlaywrightComputerTool20251124,
        }
        ComputerTool = computer_tool_map[beta_version]
        self.tools: list[
            BasePlaywrightComputerTool
            | PlaywrightSetURLTool
            | PlaywrightBackTool
            | PlaywrightSaveScreenshotTool
            | PlaywrightDownloadImageTool
            | PlaywrightElementScreenshotTool
            | PlaywrightExtractImageUrlTool
            | PlaywrightListImagesOnPageTool
        ] = [
            ComputerTool(page, use_cursor=use_cursor, screenshot_wait_until=screenshot_wait_until),
            PlaywrightSetURLTool(page),
            PlaywrightBackTool(page),
            PlaywrightSaveScreenshotTool(page, save_dir=save_dir),
            PlaywrightDownloadImageTool(save_dir=save_dir),
            PlaywrightElementScreenshotTool(page, save_dir=save_dir),
            PlaywrightExtractImageUrlTool(page),
            PlaywrightListImagesOnPageTool(page),
        ]

    def to_params(self) -> list[BetaToolParam]:
        """Expose the params of all the tools in the toolbox."""
        return [tool.to_params() for tool in self.tools]

    async def run_tool(self, name: str, input: dict, tool_use_id: str) -> BetaToolResultBlockParam:
        """Pick the right tool using `name` and run it."""
        if name not in [tool.name for tool in self.tools]:
            return ToolError(message=f"Unknown tool {name}, only computer use allowed")
        tool = next(tool for tool in self.tools if tool.name == name)
        result = await tool(**input)
        return _make_api_tool_result(tool_use_id=tool_use_id, result=result)


class PlaywrightSetURLTool:
    """Tool to navigate to a specific URL."""

    name: Literal["set_url"] = "set_url"

    def __init__(self, page: Page):
        """Create a new PlaywrightSetURLTool.

        Args:
            page: The Async Playwright page to interact with.
        """
        super().__init__()
        self.page = page

    def to_params(self) -> BetaToolParam:
        """Params describing the tool. Description used by Claude to understand how to this use tool."""
        return BetaToolParam(
            name=self.name,
            description="This tool allows to go directly to a specified URL.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the web page to navigate to.",
                    }
                },
                "required": ["url"],
            },
        )

    async def __call__(self, *, url: str):
        """Trigger goto the chosen url."""
        try:
            await self.page.goto(url)
            return ToolResult()
        except Exception as e:
            return ToolResult(error=str(e))


class PlaywrightBackTool:
    """Tool to navigate to the previous page."""

    name: Literal["previous_page"] = "previous_page"

    def __init__(self, page: Page):
        """Create a new PlaywrightBackTool.

        Args:
            page: The Async Playwright page to interact with.
        """
        super().__init__()
        self.page = page

    def to_params(self) -> BetaToolParam:
        """Params describing the tool. Description used by Claude to understand how to this use tool."""
        return BetaToolParam(
            name=self.name,
            description="This tool navigate to the previous page.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    async def __call__(self):
        """Trigger the back button in the browser."""
        try:
            await self.page.go_back()
            return ToolResult()
        except Exception as e:
            return ToolResult(error=str(e))


class PlaywrightSaveScreenshotTool:
    """Tool to save the current page screenshot to a file."""

    name: Literal["save_screenshot"] = "save_screenshot"

    def __init__(self, page: Page, save_dir: str = "./downloads"):
        """Create a new PlaywrightSaveScreenshotTool.

        Args:
            page: The Async Playwright page to interact with.
            save_dir: Directory to save screenshots to.
        """
        super().__init__()
        self.page = page
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def to_params(self) -> BetaToolParam:
        """Params describing the tool."""
        return BetaToolParam(
            name=self.name,
            description="Save a screenshot of the current page to a file. Returns the file path where the screenshot was saved. Supports full page capture or clipping to a specific region.",
            input_schema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename to save the screenshot as (e.g., 'cat_photo.png'). Will be saved in the downloads directory.",
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Whether to capture the full scrollable page or just the visible viewport. Default is False.",
                    },
                    "clip_x": {
                        "type": "number",
                        "description": "X coordinate of the top-left corner of the clip region (in pixels). Must be used with clip_y, clip_width, and clip_height.",
                    },
                    "clip_y": {
                        "type": "number",
                        "description": "Y coordinate of the top-left corner of the clip region (in pixels). Must be used with clip_x, clip_width, and clip_height.",
                    },
                    "clip_width": {
                        "type": "number",
                        "description": "Width of the clip region (in pixels). Must be used with clip_x, clip_y, and clip_height.",
                    },
                    "clip_height": {
                        "type": "number",
                        "description": "Height of the clip region (in pixels). Must be used with clip_x, clip_y, and clip_width.",
                    },
                },
                "required": ["filename"],
            },
        )

    async def __call__(
        self,
        *,
        filename: str,
        full_page: bool = False,
        clip_x: float | None = None,
        clip_y: float | None = None,
        clip_width: float | None = None,
        clip_height: float | None = None,
    ):
        """Save a screenshot to a file."""
        try:
            # Ensure filename has an extension
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filename += ".png"

            filepath = self.save_dir / filename

            clip = None
            if (
                clip_x is not None
                and clip_y is not None
                and clip_width is not None
                and clip_height is not None
            ):
                clip = {"x": clip_x, "y": clip_y, "width": clip_width, "height": clip_height}
            elif any(p is not None for p in [clip_x, clip_y, clip_width, clip_height]):
                return ToolResult(
                    error="All clip parameters (clip_x, clip_y, clip_width, clip_height) must be provided together."
                )

            # Add timeout to avoid hanging on slow pages (e.g., waiting for fonts)
            await self.page.screenshot(
                path=str(filepath), full_page=full_page, clip=clip, timeout=10000
            )
            return ToolResult(output=f"Screenshot saved to: {filepath.absolute()}")
        except Exception as e:
            return ToolResult(
                error=f"Failed to save screenshot: {str(e)}. Consider navigating to a different page."
            )


class PlaywrightDownloadImageTool:
    """Tool to download an image from a URL and save it to a file."""

    name: Literal["download_image"] = "download_image"

    def __init__(self, save_dir: str = "./downloads"):
        """Create a new PlaywrightDownloadImageTool.

        Args:
            save_dir: Directory to save downloaded images to.
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def to_params(self) -> BetaToolParam:
        """Params describing the tool."""
        return BetaToolParam(
            name=self.name,
            description="Download an image from a URL and save it to a file. Use this when you find an image on a webpage that you want to save.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the image to download.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "The filename to save the image as (e.g., 'cat_photo.jpg'). Will be saved in the downloads directory.",
                    },
                },
                "required": ["url", "filename"],
            },
        )

    async def __call__(self, *, url: str, filename: str):
        """Download an image from a URL and save it to a file."""
        try:
            # Infer extension from URL if filename doesn't have one
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
                # Try to get extension from URL
                url_path = url.split("?")[0]  # Remove query params
                if "." in url_path.split("/")[-1]:
                    ext = "." + url_path.split(".")[-1].lower()
                    if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]:
                        filename += ext
                    else:
                        filename += ".png"
                else:
                    filename += ".png"

            filepath = self.save_dir / filename

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return ToolResult(error=f"Failed to download image: HTTP {response.status}")

                    content = await response.read()

                    # Save the image
                    with open(filepath, "wb") as f:
                        f.write(content)

            return ToolResult(output=f"Image downloaded and saved to: {filepath.absolute()}")
        except aiohttp.ClientError as e:
            return ToolResult(error=f"Network error downloading image: {str(e)}")
        except Exception as e:
            return ToolResult(error=f"Failed to download image: {str(e)}")


class PlaywrightElementScreenshotTool:
    """Tool to screenshot a specific element on the page using a CSS selector."""

    name: Literal["screenshot_element"] = "screenshot_element"

    def __init__(self, page: Page, save_dir: str = "./downloads"):
        """Create a new PlaywrightElementScreenshotTool.

        Args:
            page: The Async Playwright page to interact with.
            save_dir: Directory to save screenshots to.
        """
        super().__init__()
        self.page = page
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def to_params(self) -> BetaToolParam:
        """Params describing the tool."""
        return BetaToolParam(
            name=self.name,
            description="Take a screenshot of a specific element on the page using a CSS selector. Useful for capturing images, buttons, or specific UI components.",
            input_schema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element to screenshot (e.g., 'img[alt=\"Cat photo\"]', '#product-image', '.profile-picture')",
                    },
                    "filename": {
                        "type": "string",
                        "description": "The filename to save the screenshot as (e.g., 'element.png'). Will be saved in the downloads directory.",
                    },
                    "index": {
                        "type": "number",
                        "description": "If multiple elements match the selector, which one to screenshot (0-indexed). Default is 0 (first match).",
                    },
                },
                "required": ["selector", "filename"],
            },
        )

    async def __call__(self, *, selector: str, filename: str, index: int = 0):
        """Screenshot a specific element on the page."""
        try:
            # Ensure filename has an extension
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filename += ".png"

            filepath = self.save_dir / filename

            # Locate the element
            locator = self.page.locator(selector)

            # Check if element exists
            count = await locator.count()
            if count == 0:
                return ToolResult(error=f"No elements found matching selector: {selector}")

            if index >= count:
                return ToolResult(
                    error=f"Index {index} out of range. Found {count} element(s) matching selector: {selector}"
                )

            # Screenshot the specific element (with timeout to avoid hanging)
            element = locator.nth(index)
            await element.screenshot(path=str(filepath), timeout=10000)

            return ToolResult(output=f"Element screenshot saved to: {filepath.absolute()}")
        except Exception as e:
            return ToolResult(error=f"Failed to screenshot element: {str(e)}")


class PlaywrightExtractImageUrlTool:
    """Tool to extract the source URL from an image element on the page."""

    name: Literal["extract_image_url"] = "extract_image_url"

    def __init__(self, page: Page):
        """Create a new PlaywrightExtractImageUrlTool.

        Args:
            page: The Async Playwright page to interact with.
        """
        super().__init__()
        self.page = page

    def to_params(self) -> BetaToolParam:
        """Params describing the tool."""
        return BetaToolParam(
            name=self.name,
            description=(
                "Extract the source URL (src attribute) from an image element on the page. "
                "Use this to get the direct image URL which can then be used with download_image. "
                "This is especially useful for Google Image search results - instead of clicking "
                "on images (which redirects to websites), use this tool to extract the image URL "
                "and then download it directly."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": (
                            "CSS selector for the image element (e.g., 'img.rg_i', 'img[data-src]', "
                            "'img[alt=\"Cat photo\"]'). For Google Images, try 'img.rg_i' for thumbnails."
                        ),
                    },
                    "index": {
                        "type": "number",
                        "description": "If multiple elements match the selector, which one to extract from (0-indexed). Default is 0 (first match).",
                    },
                    "attribute": {
                        "type": "string",
                        "description": (
                            "Which attribute to extract. Default is 'src'. Other options: 'data-src', "
                            "'data-iurl' (Google Images full URL), 'srcset'. For Google Image search, "
                            "try 'data-src' or inspect the element attributes."
                        ),
                    },
                },
                "required": ["selector"],
            },
        )

    async def __call__(self, *, selector: str, index: int = 0, attribute: str = "src"):
        """Extract the image URL from an element."""
        try:
            # Locate the element
            locator = self.page.locator(selector)

            # Check if element exists
            count = await locator.count()
            if count == 0:
                return ToolResult(error=f"No elements found matching selector: {selector}")

            if index >= count:
                return ToolResult(
                    error=f"Index {index} out of range. Found {count} element(s) matching selector: {selector}"
                )

            # Get the element
            element = locator.nth(index)

            # Extract the attribute
            url = await element.get_attribute(attribute)

            if not url:
                # Try to get all available attributes to help debugging
                all_attrs = await element.evaluate(
                    "el => Array.from(el.attributes).map(a => `${a.name}=${a.value.substring(0, 100)}`).join(', ')"
                )
                return ToolResult(
                    error=f"Attribute '{attribute}' not found or empty on element. "
                    f"Available attributes: {all_attrs}"
                )

            return ToolResult(output=f"Extracted URL: {url}")
        except Exception as e:
            return ToolResult(error=f"Failed to extract image URL: {str(e)}")


class PlaywrightListImagesOnPageTool:
    """Tool to list all images on the current page with their URLs and attributes."""

    name: Literal["list_images"] = "list_images"

    def __init__(self, page: Page):
        """Create a new PlaywrightListImagesOnPageTool.

        Args:
            page: The Async Playwright page to interact with.
        """
        super().__init__()
        self.page = page

    def to_params(self) -> BetaToolParam:
        """Params describing the tool."""
        return BetaToolParam(
            name=self.name,
            description=(
                "List all images currently visible on the page, showing their URLs and key attributes. "
                "Useful for discovering what images are available to download and what selectors to use. "
                "Returns up to 20 images with their src, alt text, and CSS selector hints."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "filter_selector": {
                        "type": "string",
                        "description": "Optional CSS selector to filter which images to list (e.g., '.main-content img'). If not provided, lists all images.",
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Maximum number of images to return. Default is 20.",
                    },
                },
                "required": [],
            },
        )

    async def __call__(self, *, filter_selector: str = "img", max_results: int = 20):
        """List images on the page."""
        try:
            # Use JavaScript to gather image information
            images_info = await self.page.evaluate(
                """
                (args) => {
                    const selector = args.selector;
                    const maxResults = args.maxResults;
                    const images = document.querySelectorAll(selector);
                    const results = [];
                    
                    for (let i = 0; i < Math.min(images.length, maxResults); i++) {
                        const img = images[i];
                        const rect = img.getBoundingClientRect();
                        
                        // Skip tiny or hidden images
                        if (rect.width < 20 || rect.height < 20) continue;
                        
                        results.push({
                            index: results.length,
                            src: img.src || img.dataset.src || '',
                            dataSrc: img.dataset.src || '',
                            alt: img.alt || '',
                            width: Math.round(rect.width),
                            height: Math.round(rect.height),
                            classes: img.className,
                            id: img.id || ''
                        });
                    }
                    return results;
                }
                """,
                {"selector": filter_selector, "maxResults": max_results},
            )

            if not images_info:
                return ToolResult(output="No images found matching the filter.")

            # Format the output
            output_lines = [f"Found {len(images_info)} images:\n"]
            for img in images_info:
                lines = [f"[{img['index']}]"]
                if img["src"]:
                    # Truncate long URLs
                    src = img["src"][:150] + "..." if len(img["src"]) > 150 else img["src"]
                    lines.append(f"  src: {src}")
                if img["dataSrc"] and img["dataSrc"] != img["src"]:
                    ds = (
                        img["dataSrc"][:150] + "..."
                        if len(img["dataSrc"]) > 150
                        else img["dataSrc"]
                    )
                    lines.append(f"  data-src: {ds}")
                if img["alt"]:
                    lines.append(f"  alt: {img['alt'][:80]}")
                lines.append(f"  size: {img['width']}x{img['height']}")

                # Suggest selector
                if img["id"]:
                    lines.append(f"  selector: #{img['id']}")
                elif img["classes"]:
                    first_class = img["classes"].split()[0]
                    lines.append(f"  selector: img.{first_class}")
                output_lines.append("\n".join(lines))

            return ToolResult(output="\n\n".join(output_lines))
        except Exception as e:
            return ToolResult(error=f"Failed to list images: {str(e)}")


class BasePlaywrightComputerTool:
    """A tool that allows the agent to interact with Async Playwright Page."""

    name: Literal["computer"] = "computer"

    @property
    def width(self) -> int:
        """The width of the Playwright page in pixels."""
        return self.page.viewport_size["width"]

    @property
    def height(self) -> int:
        """The height of the Playwright page in pixels."""
        return self.page.viewport_size["height"]

    @property
    def options(self) -> ComputerToolOptions:
        """The options of the tool."""
        return {
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": 1,  # hardcoded
        }

    def to_params(self):
        """Params describing the tool. Used by Claude to understand this is a computer use tool."""
        raise NotImplementedError("to_params must be implemented in the subclass")

    def __init__(
        self,
        page: Page,
        use_cursor: bool = True,
        screenshot_wait_until: Literal["load", "domcontentloaded", "networkidle"] | None = None,
    ):
        """Initializes the PlaywrightComputerTool.

        Args:
            page: The Async Playwright page to interact with.
            use_cursor: Whether to display the cursor in the screenshots or not.
            screenshot_wait_until: Optional, wait until the page is in a specific state before taking a screenshot. Default does not wait
        """
        super().__init__()
        self.page = page
        self.use_cursor = use_cursor
        self.mouse_position: tuple[int, int] = (0, 0)
        self.screenshot_wait_until = screenshot_wait_until

    async def __call__(
        self,
        *,
        action: Action_20241022,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        """Run an action. text and coordinate are potential additional parameters."""
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = coordinate

            if action == "mouse_move":
                await self.page.mouse.move(x, y)
                self.mouse_position = (x, y)
                return ToolResult(output=None, error=None, base64_image=None)
            elif action == "left_click_drag":
                raise NotImplementedError("left_click_drag is not implemented yet")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # hande shifts
                await self.press_key(text)
                return ToolResult()
            elif action == "type":
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    await self.page.keyboard.type(chunk)
                return await self.screenshot()

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                return ToolResult(output=f"X={self.mouse_position[0]},Y={self.mouse_position[1]}")
            else:
                click_arg = {
                    "left_click": {"button": "left", "click_count": 1},
                    "right_click": {"button": "right", "click_count": 1},
                    "middle_click": {"button": "middle", "click_count": 1},
                    "double_click": {"button": "left", "click_count": 2, "delay": 100},
                }[action]
                await self.page.mouse.click(
                    self.mouse_position[0], self.mouse_position[1], **click_arg
                )
                return ToolResult()

        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self) -> ToolResult:
        """Take a screenshot of the current screen and return the base64 encoded image."""
        if self.screenshot_wait_until is not None:
            await self.page.wait_for_load_state(self.screenshot_wait_until)
        # Use domcontentloaded with timeout to avoid hanging on heavy sites with ads/trackers
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass

        # Take screenshot with timeout to avoid hanging on slow pages
        try:
            screenshot = await self.page.screenshot(timeout=10000)
        except Exception as e:
            # Return error so Claude can navigate away or retry
            return ToolResult(
                error=f"Screenshot failed (page may still be loading): {str(e)}. "
                "Consider navigating to a different page or waiting a moment."
            )

        image = Image.open(io.BytesIO(screenshot))
        img_small = image.resize((self.width, self.height), Image.LANCZOS)
        if self.use_cursor:
            cursor = load_cursor_image()
            img_small.paste(cursor, self.mouse_position, cursor)
        buffered = io.BytesIO()
        img_small.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        return ToolResult(base64_image=base64_image)

    async def press_key(self, key: str):
        """Press a key on the keyboard. Handle + shifts. Eg: Ctrl+Shift+T."""
        shifts = []
        if "+" in key:
            shifts += key.split("+")[:-1]
            key = key.split("+")[-1]
        for shift in shifts:
            await self.page.keyboard.down(shift)
        await self.page.keyboard.press(to_playwright_key(key))
        for shift in shifts:
            await self.page.keyboard.up(shift)


class PlaywrightComputerTool20241022(BasePlaywrightComputerTool):
    """Tool to interact with the computer using Playwright (Beta 22/10/2024)."""

    api_type: Literal["computer_20241022"] = "computer_20241022"

    def to_params(self) -> BetaToolComputerUse20241022Param:
        """Params describing the tool. Used by Claude to understand this is a computer use tool."""
        return {"name": self.name, "type": self.api_type, **self.options}


class PlaywrightComputerTool20250124(BasePlaywrightComputerTool):
    """Tool to interact with the computer using Playwright (Beta 24/01/2025)."""

    api_type: Literal["computer_20250124"] = "computer_20250124"

    def to_params(self) -> BetaToolComputerUse20250124Param:
        """Params describing the tool. Used by Claude to understand this is a computer use tool."""
        return {"name": self.name, "type": self.api_type, **self.options}

    async def __call__(
        self,
        *,
        action: Action_20250124,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        scroll_direction: ScrollDirection | None = None,
        scroll_amount: int | None = None,
        duration: int | float | None = None,
        key: str | None = None,
        **kwargs,
    ):
        """Run an action. text, coordinate, scroll_directions, scroll_amount, duration, key are potential additional parameters."""
        if action in ("left_mouse_down", "left_mouse_up"):
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action=}.")
            (
                await self.page.mouse.down()
                if action == "left_mouse_down"
                else await self.page.mouse.up()
            )
            return ToolResult()
        if action == "scroll":
            if scroll_direction is None or scroll_direction not in get_args(ScrollDirection):
                raise ToolError(f"{scroll_direction=} must be 'up', 'down', 'left', or 'right'")
            if not isinstance(scroll_amount, int) or scroll_amount < 0:
                raise ToolError(f"{scroll_amount=} must be a non-negative int")
            if coordinate is not None:
                x, y = coordinate
                await self.page.mouse.move(x, y)
                self.mouse_position = (x, y)
            scroll_amount *= SCROLL_MULTIPLIER_FACTOR
            scroll_params = {
                "up": {"delta_y": -scroll_amount, "delta_x": 0},
                "down": {"delta_y": scroll_amount, "delta_x": 0},
                "left": {"delta_y": 0, "delta_x": scroll_amount},
                "right": {"delta_y": 0, "delta_x": -scroll_amount},
            }[scroll_direction]

            await self.page.mouse.wheel(**scroll_params)
            return ToolResult()

        if action in ("hold_key", "wait"):
            if duration is None or not isinstance(duration, (int, float)):
                raise ToolError(f"{duration=} must be a number")
            if duration < 0:
                raise ToolError(f"{duration=} must be non-negative")
            if duration > 100:
                raise ToolError(f"{duration=} is too long.")

            if action == "hold_key":
                if text is None:
                    raise ToolError(f"text is required for {action}")
                await self.page.keyboard.press(to_playwright_key(text), delay=duration)
                return ToolResult()

            if action == "wait":
                await sleep(duration)
                return await self.screenshot()

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "triple_click",
            "middle_click",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            mouse_move_part = ""
            if coordinate is not None:
                x, y = coordinate
                await self.page.mouse.move(x, y)
                self.mouse_position = (x, y)

            click_arg = {
                "left_click": {"button": "left", "click_count": 1},
                "right_click": {"button": "right", "click_count": 1},
                "middle_click": {"button": "middle", "click_count": 1},
                "double_click": {"button": "left", "click_count": 2, "delay": 10},
                "triple_click": {"button": "left", "click_count": 3, "delay": 10},
            }[action]
            if key:
                self.page.keyboard.down(to_playwright_key(key))
            await self.page.mouse.click(self.mouse_position[0], self.mouse_position[1], **click_arg)
            if key:
                self.page.keyboard.up(to_playwright_key(key))

            return ToolResult()
        action = cast(Action_20241022, action)
        return await super().__call__(
            action=action, text=text, coordinate=coordinate, key=key, **kwargs
        )


class PlaywrightComputerTool20251124(PlaywrightComputerTool20250124):
    """Tool to interact with the computer using Playwright (Beta 24/11/2025 - Claude Opus 4.5)."""

    api_type: Literal["computer_20251124"] = "computer_20251124"

    def to_params(self) -> dict:
        """Params describing the tool. Used by Claude to understand this is a computer use tool."""
        return {"name": self.name, "type": self.api_type, **self.options}


def to_playwright_key(key: str) -> str:
    """Convert a key to the Playwright key format."""
    valid_keys = (
        ["F{i}" for i in range(1, 13)]
        + ["Digit{i}" for i in range(10)]
        + ["Key{i}" for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        + [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        + [i.lower() for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        + [
            "Backquote",
            "Minus",
            "Equal",
            "Backslash",
            "Backspace",
            "Tab",
            "Delete",
            "Escape",
            "ArrowDown",
            "End",
            "Enter",
            "Home",
            "Insert",
            "PageDown",
            "PageUp",
            "ArrowRight",
            "ArrowUp",
        ]
    )
    if key in valid_keys:
        return key
    if key == "Return":
        return "Enter"
    if key == "Page_Down":
        return "PageDown"
    if key == "Page_Up":
        return "PageUp"
    if key == "Left":
        return "ArrowLeft"
    if key == "Right":
        return "ArrowRight"
    if key == "Up":
        return "ArrowUp"
    if key == "Down":
        return "ArrowDown"
    if key == "BackSpace":
        return "Backspace"
    if key == "alt":
        return "Alt"
    print(f"Key {key} is not properly mapped into playwright")
    return key


def load_cursor_image():
    """Access the cursor.png file in the assets directory."""
    with importlib.resources.open_binary(
        "agentlab.trajectory_agent.playwright_computer_use.assets", "cursor.png"
    ) as img_file:
        image = Image.open(img_file)
        image.load()  # Ensure the image is fully loaded into memory
    return image


def _make_api_tool_result(result: ToolResult, tool_use_id: str) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    if result.error:
        return BetaToolResultBlockParam(
            tool_use_id=tool_use_id,
            is_error=True,
            content=result.error,
            type="tool_result",
        )
    else:
        tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] = []
        if result.output:
            tool_result_content.append(
                BetaTextBlockParam(
                    type="text",
                    text=result.output,
                )
            )
        if result.base64_image:
            tool_result_content.append(
                BetaImageBlockParam(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                )
            )
        return BetaToolResultBlockParam(
            tool_use_id=tool_use_id,
            is_error=False,
            content=tool_result_content,
            type="tool_result",
        )
