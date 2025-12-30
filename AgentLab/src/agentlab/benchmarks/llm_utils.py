"""
Shared LLM utils.
"""

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


def get_openrouter_client() -> OpenAI:
    """
    Get an OpenRouter client instance.

    Returns:
        OpenAI client configured for OpenRouter

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment. " "Please set it in your .env file."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def encode_image_base64(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Encode an image file as a base64 data URL for OpenRouter multimodal API.

    Args:
        file_path: Path to the image file

    Returns:
        Dict with image_url format for OpenRouter, or None if failed

    Per OpenRouter docs: https://openrouter.ai/docs/guides/overview/multimodal/images
    Format: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    if not file_path.exists():
        logger.warning(f"Image file not found: {file_path}")
        return None

    if file_path.is_dir():
        logger.warning(f"Expected file but got directory: {file_path}")
        return None

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        # Default to PNG for unknown types
        mime_type = "image/png"

    try:
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        data_url = f"data:{mime_type};base64,{image_data}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    except Exception as e:
        logger.warning(f"Failed to encode image {file_path}: {e}")
        return None


def resolve_screenshot_paths(screenshots: List[str] | str) -> List[Path]:
    """
    Resolve screenshot paths to a list of file paths.

    Args:
        screenshots: List of screenshot paths or folder path

    Returns:
        List of resolved Path objects
    """

    def resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else Path("assets") / p

    paths: List[Path] = []

    if isinstance(screenshots, str):
        dir_path = resolve(screenshots)
        if dir_path.is_dir():
            paths = [f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    else:
        for s in screenshots:
            p = resolve(s)
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                paths.extend(f for f in p.iterdir() if f.is_file() and not f.name.startswith("."))

    return paths


class OpenRouterLLMClient:
    """
    OpenRouter-based LLM client for all model providers.

    This client provides a unified interface for calling LLMs via OpenRouter,
    supporting both text-only and multimodal (with images) requests.

    Usage:
        client = OpenRouterLLMClient("anthropic/claude-opus-4.5")
        response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
        response = client.call_with_images("Describe this image", screenshots=["path/to/image.png"])
    """

    def __init__(self, model: str):
        """
        Initialize LLM client using OpenRouter.

        Args:
            model: Model name in OpenRouter format
        """
        self.model = model
        self.client = get_openrouter_client()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = None,
    ) -> Dict[str, Any]:
        """
        Call LLM with a list of messages (chat completion).

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (None uses model default)

        Returns:
            Full response object from OpenRouter

        Raises:
            Exception: If API call fails
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        return self.client.chat.completions.create(**kwargs)

    def chat_simple(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = None,
    ) -> str:
        """
        Call LLM and return just the response text.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (None uses model default)

        Returns:
            Response text content
        """
        response = self.chat(messages, max_tokens, temperature)
        return response.choices[0].message.content or ""

    def call_with_images(
        self,
        prompt: str,
        screenshots: Optional[List[str] | str] = None,
        max_tokens: int = 16000,
    ) -> str:
        """
        Call LLM with prompt and optional images via OpenRouter.

        Args:
            prompt: Text prompt
            screenshots: Optional screenshot paths (files or folders)
            max_tokens: Maximum tokens in response

        Returns:
            Response text

        Raises:
            ValueError: If prompt is empty or whitespace-only
        """
        # Validate prompt is non-empty (some providers like Google reject empty text blocks)
        prompt = prompt.strip() if prompt else ""
        if not prompt:
            raise ValueError("Prompt cannot be empty or whitespace-only")

        # Build content array
        content = [{"type": "text", "text": prompt}]

        # Add images as base64 data URLs
        if screenshots:
            paths = resolve_screenshot_paths(screenshots)
            for path in paths:
                image_content = encode_image_base64(path)
                if image_content:
                    content.append(image_content)
                    logger.debug(f"Added image: {path}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            # #region agent log
            import json
            import traceback

            error_type = type(e).__name__
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            # Try to extract raw response from exception chain
            raw_response = None
            response_content = None
            response_status = None

            # Check exception and its causes for response object
            exc = e
            for _ in range(5):  # Check up to 5 levels deep
                if hasattr(exc, "response"):
                    raw_response = exc.response
                    break
                if hasattr(exc, "http_response"):
                    raw_response = exc.http_response
                    break
                if hasattr(exc, "__cause__") and exc.__cause__:
                    exc = exc.__cause__
                else:
                    break

            # Try to extract content from response
            if raw_response:
                if hasattr(raw_response, "status_code"):
                    response_status = raw_response.status_code
                if hasattr(raw_response, "content"):
                    try:
                        response_content = raw_response.content.decode("utf-8", errors="replace")
                    except:
                        try:
                            response_content = str(raw_response.content)
                        except:
                            pass
                elif hasattr(raw_response, "text"):
                    response_content = raw_response.text
                elif hasattr(raw_response, "read"):
                    try:
                        response_content = raw_response.read().decode("utf-8", errors="replace")
                    except:
                        pass

            debug_data = {
                "hypothesisId": "API_RESPONSE_ERROR",
                "location": "llm_utils.py:223",
                "message": "OpenRouter API JSON parse error",
                "data": {
                    "error_type": error_type,
                    "error_message": error_msg,
                    "model": self.model,
                    "response_status": response_status,
                    "has_raw_response": raw_response is not None,
                    "response_content_preview": (
                        response_content[:2000] if response_content else None
                    ),
                    "response_content_length": len(response_content) if response_content else None,
                    "error_traceback": (
                        error_traceback[-1000:] if len(error_traceback) > 1000 else error_traceback
                    ),
                },
                "timestamp": __import__("time").time(),
            }
            open("/Users/jasminexli/grayswan/site-snapshot-test/.cursor/debug.log", "a").write(
                json.dumps(debug_data) + "\n"
            )
            # #endregion
            logger.error(f"OpenRouter API call failed: {e}")
            if response_content:
                logger.error(
                    f"Response status: {response_status}, Content preview (first 1000 chars):\n{response_content[:1000]}"
                )
            raise


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from text."""
    text = text.strip()

    # Remove opening code fence
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    # Remove closing code fence
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()
