import os
import requests
from dotenv import load_dotenv
import json
import base64
from PIL import Image
from io import BytesIO
import io
from urllib.parse import urlparse
import time
from playwright._impl._errors import TimeoutError


load_dotenv(override=True)

BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]

MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds


def retry_dom_action(action_fn, *args, **kwargs):
    """
    Try a DOM action up to MAX_RETRIES times.
    Never raises; returns {"ok": True, "result": ...} on success
    or {"ok": False, "error": "..."} on failure.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = action_fn(*args, **kwargs)
            return {"ok": True, "result": res}
        except (PWTimeoutError, Exception) as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                print(
                    f"[WARN] {action_fn.__name__} failed ({attempt}/{MAX_RETRIES}): {e}. Retrying..."
                )
                time.sleep(RETRY_DELAY)
            else:
                print(
                    f"[ERROR] {action_fn.__name__} failed after {MAX_RETRIES} attempts: {e}"
                )
                return {"ok": False, "error": last_err}


def pp(obj):
    print(json.dumps(obj, indent=4))


def show_image(base_64_image):
    image_data = base64.b64decode(base_64_image)
    image = Image.open(BytesIO(image_data))
    image.show()


def calculate_image_dimensions(base_64_image):
    image_data = base64.b64decode(base_64_image)
    image = Image.open(io.BytesIO(image_data))
    return image.size


def sanitize_message(msg: dict) -> dict:
    """Return a copy of the message with image_url omitted for computer_call_output messages."""
    if msg.get("type") == "computer_call_output":
        output = msg.get("output", {})
        if isinstance(output, dict):
            sanitized = msg.copy()
            sanitized["output"] = {**output, "image_url": "[omitted]"}
            return sanitized
    return msg


MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds


def retry_dom_action(action_fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return action_fn(*args, **kwargs)
        except TimeoutError as e:
            if attempt < MAX_RETRIES:
                print(
                    f"[WARN] Timeout during {action_fn.__name__}, retrying ({attempt}/{MAX_RETRIES})..."
                )
                time.sleep(RETRY_DELAY)
                continue
            else:
                print(
                    f"[ERROR] {action_fn.__name__} failed after {MAX_RETRIES} attempts: {e}"
                )
                raise
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(
                    f"[WARN] Exception in {action_fn.__name__}: {e}, retrying ({attempt}/{MAX_RETRIES})..."
                )
                time.sleep(RETRY_DELAY)
                continue
            raise


def create_response(**kwargs):
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    openai_org = os.getenv("OPENAI_ORG")
    if openai_org:
        headers["OpenAI-Organization"] = openai_org

    # Inject system message if provided
    system_prompt = kwargs.pop("instructions", None)
    input = kwargs.get("input", [])

    if system_prompt:
        # Prepend system message without overwriting user messages
        input = [{"role": "system", "content": system_prompt}] + input

    kwargs["input"] = input

    response = requests.post(url, headers=headers, json=kwargs)

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None

    data = response.json()
    print("✅ RESPONSE:", data)
    return data


def check_blocklisted_url(url: str) -> None:
    """Raise ValueError if the given URL (including subdomains) is in the blocklist."""
    hostname = urlparse(url).hostname or ""
    if any(
        hostname == blocked or hostname.endswith(f".{blocked}")
        for blocked in BLOCKED_DOMAINS
    ):
        raise ValueError(f"Blocked URL: {url}")
