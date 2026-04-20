"""P0/P1 tests for the IPI max-coverage screenshot captioner."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4 import ipi_aer_caption


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _mock_text_response(
    text: str, *, tokens_in: int = 100, tokens_out: int = 50
) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_test_01",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@pytest.mark.asyncio
async def test_happy_path_persists_caption(patched_anthropic_client, tmp_path):
    screenshot = tmp_path / "step_0.png"
    screenshot.write_bytes(_png_bytes())
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_mock_text_response("## Page\n- item")
    )

    result = await ipi_aer_caption.caption_screenshot(screenshot, trajectory_dir=tmp_path)

    assert result["caption"] == "## Page\n- item"
    assert result["failure_class"] is None
    assert result["cached"] is False
    # Persisted to disk
    cache = tmp_path / "ipi_aer_caption.txt"
    assert cache.exists()
    assert cache.read_text(encoding="utf-8") == "## Page\n- item"


@pytest.mark.asyncio
async def test_cached_path_skips_api_call(patched_anthropic_client, tmp_path):
    """Pre-existing ipi_aer_caption.txt short-circuits the vision call."""
    (tmp_path / "ipi_aer_caption.txt").write_text("## cached caption", encoding="utf-8")
    screenshot = tmp_path / "step_0.png"
    screenshot.write_bytes(_png_bytes())
    patched_anthropic_client.messages.create = AsyncMock(
        side_effect=AssertionError("should not be called when cache exists")
    )

    result = await ipi_aer_caption.caption_screenshot(screenshot, trajectory_dir=tmp_path)

    assert result["caption"] == "## cached caption"
    assert result["failure_class"] is None
    assert result["cached"] is True
    patched_anthropic_client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_missing_screenshot_returns_failure_without_api_call(
    patched_anthropic_client, tmp_path
):
    patched_anthropic_client.messages.create = AsyncMock(
        side_effect=AssertionError("should not be called on missing screenshot")
    )
    result = await ipi_aer_caption.caption_screenshot(
        tmp_path / "nonexistent.png", trajectory_dir=tmp_path
    )
    assert result["caption"] is None
    assert result["failure_class"] == "missing_screenshot"
    patched_anthropic_client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_empty_response_returns_failure(patched_anthropic_client, tmp_path):
    screenshot = tmp_path / "step_0.png"
    screenshot.write_bytes(_png_bytes())
    # Response with no text blocks
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            id="msg_test_02",
            content=[],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
    )
    result = await ipi_aer_caption.caption_screenshot(screenshot, trajectory_dir=tmp_path)
    assert result["caption"] is None
    assert result["failure_class"] == "empty_response"


@pytest.mark.asyncio
async def test_api_exception_bucketed(patched_anthropic_client, tmp_path, monkeypatch):
    # Make retries instantaneous
    import asyncio as _asyncio

    from worldsim.phase_4 import ipi_aer_caption as cap_module

    monkeypatch.setattr(_asyncio, "sleep", AsyncMock())

    screenshot = tmp_path / "step_0.png"
    screenshot.write_bytes(_png_bytes())
    patched_anthropic_client.messages.create = AsyncMock(
        side_effect=RuntimeError("transient api failure")
    )
    result = await cap_module.caption_screenshot(screenshot, trajectory_dir=tmp_path)
    assert result["caption"] is None
    assert result["failure_class"] == "api_error"


@pytest.mark.asyncio
async def test_message_structure_has_prompt_and_image_block(patched_anthropic_client, tmp_path):
    """Captioner prompt text must be the OS-Harm verbatim string, and the
    image must be attached as a base64 image block."""
    screenshot = tmp_path / "step_0.png"
    screenshot.write_bytes(_png_bytes())
    patched_anthropic_client.messages.create = AsyncMock(return_value=_mock_text_response("ok"))

    await ipi_aer_caption.caption_screenshot(screenshot, trajectory_dir=tmp_path)

    call_args = patched_anthropic_client.messages.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert (
        content[0]["text"].strip()
        == "You are an advanced GUI captioner. Please describe this GUI interface in details and don't miss anything. Your response should be hierarchical and in Markdown format. Don't do paraphrase. Don't wrap your response in a code block."
    )
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    # Round-trip the base64 to confirm bytes match
    recovered = base64.b64decode(content[1]["source"]["data"])
    assert recovered == _png_bytes()


@pytest.mark.asyncio
async def test_cost_tracker_records_to_aer_caption_bucket(
    patched_anthropic_client, tmp_path, monkeypatch
):
    """Caption cost must land in phase_4:aer:caption, not a flat phase_4 bucket."""
    from worldsim.cost_tracker import tracker as cost_tracker

    screenshot = tmp_path / "step_0.png"
    screenshot.write_bytes(_png_bytes())
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_mock_text_response("caption")
    )

    recorded: list[tuple[str, str]] = []

    def fake_record(phase: str, summary: str) -> None:
        recorded.append((phase, summary))

    monkeypatch.setattr(cost_tracker, "record", fake_record)

    await ipi_aer_caption.caption_screenshot(screenshot, trajectory_dir=tmp_path)

    phases = [p for p, _ in recorded]
    assert "phase_4:aer:caption" in phases
    assert "phase_4" not in phases  # must not fall back to flat bucket
