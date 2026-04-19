from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from worldsim.phase_4.p_eval_api import run_p_eval_api


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), (0, 128, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def reference_png(tmp_path: Path) -> Path:
    shots = tmp_path / "screenshots"
    shots.mkdir(parents=True)
    path = shots / "step_0.png"
    path.write_bytes(_make_png_bytes())
    return path


@pytest.fixture
def trajectory_dir(tmp_path: Path) -> Path:
    return tmp_path


def _usage(in_tok: int = 1000, out_tok: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _tool_use_response(tool_input: dict, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="score_p_eval", input=tool_input)],
        stop_reason=stop_reason,
        usage=_usage(),
        id="msg_peval",
        model="claude-sonnet-4-6",
    )


def _valid_probes(scores: list[float] | None = None) -> list[dict[str, Any]]:
    scores = scores if scores is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
    names = [
        "content_realism",
        "ui_realism",
        "data_coherence",
        "artifact_presence",
        "overall_plausibility",
    ]
    return [
        {"name": n, "score": s, "justification": f"{n} looks fine"}
        for n, s in zip(names, scores, strict=True)
    ]


@pytest.mark.asyncio
async def test_happy_path_host_computes_mean(
    patched_anthropic_client, reference_png, trajectory_dir
):
    probes = _valid_probes([0.1, 0.2, 0.3, 0.4, 0.5])
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)

    assert result["failure_class"] is None
    assert result["p_eval_score"] == pytest.approx(sum([0.1, 0.2, 0.3, 0.4, 0.5]) / 5)
    assert result["p_eval_breakdown"] == probes

    raw = json.loads((trajectory_dir / "p_eval_raw_response.json").read_text())
    assert raw["tool_use"]["probes"][0]["name"] == "content_realism"
    assert raw["site_context"] == "shopping_admin"


@pytest.mark.asyncio
async def test_image_block_present_and_base64_valid(
    patched_anthropic_client, reference_png, trajectory_dir
):
    patched_anthropic_client.messages.create.return_value = _tool_use_response(
        {"probes": _valid_probes()}
    )

    await run_p_eval_api(reference_png, "reddit", trajectory_dir)

    call = patched_anthropic_client.messages.create.call_args
    assert call is not None
    messages = call.kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    assert len(image_blocks) == 1
    source = image_blocks[0]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    # round-trip decodes to the same bytes we wrote to disk
    assert base64.standard_b64decode(source["data"]) == reference_png.read_bytes()

    # tool_choice is forced; exactly one probe tool registered
    assert call.kwargs["tool_choice"] == {"type": "tool", "name": "score_p_eval"}
    tools = call.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "score_p_eval"


@pytest.mark.asyncio
async def test_schema_violation_four_probes(
    patched_anthropic_client, reference_png, trajectory_dir
):
    probes = _valid_probes()[:4]  # one short
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "gitlab", trajectory_dir)
    assert result["failure_class"] == "schema_violation"
    assert result["p_eval_score"] is None
    assert result["p_eval_breakdown"] is None


@pytest.mark.asyncio
async def test_schema_violation_duplicate_probe_name(
    patched_anthropic_client, reference_png, trajectory_dir
):
    probes = _valid_probes()
    probes[4]["name"] = probes[0]["name"]  # duplicate content_realism
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "map", trajectory_dir)
    assert result["failure_class"] == "schema_violation"


@pytest.mark.asyncio
async def test_schema_violation_wrong_probe_name(
    patched_anthropic_client, reference_png, trajectory_dir
):
    probes = _valid_probes()
    probes[0]["name"] = "task_realism"  # old taxonomy
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "shopping", trajectory_dir)
    assert result["failure_class"] == "schema_violation"


@pytest.mark.asyncio
async def test_non_numeric_score_string(patched_anthropic_client, reference_png, trajectory_dir):
    probes = _valid_probes()
    probes[2]["score"] = "0.5"  # string, not number
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "non_numeric_score"


@pytest.mark.asyncio
async def test_non_numeric_score_list(patched_anthropic_client, reference_png, trajectory_dir):
    probes = _valid_probes()
    probes[1]["score"] = [0.5]
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "non_numeric_score"


@pytest.mark.asyncio
async def test_bool_score_rejected(patched_anthropic_client, reference_png, trajectory_dir):
    """Bool is a subclass of int in Python; the host must reject it explicitly.

    Regression guard for the documented bug in the prior sandbox probe where
    ``{"score": true}`` was silently coerced to ``1.0``.
    """
    probes = _valid_probes()
    probes[0]["score"] = True
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "non_numeric_score"


@pytest.mark.asyncio
async def test_score_out_of_range(patched_anthropic_client, reference_png, trajectory_dir):
    probes = _valid_probes()
    probes[3]["score"] = 1.5
    patched_anthropic_client.messages.create.return_value = _tool_use_response({"probes": probes})

    result = await run_p_eval_api(reference_png, "reddit", trajectory_dir)
    assert result["failure_class"] == "schema_violation"


@pytest.mark.asyncio
async def test_no_tool_use_block(patched_anthropic_client, reference_png, trajectory_dir):
    patched_anthropic_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="I won't score this.")],
        stop_reason="end_turn",
        usage=_usage(),
        id="msg_refuse",
        model="claude-sonnet-4-6",
    )

    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "no_tool_use"
    assert result["p_eval_score"] is None


@pytest.mark.asyncio
async def test_api_error_generic(patched_anthropic_client, reference_png, trajectory_dir):
    patched_anthropic_client.messages.create.side_effect = RuntimeError("synthetic boom")
    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "api_error"
    assert result["p_eval_score"] is None
    # Raw response persisted with error payload.
    raw = json.loads((trajectory_dir / "p_eval_raw_response.json").read_text())
    assert raw["kind"] == "api_error"
    assert raw["site_context"] == "shopping_admin"


def _api_status_error(status_code: int, message: str = "synthetic") -> Exception:
    import httpx
    from anthropic import APIStatusError

    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(
        message=message,
        response=response,
        body={"error": {"code": status_code}},
    )


@pytest.mark.asyncio
async def test_insufficient_credits_bucketed(
    patched_anthropic_client, reference_png, trajectory_dir
):
    patched_anthropic_client.messages.create.side_effect = _api_status_error(402)
    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "insufficient_credits"


@pytest.mark.asyncio
async def test_auth_invalid_bucketed(patched_anthropic_client, reference_png, trajectory_dir):
    patched_anthropic_client.messages.create.side_effect = _api_status_error(401)
    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "auth_invalid"


@pytest.mark.asyncio
async def test_quota_exceeded_bucketed(patched_anthropic_client, reference_png, trajectory_dir):
    patched_anthropic_client.messages.create.side_effect = _api_status_error(403)
    result = await run_p_eval_api(reference_png, "shopping_admin", trajectory_dir)
    assert result["failure_class"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_missing_reference_screenshot_returns_api_error(
    patched_anthropic_client, tmp_path: Path
):
    """Defensive: reference_screenshot_path must resolve to a readable file.

    The caller (phase_4_adversarial) only invokes run_p_eval_api when
    encounter.max_coverage > 0, which implies the file exists. This test
    documents the fallback behavior if the invariant is ever violated.
    """
    result = await run_p_eval_api(tmp_path / "nope.png", "shopping_admin", tmp_path)
    assert result["failure_class"] == "api_error"
    assert result["p_eval_score"] is None
    # No API call was made.
    patched_anthropic_client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_site_context_in_messages(patched_anthropic_client, reference_png, trajectory_dir):
    patched_anthropic_client.messages.create.return_value = _tool_use_response(
        {"probes": _valid_probes()}
    )
    await run_p_eval_api(reference_png, "gitlab", trajectory_dir)
    call = patched_anthropic_client.messages.create.call_args
    content = call.kwargs["messages"][0]["content"]
    text_blocks = [b for b in content if b["type"] == "text"]
    # First text block is the cached prompt, second is the site_context header.
    assert any("gitlab" in b["text"] for b in text_blocks)
    assert text_blocks[0].get("cache_control") == {"type": "ephemeral"}
