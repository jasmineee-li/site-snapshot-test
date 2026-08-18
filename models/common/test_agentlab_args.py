"""The browser track must reach the port the serve script binds.

`models/common/registry.py` gives each local model its own port: 8001, 8002 and
8003. AgentLab's `SelfHostedModelArgs(backend="vllm")` sends every request to
http://0.0.0.0:8000/v1 instead, because `VLLMChatModel` hardcodes that value
and `make_model()` drops `model_url`. These tests pin the replacement.
"""

import pytest

from models.common.registry import LOCAL_MODELS
from models.common.vllm_client import make_agentlab_chat_model_args

pytest.importorskip("agentlab", reason="AgentLab is needed for the model args")


@pytest.fixture(autouse=True)
def _no_url_override(monkeypatch):
    """Drop any LOCAL_OPENAI_BASE_URL_* override so the default port applies."""
    for key in list(__import__("os").environ):
        if key.startswith("LOCAL_OPENAI_BASE_URL_"):
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("short_id", sorted(LOCAL_MODELS))
def test_model_args_carry_the_spec_url(short_id):
    spec = LOCAL_MODELS[short_id]
    args = make_agentlab_chat_model_args(spec)
    assert args.model_url == spec.resolve_url()


@pytest.mark.parametrize("short_id", sorted(LOCAL_MODELS))
def test_built_client_targets_the_spec_url(short_id):
    # The regression: the client, not just the args, must carry the port.
    spec = LOCAL_MODELS[short_id]
    model = make_agentlab_chat_model_args(spec).make_model()
    assert str(model.client.base_url).rstrip("/") == spec.resolve_url().rstrip("/")


def test_built_client_does_not_use_agentlabs_hardcoded_port():
    spec = LOCAL_MODELS["gui-owl-32b-think"]
    model = make_agentlab_chat_model_args(spec).make_model()
    assert "8000" not in str(model.client.base_url)
    assert "8003" in str(model.client.base_url)


def test_missing_url_raises_rather_than_defaulting():
    from models.common.agentlab_args import LocalVLLMModelArgs

    args = LocalVLLMModelArgs(model_name="stub", model_url=None, backend="vllm")
    with pytest.raises(ValueError, match="model_url"):
        args.make_model()
