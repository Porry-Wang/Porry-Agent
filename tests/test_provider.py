import pytest

from porry_agent.config import load_settings
from porry_agent.messages import Message
from porry_agent.openai_provider import OpenAIProvider, ProviderError
from porry_agent.provider import MockProvider, build_provider
from porry_agent.tools import build_default_tools


def test_mock_echoes_last_user():
    msgs = [
        Message(role="user", content="first"),
        Message(role="assistant", content="hi"),
        Message(role="user", content="second"),
    ]
    assert MockProvider().complete(msgs).content == "mock: second"


def test_mock_without_user_message():
    msgs = [Message(role="system", content="s")]
    assert MockProvider().complete(msgs).content == "mock: "


def test_message_defaults():
    m = Message(role="user", content="x")
    assert m.tool_calls == []
    assert m.tool_call_id is None


def test_factory_mock(tmp_path):
    provider = build_provider(load_settings(), build_default_tools(tmp_path))
    assert isinstance(provider, MockProvider)


def test_prefix_selection_strips_vendor(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "sk-test")
    settings = load_settings(model="openlux:qwen3.8-flash")
    provider = build_provider(settings, build_default_tools(tmp_path))
    assert isinstance(provider, OpenAIProvider)
    assert provider._model == "qwen3.8-flash"
    assert str(provider._client.base_url) == "https://api.openlux.ai/v1/"
    assert provider._client.headers["Authorization"] == "Bearer sk-test"


def test_default_vendor_for_bare_name(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "sk-test")
    settings = load_settings(model="qwen3.8-flash", vendor="openlux")
    provider = build_provider(settings, build_default_tools(tmp_path))
    assert isinstance(provider, OpenAIProvider)
    assert provider._model == "qwen3.8-flash"
    assert str(provider._client.base_url) == "https://api.openlux.ai/v1/"


def test_same_name_two_vendors_coexist(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "sk-relay")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-official")
    tools = build_default_tools(tmp_path)
    relay = build_provider(load_settings(model="openlux:qwen3.8-flash"), tools)
    official = build_provider(load_settings(model="dashscope:qwen3.8-flash"), tools)
    assert relay._model == official._model == "qwen3.8-flash"
    assert str(relay._client.base_url) == "https://api.openlux.ai/v1/"
    assert str(official._client.base_url) == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/")


def test_unknown_vendor_lists_known(tmp_path):
    settings = load_settings(model="nope:m")
    with pytest.raises(ProviderError, match="已知厂商"):
        build_provider(settings, build_default_tools(tmp_path))


def test_bare_name_without_default_vendor_fails(tmp_path):
    settings = load_settings(model="qwen3.8-flash")
    with pytest.raises(ProviderError, match="PORRY_AGENT_VENDOR"):
        build_provider(settings, build_default_tools(tmp_path))


def test_empty_default_vendor_treated_as_unset(tmp_path):
    settings = load_settings(model="qwen3.8-flash", vendor="")
    with pytest.raises(ProviderError, match="PORRY_AGENT_VENDOR"):
        build_provider(settings, build_default_tools(tmp_path))


def test_missing_key_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLUX_API_KEY", raising=False)
    settings = load_settings(model="openlux:m")
    with pytest.raises(ProviderError, match="OPENLUX_API_KEY"):
        build_provider(settings, build_default_tools(tmp_path))


def test_missing_key_mentions_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLUX_API_KEY", raising=False)
    settings = load_settings(model="openlux:m")
    with pytest.raises(ProviderError, match="重新启动"):
        build_provider(settings, build_default_tools(tmp_path))


def test_empty_model_after_colon_fails(tmp_path):
    settings = load_settings(model="openlux:")
    with pytest.raises(ProviderError, match="模型名为空"):
        build_provider(settings, build_default_tools(tmp_path))
