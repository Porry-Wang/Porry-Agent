import json

import httpx
import pytest

from porry_agent import openai_provider
from porry_agent.config import load_settings
from porry_agent.messages import Message, ToolCall
from porry_agent.models import ModelSpec, load_models
from porry_agent.openai_provider import OpenAIProvider, ProviderError, resolve_connection
from porry_agent.provider import MockProvider, build_provider
from porry_agent.tools import ToolResult, ToolSpec, build_default_registry

TEXT_REPLY = {"choices": [{"message": {"role": "assistant", "content": "你好"}}]}
TOOL_REPLY = {"choices": [{"message": {
    "role": "assistant", "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "read_file",
                                 "arguments": '{"path": "a.txt"}'}}]}}]}


def make_provider(handler, **overrides) -> OpenAIProvider:
    kwargs = dict(model="deepseek-chat", base_url="https://api.deepseek.com/v1",
                  api_key="sk-test", max_output_tokens=1024)
    kwargs.update(overrides)
    return OpenAIProvider(transport=httpx.MockTransport(handler), **kwargs)


def test_text_reply_parsed():
    provider = make_provider(lambda request: httpx.Response(200, json=TEXT_REPLY))
    reply = provider.complete([Message(role="user", content="hi")])
    assert reply.role == "assistant"
    assert reply.content == "你好"
    assert reply.tool_calls == []


def test_tool_calls_parsed():
    provider = make_provider(lambda request: httpx.Response(200, json=TOOL_REPLY))
    reply = provider.complete([Message(role="user", content="读文件")])
    assert reply.content == ""
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert (call.id, call.name, call.arguments) == (
        "call_1", "read_file", '{"path": "a.txt"}')


def test_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=TEXT_REPLY)

    spec = ToolSpec(name="read_file", description="读", parameters={},
                    handler=lambda args: ToolResult(ok=True, content=""))
    provider = make_provider(handler, tool_specs=[spec])
    messages = [
        Message(role="system", content="规则"),
        Message(role="user", content="改文件"),
        Message(role="assistant", tool_calls=[
            ToolCall(id="c1", name="read_file", arguments='{"path": "a.txt"}')]),
        Message(role="tool", content="hello", tool_call_id="c1"),
    ]
    provider.complete(messages)

    assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    payload = seen["payload"]
    assert payload["model"] == "deepseek-chat"
    assert payload["max_tokens"] == 1024
    assert payload["tools"] == [{"type": "function",
                                 "function": {"name": "read_file", "description": "读",
                                              "parameters": {}}}]
    wire = payload["messages"]
    assert wire[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert wire[3] == {"role": "tool", "content": "hello", "tool_call_id": "c1"}


def test_resolve_missing_env_key():
    spec = ModelSpec(context_window=1000, max_output_tokens=100, api_key_env="MY_KEY")
    with pytest.raises(ProviderError, match="MY_KEY"):
        resolve_connection("m", spec, fallback_base_url="https://x.com/v1",
                           fallback_api_key="sk-fallback", env={})


def test_resolve_model_overrides_fallback():
    spec = ModelSpec(context_window=1000, max_output_tokens=100,
                     base_url="https://api.deepseek.com/v1", api_key_env="MY_KEY")
    base_url, key = resolve_connection(
        "m", spec, fallback_base_url="https://api.openai.com/v1",
        fallback_api_key="sk-fallback", env={"MY_KEY": "sk-model"})
    assert base_url == "https://api.deepseek.com/v1"
    assert key == "sk-model"


def test_resolve_fallback_when_spec_empty():
    spec = ModelSpec(context_window=1000, max_output_tokens=100)
    base_url, key = resolve_connection(
        "m", spec, fallback_base_url="https://api.openai.com/v1",
        fallback_api_key="sk-fallback", env={})
    assert base_url == "https://api.openai.com/v1"
    assert key == "sk-fallback"


def test_resolve_missing_all_keys():
    spec = ModelSpec(context_window=1000, max_output_tokens=100)
    with pytest.raises(ProviderError, match="PORRY_AGENT_API_KEY"):
        resolve_connection("m", spec, fallback_base_url="https://x.com/v1",
                           fallback_api_key=None, env={})


def test_resolve_invalid_base_url():
    spec = ModelSpec(context_window=1000, max_output_tokens=100)
    with pytest.raises(ProviderError, match="base_url"):
        resolve_connection("m", spec, fallback_base_url="not-a-url",
                           fallback_api_key="sk", env={})


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setattr(openai_provider, "RETRY_BACKOFF_SECONDS", 0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=TEXT_REPLY)

    reply = make_provider(handler).complete([Message(role="user", content="hi")])
    assert reply.content == "你好"
    assert calls["n"] == 2


def test_network_error_retries(monkeypatch):
    monkeypatch.setattr(openai_provider, "RETRY_BACKOFF_SECONDS", 0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=TEXT_REPLY)

    reply = make_provider(handler).complete([Message(role="user", content="hi")])
    assert reply.content == "你好"
    assert calls["n"] == 2


def test_permanent_error_no_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="invalid key")

    with pytest.raises(ProviderError, match="401"):
        make_provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == 1


def test_retry_exhausted(monkeypatch):
    monkeypatch.setattr(openai_provider, "RETRY_BACKOFF_SECONDS", 0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(ProviderError, match="重试"):
        make_provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == openai_provider.MAX_RETRIES + 1


def test_malformed_response():
    provider = make_provider(
        lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError, match="格式"):
        provider.complete([Message(role="user", content="hi")])


def test_factory_mock(tmp_path):
    settings = load_settings()
    provider = build_provider(settings, load_models(), build_default_registry(tmp_path))
    assert isinstance(provider, MockProvider)


def test_factory_openai(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    settings = load_settings(model="deepseek-chat")
    provider = build_provider(settings, load_models(), build_default_registry(tmp_path))
    assert isinstance(provider, OpenAIProvider)


def test_follows_redirect():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.scheme == "http":
            return httpx.Response(
                307, headers={"Location": "https://api.deepseek.com/v1/chat/completions"})
        return httpx.Response(200, json=TEXT_REPLY)

    provider = make_provider(handler, base_url="http://api.deepseek.com/v1")
    reply = provider.complete([Message(role="user", content="hi")])
    assert reply.content == "你好"
    assert calls["n"] == 2


def test_context_manager_closes_client():
    provider = make_provider(lambda request: httpx.Response(200, json=TEXT_REPLY))
    with provider as p:
        assert p is provider
        p.complete([Message(role="user", content="hi")])
    assert provider._client.is_closed


def test_error_message_extracted_from_json():
    body = {"error": {"message": "Invalid API key provided"}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json=body)

    with pytest.raises(ProviderError, match="Invalid API key provided"):
        make_provider(handler).complete([Message(role="user", content="hi")])


def test_dict_arguments_normalized():
    body = {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "read_file",
                                     "arguments": {"path": "a.txt"}}}]}}]}
    provider = make_provider(lambda request: httpx.Response(200, json=body))
    reply = provider.complete([Message(role="user", content="读文件")])
    assert reply.tool_calls[0].arguments == '{"path": "a.txt"}'


def test_length_finish_reason_raises():
    body = {"choices": [{"finish_reason": "length",
                         "message": {"role": "assistant", "content": "半句话"}}]}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderError, match="截断"):
        make_provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == 1