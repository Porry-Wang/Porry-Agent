import json
import time

import httpx2
import pytest

from porry_agent import openai_provider
from porry_agent.messages import Message, ToolCall
from porry_agent.openai_provider import OpenAIProvider, ProviderError
from porry_agent.tools import ToolResult, ToolSpec

TEXT_REPLY = {"choices": [{"message": {"role": "assistant", "content": "你好"}}]}
TOOL_REPLY = {"choices": [{"message": {
    "role": "assistant", "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "read_file",
                                 "arguments": '{"path": "a.txt"}'}}]}}]}


def make_provider(handler, **overrides) -> OpenAIProvider:
    kwargs = dict(model="deepseek-flash", base_url="https://api.deepseek.com/v1",
                  api_key="sk-test", max_output_tokens=1024)
    kwargs.update(overrides)
    return OpenAIProvider(transport=httpx2.MockTransport(handler), **kwargs)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def test_text_reply_parsed():
    provider = make_provider(lambda request: httpx2.Response(200, json=TEXT_REPLY))
    reply = provider.complete([Message(role="user", content="hi")])
    assert reply.role == "assistant"
    assert reply.content == "你好"
    assert reply.tool_calls == []


def test_tool_calls_parsed():
    provider = make_provider(lambda request: httpx2.Response(200, json=TOOL_REPLY))
    reply = provider.complete([Message(role="user", content="读文件")])
    assert reply.content == ""
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert (call.id, call.name, call.arguments) == (
        "call_1", "read_file", '{"path": "a.txt"}')


def test_request_shape():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx2.Response(200, json=TEXT_REPLY)

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
    assert payload["model"] == "deepseek-flash"
    assert payload["max_tokens"] == 1024
    assert payload["tools"] == [{"type": "function",
                                 "function": {"name": "read_file", "description": "读",
                                              "parameters": {}}}]
    wire = payload["messages"]
    assert wire[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert wire[3] == {"role": "tool", "content": "hello", "tool_call_id": "c1"}


def test_retry_on_429_then_success(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx2.Response(429, text="rate limited")
        return httpx2.Response(200, json=TEXT_REPLY)

    reply = make_provider(handler).complete([Message(role="user", content="hi")])
    assert reply.content == "你好"
    assert calls["n"] == 2


def test_network_error_retries(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx2.ConnectError("connection refused")
        return httpx2.Response(200, json=TEXT_REPLY)

    reply = make_provider(handler).complete([Message(role="user", content="hi")])
    assert reply.content == "你好"
    assert calls["n"] == 2


def test_network_error_exhausted(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        raise httpx2.ConnectError("connection refused")

    with pytest.raises(ProviderError, match="网络错误"):
        make_provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == openai_provider.MAX_RETRIES + 1


def test_permanent_error_no_retry():
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(401, text="invalid key")

    with pytest.raises(ProviderError, match="401"):
        make_provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == 1


def test_retry_exhausted(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(503, text="unavailable")

    with pytest.raises(ProviderError, match="重试"):
        make_provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == openai_provider.MAX_RETRIES + 1


def test_malformed_response():
    provider = make_provider(
        lambda request: httpx2.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError, match="格式"):
        provider.complete([Message(role="user", content="hi")])


def test_follows_redirect():
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if request.url.scheme == "http":
            return httpx2.Response(
                307, headers={"Location": "https://api.deepseek.com/v1/chat/completions"})
        return httpx2.Response(200, json=TEXT_REPLY)

    provider = make_provider(handler, base_url="http://api.deepseek.com/v1")
    reply = provider.complete([Message(role="user", content="hi")])
    assert reply.content == "你好"
    assert calls["n"] == 2


def test_context_manager_closes_client():
    provider = make_provider(lambda request: httpx2.Response(200, json=TEXT_REPLY))
    with provider as p:
        assert p is provider
        p.complete([Message(role="user", content="hi")])
    assert provider._http_client.is_closed


def test_error_message_extracted_from_json():
    body = {"error": {"message": "Invalid API key provided"}}

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json=body)

    with pytest.raises(ProviderError, match="Invalid API key provided"):
        make_provider(handler).complete([Message(role="user", content="hi")])


def test_dict_arguments_normalized():
    body = {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "read_file",
                                     "arguments": {"path": "a.txt"}}}]}}]}
    provider = make_provider(lambda request: httpx2.Response(200, json=body))
    reply = provider.complete([Message(role="user", content="读文件")])
    assert reply.tool_calls[0].arguments == '{"path": "a.txt"}'


def test_length_finish_reason_raises():
    body = {"choices": [{"finish_reason": "length",
                         "message": {"role": "assistant", "content": "半句话"}}]}
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(200, json=body)

    with pytest.raises(ProviderError, match="截断"):
        make_provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == 1


def test_no_max_tokens_when_unset():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["payload"] = json.loads(request.content)
        return httpx2.Response(200, json=TEXT_REPLY)

    provider = make_provider(handler, max_output_tokens=None)
    provider.complete([Message(role="user", content="hi")])
    assert "max_tokens" not in seen["payload"]


def _sse_chunk(content=None, tool_calls=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def _sse_response(chunks):
    events = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks]
    events.append("data: [DONE]\n\n")
    body = "".join(events).encode("utf-8")
    return httpx2.Response(200, content=iter([body]))


def test_stream_content_and_callback():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["payload"] = json.loads(request.content)
        return _sse_response([
            _sse_chunk(content="你"),
            _sse_chunk(content="好", finish="stop"),
        ])

    fragments = []
    reply = make_provider(handler).stream_complete(
        [Message(role="user", content="hi")], on_text=fragments.append)
    assert seen["payload"]["stream"] is True
    assert fragments == ["你", "好"]
    assert reply.content == "你好"
    assert reply.tool_calls == []


def test_stream_tool_calls_silent():
    chunks = [
        _sse_chunk(tool_calls=[{"index": 0, "id": "c1",
                                "function": {"name": "read_file",
                                             "arguments": '{"pa'}}]),
        _sse_chunk(tool_calls=[{"index": 0,
                                "function": {"arguments": 'th": "a.txt"}'}}]),
        _sse_chunk(finish="tool_calls"),
    ]
    fragments = []
    reply = make_provider(lambda request: _sse_response(chunks)).stream_complete(
        [Message(role="user", content="读")], on_text=fragments.append)
    assert fragments == []
    assert reply.tool_calls[0].arguments == '{"path": "a.txt"}'


def test_stream_truncation_raises():
    provider = make_provider(lambda request: _sse_response([
        _sse_chunk(content="半句话", finish="length")]))
    with pytest.raises(ProviderError, match="截断"):
        provider.stream_complete(
            [Message(role="user", content="hi")], on_text=lambda t: None)


def test_stream_break_after_text_no_retry():
    calls = {"n": 0}

    def broken_body():
        yield 'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'.encode("utf-8")
        raise httpx2.ReadError("peer closed connection")

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(200, content=broken_body())

    fragments = []
    with pytest.raises(ProviderError, match="中断"):
        make_provider(handler).stream_complete(
            [Message(role="user", content="hi")], on_text=fragments.append)
    assert fragments == ["你"]
    assert calls["n"] == 1


def test_stream_connect_error_retries(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx2.ConnectError("connection refused")
        return _sse_response([_sse_chunk(content="你好", finish="stop")])

    reply = make_provider(handler).stream_complete(
        [Message(role="user", content="hi")], on_text=lambda t: None)
    assert reply.content == "你好"
    assert calls["n"] == 2


def test_stream_invalid_json_raises():
    provider = make_provider(
        lambda request: httpx2.Response(200, content=iter([b"data: {broken\n\n"])))
    with pytest.raises(ProviderError, match="JSON"):
        provider.stream_complete(
            [Message(role="user", content="hi")], on_text=lambda t: None)


def test_stream_without_finish_marker_completes():
    """真实中转形状：finish_reason 全程 null、[DONE] 收尾，应正常完成。"""
    provider = make_provider(lambda request: _sse_response([
        _sse_chunk(content="你好")]))
    reply = provider.stream_complete(
        [Message(role="user", content="hi")], on_text=lambda t: None)
    assert reply.content == "你好"


def test_stream_empty_choices_skipped():
    usage_chunk = {"choices": [],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                             "total_tokens": 2}}
    provider = make_provider(lambda request: _sse_response([
        _sse_chunk(content="你好", finish="stop"), usage_chunk]))
    reply = provider.stream_complete(
        [Message(role="user", content="hi")], on_text=lambda t: None)
    assert reply.content == "你好"


def test_stream_permanent_error_no_retry():
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(ProviderError, match="401"):
        make_provider(handler).stream_complete(
            [Message(role="user", content="hi")], on_text=lambda t: None)
    assert calls["n"] == 1