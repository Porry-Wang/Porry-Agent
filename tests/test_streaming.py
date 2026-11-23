import json

import pytest

from porry_agent.streaming import StreamAssembler, parse_sse_data


def _chunk(content=None, tool_calls=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def _line(chunk):
    return f"data: {json.dumps(chunk, ensure_ascii=False)}"


def test_parse_variants():
    chunk = _chunk(content="你好")
    assert parse_sse_data(_line(chunk)) == chunk
    assert parse_sse_data(f"data:{json.dumps(chunk)}") == chunk
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data(": ping") is None
    assert parse_sse_data("") is None
    assert parse_sse_data("  ") is None


def test_parse_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_sse_data("data: {broken")


def test_content_fragments_assemble():
    asm = StreamAssembler()
    assert asm.feed(_chunk(content="你好")) == "你好"
    assert asm.feed(_chunk(content="，")) == "，"
    assert asm.feed(_chunk(content="世界", finish="stop")) == "世界"
    assert asm.finish().content == "你好，世界"
    assert asm.finish_reason == "stop"


def test_tool_call_arguments_accumulate():
    asm = StreamAssembler()
    first = _chunk(tool_calls=[{
        "index": 0, "id": "call_1",
        "function": {"name": "read_file", "arguments": '{"pa'},
    }])
    assert asm.feed(first) == ""
    asm.feed(_chunk(tool_calls=[{
        "index": 0, "function": {"arguments": 'th": "a.py"}'},
    }]))
    msg = asm.finish()
    assert msg.content == ""
    assert msg.tool_calls[0].id == "call_1"
    assert msg.tool_calls[0].name == "read_file"
    assert msg.tool_calls[0].arguments == '{"path": "a.py"}'


def test_tool_calls_interleaved_by_index():
    asm = StreamAssembler()
    asm.feed(_chunk(tool_calls=[{
        "index": 1, "id": "c2", "function": {"name": "edit_file"},
    }]))
    asm.feed(_chunk(tool_calls=[{
        "index": 0, "id": "c1", "function": {"name": "read_file"},
    }]))
    calls = asm.finish().tool_calls
    assert [c.id for c in calls] == ["c1", "c2"]


def test_content_and_tool_calls_mixed():
    asm = StreamAssembler()
    assert asm.feed(_chunk(content="我先读文件。")) == "我先读文件。"
    asm.feed(_chunk(tool_calls=[{
        "index": 0, "id": "c1",
        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
    }]))
    msg = asm.finish()
    assert msg.content == "我先读文件。"
    assert len(msg.tool_calls) == 1


def test_truncation_recorded_not_raised():
    asm = StreamAssembler()
    asm.feed(_chunk(content="半截话", finish="length"))
    assert asm.finish_reason == "length"
    assert asm.finish().content == "半截话"


def test_usage_only_chunk_ignored():
    asm = StreamAssembler()
    assert asm.feed({"choices": [], "usage": {"total_tokens": 9}}) == ""
    assert asm.finish().content == ""


def test_name_split_fragments():
    asm = StreamAssembler()
    asm.feed(_chunk(tool_calls=[{
        "index": 0, "id": "c1", "function": {"name": "read_"},
    }]))
    asm.feed(_chunk(tool_calls=[{
        "index": 0, "function": {"name": "file"},
    }]))
    assert asm.finish().tool_calls[0].name == "read_file"


def test_empty_stream_finish():
    msg = StreamAssembler().finish()
    assert msg.role == "assistant"
    assert msg.content == ""
    assert msg.tool_calls == []