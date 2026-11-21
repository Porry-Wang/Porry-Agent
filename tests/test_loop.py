import json

import pytest

from porry_agent.messages import Message, ToolCall
from porry_agent.provider import MockProvider
from porry_agent.loop import LoopError, run_loop
from porry_agent.tools import build_default_tools, make_executor, render_system_prompt


class LoopingProvider:
    """永远请求工具调用，用来逼循环撞上限。"""

    def complete(self, messages: list[Message]) -> Message:
        return Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="noop")],
        )


class ScriptedProvider:
    """按剧本走三步：读文件 → 改文件 → 汇报完成。"""

    def __init__(self) -> None:
        self.step = 0

    def complete(self, messages: list[Message]) -> Message:
        self.step += 1
        if self.step == 1:
            return Message(role="assistant", tool_calls=[ToolCall(
                id="c1", name="read_file",
                arguments=json.dumps({"path": "a.txt"}))])
        if self.step == 2:
            return Message(role="assistant", tool_calls=[ToolCall(
                id="c2", name="edit_file",
                arguments=json.dumps({
                    "path": "a.txt", "old_string": "world", "new_string": "there"}))])
        return Message(role="assistant", content="done")


def test_single_turn_completes():
    result = run_loop(MockProvider(), "hello", max_turns=5)
    assert result.stop_reason == "completed"
    assert result.turns == 1
    assert result.messages[-1].content == "mock: hello"


def test_max_turns_stops_and_pairs_results():
    result = run_loop(
        LoopingProvider(), "go", max_turns=3,
        execute_tool=lambda call: f"done:{call.name}")
    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    assert len(result.messages) == 7  # 1 user + 3 assistant + 3 tool
    tool_msg = result.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "c1"
    assert tool_msg.content == "done:noop"


def test_tool_calls_without_executor_raises():
    with pytest.raises(LoopError, match="工具执行器"):
        run_loop(LoopingProvider(), "go", max_turns=3)


def test_invalid_max_turns_raises():
    with pytest.raises(LoopError):
        run_loop(MockProvider(), "x", max_turns=0)


def test_end_to_end_file_edit(tmp_path):
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    tools = build_default_tools(tmp_path)

    result = run_loop(
        ScriptedProvider(), "把 world 改成 there", max_turns=5,
        system=render_system_prompt(tools),
        execute_tool=make_executor(tools))

    assert result.stop_reason == "completed"
    assert result.turns == 3
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello\nthere\n"
    assert result.messages[0].role == "system"
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]
    assert "hello" in tool_msgs[0].content
    assert "已编辑" in tool_msgs[1].content


def test_history_continues_across_calls():
    first = run_loop(MockProvider(), "第一问", max_turns=5, system="s")
    second = run_loop(MockProvider(), "第二问", max_turns=5,
                      system="s", history=first.messages)
    contents = [m.content for m in second.messages]
    assert "第一问" in contents and "mock: 第一问" in contents
    assert second.messages[-1].content == "mock: 第二问"
    assert sum(m.role == "system" for m in second.messages) == 1


def test_history_ignores_system_param():
    first = run_loop(MockProvider(), "一", max_turns=5, system="原系统")
    second = run_loop(MockProvider(), "二", max_turns=5,
                      system="新系统", history=first.messages)
    assert second.messages[0].content == "原系统"
    assert "新系统" not in [m.content for m in second.messages]


def test_history_not_mutated():
    first = run_loop(MockProvider(), "一", max_turns=5)
    snapshot = list(first.messages)
    run_loop(MockProvider(), "二", max_turns=5, history=first.messages)
    assert first.messages == snapshot
