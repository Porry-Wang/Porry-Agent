import pytest

from porry_agent.messages import Message, ToolCall
from porry_agent.provider import MockProvider
from porry_agent.loop import LoopError, run_loop


class LoopingProvider:
    """永远请求工具调用，用来逼循环撞上限。"""

    def complete(self, messages: list[Message]) -> Message:
        return Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="noop")],
        )


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