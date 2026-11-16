"""主循环：消息进 → 模型回 → 无工具调用即止，max_turns 兜底。"""

from __future__ import annotations

from typing import Callable, Literal, Optional

from pydantic import BaseModel

from porry_agent.messages import Message, ToolCall
from porry_agent.provider import Provider


class LoopError(Exception):
    """主循环错误。"""


class LoopResult(BaseModel):
    messages: list[Message]
    turns: int
    stop_reason: Literal["completed", "max_turns"]


def run_loop(
    provider: Provider,
    user_input: str,
    *,
    max_turns: int,
    execute_tool: Optional[Callable[[ToolCall], str]] = None,
) -> LoopResult:
    if max_turns < 1:
        raise LoopError("max_turns 必须大于 0。")
    messages = [Message(role="user", content=user_input)]
    for turn in range(1, max_turns + 1):
        reply = provider.complete(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return LoopResult(messages=messages, turns=turn, stop_reason="completed")
        if execute_tool is None:
            raise LoopError("模型请求了工具调用，但没有接入工具执行器。")
        for call in reply.tool_calls:
            messages.append(Message(
                role="tool", content=execute_tool(call), tool_call_id=call.id))
    return LoopResult(messages=messages, turns=max_turns, stop_reason="max_turns")