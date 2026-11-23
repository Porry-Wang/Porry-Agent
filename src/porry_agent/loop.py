"""主循环：消息进 → 模型回 → 无工具调用即止，max_turns 兜底。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

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
    system: str | None = None,
    execute_tool: Callable[[ToolCall], str] | None = None,
    history: list[Message] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> LoopResult:
    """跑 think→act→observe 循环，模型不再请求工具即完成，max_turns 兜底。

    on_text 非空走流式，文本碎片实时回调；否则走 complete。
    两条路径产出的历史结构一致。
    """
    if max_turns < 1:
        raise LoopError("max_turns 必须大于 0。")
    if history:
        messages: list[Message] = list(history)
    else:
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=user_input))
    for turn in range(1, max_turns + 1):
        if on_text is not None:
            reply = provider.stream_complete(messages, on_text=on_text)
        else:
            reply = provider.complete(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return LoopResult(messages=messages, turns=turn, stop_reason="completed")
        if execute_tool is None:
            raise LoopError("模型请求了工具调用，但没有接入工具执行器。")
        for call in reply.tool_calls:
            messages.append(
                Message(role="tool", content=execute_tool(call), tool_call_id=call.id)
            )
    return LoopResult(messages=messages, turns=max_turns, stop_reason="max_turns")
