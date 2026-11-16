"""Provider 接口与离线 mock 实现。"""

from __future__ import annotations

from typing import Protocol

from porry_agent.messages import Message


class Provider(Protocol):
    def complete(self, messages: list[Message]) -> Message: ...


class MockProvider:
    """离线确定性回复：回声最近一条用户消息。"""

    def complete(self, messages: list[Message]) -> Message:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), "")
        return Message(role="assistant", content=f"mock: {last_user}")