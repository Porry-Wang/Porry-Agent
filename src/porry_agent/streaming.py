"""SSE 行解析与 delta 拼装：纯逻辑不碰网络，策略报错归 provider 层。"""

from __future__ import annotations

import json
from typing import Any

from porry_agent.messages import Message, ToolCall


def parse_sse_data(line: str) -> dict[str, Any] | None:
    """取一行 SSE：data 行返回 payload；空行、注释、[DONE] 返回 None。

    非法 JSON 抛 json.JSONDecodeError，由调用方包成 ProviderError。
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


class StreamAssembler:
    """把 chat.completion.chunk 的 delta 拼成完整 Message；截断只记录不抛错。"""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._calls: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None

    def feed(self, chunk: dict[str, Any]) -> str:
        """喂一个 chunk，返回本次新增 content（供实时打印）；tool_calls 静默拼装。"""
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        if text:
            self._content.append(text)
        for call in delta.get("tool_calls") or []:
            slot = self._calls.setdefault(
                call.get("index", 0), {"id": "", "name": "", "arguments": ""}
            )
            if call.get("id"):
                slot["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                slot["name"] += function["name"]
            if function.get("arguments"):
                slot["arguments"] += function["arguments"]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        return text

    def finish(self) -> Message:
        """产出最终 Message：content 拼接，tool_calls 按 index 排序。"""
        calls = [
            ToolCall(id=slot["id"], name=slot["name"], arguments=slot["arguments"])
            for _, slot in sorted(self._calls.items())
        ]
        return Message(
            role="assistant", content="".join(self._content), tool_calls=calls
        )