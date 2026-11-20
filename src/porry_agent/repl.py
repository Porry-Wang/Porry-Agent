"""交互会话：多轮用户对话，历史跨轮累积，单轮失败不结束会话。"""

from __future__ import annotations

import sys
from collections.abc import Callable

from porry_agent.loop import LoopError, run_loop
from porry_agent.messages import Message, ToolCall
from porry_agent.openai_provider import ProviderError
from porry_agent.provider import Provider


def run_repl(
    provider: Provider,
    execute_tool: Callable[[ToolCall], str],
    *,
    max_turns: int,
    system: str | None = None,
) -> int:
    """读输入 → 跑一轮 → 打印回复；exit/quit/EOF/Ctrl-C 退出，单轮失败不结束会话。"""
    history: list[Message] = []
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return 0
        try:
            result = run_loop(
                provider,
                text,
                max_turns=max_turns,
                system=system,
                execute_tool=execute_tool,
                history=history,
            )
        except KeyboardInterrupt:
            print()
            return 0
        except (ProviderError, LoopError) as exc:
            print(f"本轮失败：{exc}", file=sys.stderr)
            continue
        history = result.messages
        print(f"[done] turns={result.turns} stop={result.stop_reason}")
        print(result.messages[-1].content)
