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
    """读输入 → 跑一轮 → 打印回复；exit/quit/EOF/Ctrl-C 退出，单轮失败不结束会话。

    流式回复随到随打；结尾只补换行，不重打全文。
    """
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
        streamed = False

        def on_text(fragment: str) -> None:
            nonlocal streamed
            streamed = True
            print(fragment, end="", flush=True)

        try:
            result = run_loop(
                provider,
                text,
                max_turns=max_turns,
                system=system,
                execute_tool=execute_tool,
                history=history,
                on_text=on_text,
            )
        except KeyboardInterrupt:
            print()
            return 0
        except (ProviderError, LoopError) as exc:
            if streamed:
                print()
            print(f"本轮失败：{exc}", file=sys.stderr)
            continue
        history = result.messages
        if streamed:
            print()
        print(f"[done] turns={result.turns} stop={result.stop_reason}")
        if not streamed:
            print(result.messages[-1].content)
