"""命令行入口：一次性任务与交互会话双模式，串联配置/provider/工具/主循环。"""

from __future__ import annotations

import argparse
import sys

from porry_agent.config import ConfigError, Settings, load_settings
from porry_agent.logging_setup import setup_logging
from porry_agent.loop import LoopError, run_loop
from porry_agent.messages import ToolCall
from porry_agent.openai_provider import ProviderError
from porry_agent.provider import MockProvider, build_provider
from porry_agent.repl import run_repl
from porry_agent.tools import (
    build_default_tools,
    make_executor,
    render_system_prompt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="porry_agent", description="PorryAgent 任务入口"
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="自然语言任务描述；省略则进入交互模式",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型选择：「厂商:模型名」或裸模型名（用默认厂商）",
    )
    args = parser.parse_args(argv)

    settings: Settings | None = None
    try:
        overrides = {"model": args.model} if args.model is not None else {}
        settings = load_settings(**overrides)
        setup_logging(settings.log_level)
        tools = build_default_tools(settings.resolved_workspace_root)
        provider = build_provider(settings, tools)
    except (ConfigError, ProviderError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        if settings is not None:
            print(
                f"默认厂商：{settings.vendor or '未设置'}｜"
                f"工作目录：{settings.resolved_workspace_root}",
                file=sys.stderr,
            )
        return 1

    kind = "离线 mock" if isinstance(provider, MockProvider) else "OpenAI 兼容"
    print(
        f"模型：{settings.model}｜默认厂商：{settings.vendor or '未设置'}｜"
        f"类型：{kind}｜工作目录：{settings.resolved_workspace_root}"
    )
    if isinstance(provider, MockProvider) and "model" not in settings.model_fields_set:
        print(
            "[警告] 未指定模型，当前为离线 mock（只回声，不调用真实模型）。"
            "设置 PORRY_AGENT_MODEL 或 --model 接入真实模型。",
            file=sys.stderr,
        )

    executor = make_executor(tools)

    def traced(call: ToolCall) -> str:
        print(f"[tool] {call.name} {call.arguments}")
        output = executor(call)
        print(f"[result] {output[:200]}")
        return output

    try:
        if args.task is None:
            return run_repl(
                provider,
                traced,
                max_turns=settings.max_turns,
                system=render_system_prompt(tools),
            )
        result = run_loop(
            provider,
            args.task,
            max_turns=settings.max_turns,
            system=render_system_prompt(tools),
            execute_tool=traced,
        )
    except (ProviderError, LoopError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            close()

    print(f"[done] turns={result.turns} stop={result.stop_reason}")
    print(result.messages[-1].content)
    return 0 if result.stop_reason == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
