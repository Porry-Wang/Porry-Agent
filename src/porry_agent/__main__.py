"""命令行入口：一次性任务模式，串联配置/注册表/provider/工具/主循环。"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from porry_agent.config import ConfigError, load_settings
from porry_agent.logging_setup import setup_logging
from porry_agent.loop import LoopError, run_loop
from porry_agent.messages import ToolCall
from porry_agent.models import ModelRegistryError, load_models
from porry_agent.openai_provider import ProviderError
from porry_agent.provider import build_provider
from porry_agent.tools import build_default_registry, make_executor, render_system_prompt


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="porry_agent", description="PorryAgent 单任务入口")
    parser.add_argument("task", help="自然语言任务描述")
    parser.add_argument("--model", default=None, help="模型名，覆盖 PORRY_AGENT_MODEL")
    args = parser.parse_args(argv)

    try:
        overrides = {"model": args.model} if args.model else {}
        settings = load_settings(**overrides)
        setup_logging(settings.log_level)
        models = load_models(settings.models_file)
        registry = build_default_registry(settings.resolved_workspace_root)
        provider = build_provider(settings, models, registry)
    except (ConfigError, ModelRegistryError, ProviderError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    executor = make_executor(registry)

    def traced(call: ToolCall) -> str:
        print(f"[tool] {call.name} {call.arguments}")
        output = executor(call)
        print(f"[result] {output[:200]}")
        return output

    print(f"模型：{settings.model}｜工作目录：{settings.resolved_workspace_root}")
    try:
        result = run_loop(
            provider, args.task, max_turns=settings.max_turns,
            system=render_system_prompt(registry), execute_tool=traced)
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