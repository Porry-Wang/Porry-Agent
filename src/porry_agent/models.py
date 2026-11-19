"""模型注册表：按模型名查到跑它所需的事实。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 可选：缺省表示未知，填了必须为正
    context_window: Optional[int] = Field(default=None, gt=0)
    max_output_tokens: Optional[int] = Field(default=None, gt=0)
    # 连接覆盖：留空则用全局兜底；api_key_env 只存环境变量名，不存明文密钥
    protocol: Optional[Literal["mock", "openai"]] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


# 端点为示例快照，使用前核对官网；加厂商优先用 MODELS_FILE 配置。
DEFAULT_MODELS: dict[str, ModelSpec] = {
    "mock": ModelSpec(context_window=200_000, max_output_tokens=8_000, protocol="mock"),
    "gpt-4o-mini": ModelSpec(
        context_window=128_000, max_output_tokens=16_384,
        protocol="openai", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
    ),
    "deepseek-flash": ModelSpec(
        context_window=1_000_000, max_output_tokens=393_216,
        protocol="openai", base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY",
    ),
    "qwen3.8-flash": ModelSpec(
        context_window=1_000_000, max_output_tokens=131_072,
        protocol="openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
    ),
}


class ModelRegistryError(Exception):
    """模型注册表错误。"""


def load_models(path: Optional[Path] = None) -> dict[str, ModelSpec]:
    """返回内置表，并用可选 TOML 文件按模型名覆盖。"""
    models = dict(DEFAULT_MODELS)
    if path is not None:
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ModelRegistryError(f"MODELS_FILE 指向的文件不存在：{path}") from None
        except tomllib.TOMLDecodeError as exc:
            raise ModelRegistryError(f"MODELS_FILE 不是合法 TOML：{exc}") from None
        for name, table in raw.items():
            try:
                models[name] = ModelSpec(**table)
            except ValidationError as exc:
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in exc.errors())
                raise ModelRegistryError(
                    f"MODELS_FILE 中模型 '{name}' 配置不合法：{detail}") from None
    return models


def resolve_model(name: str, models: dict[str, ModelSpec]) -> ModelSpec:
    """按名字查模型事实，未注册则报错并列出已知模型。"""
    spec = models.get(name)
    if spec is None:
        known = ", ".join(sorted(models))
        raise ModelRegistryError(
            f"未知模型 '{name}'。已注册：{known}。可在 MODELS_FILE 中补充。")
    return spec