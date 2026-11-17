"""配置加载与校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "PORRY_AGENT_"


class ConfigError(Exception):
    """面向用户的配置错误。"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_file=".env",
        env_file_encoding="utf-8", extra="forbid",
    )

    # 模型接入：全局兜底，注册表里的模型条目可覆盖
    provider: Literal["mock", "openai"] = "mock"
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[SecretStr] = None
    model: str = "mock"
    models_file: Optional[Path] = None

    # 运行上限
    max_turns: int = Field(default=50, gt=0)

    # 路径
    workspace_root: Optional[Path] = None

    # 日志
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def resolved_workspace_root(self) -> Path:
        return (self.workspace_root or Path.cwd()).expanduser().resolve()


def load_settings(**overrides: object) -> Settings:
    """加载配置，失败抛 ConfigError。"""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        lines: list[str] = []
        for err in exc.errors():
            field = "_".join(str(p) for p in err["loc"])
            lines.append(f"{ENV_PREFIX}{field.upper()} 取值不合法：{err['msg']}")
        raise ConfigError("配置有误：\n" + "\n".join(lines)) from None