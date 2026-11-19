"""配置加载与校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "PORRY_AGENT_"


class ConfigError(Exception):
    """面向用户的配置错误。"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_file=".env",
        env_file_encoding="utf-8", extra="forbid",
    )

    # 模型接入：模型名可带「厂商:」前缀；不带前缀时用默认厂商
    model: str = "mock"
    vendor: Optional[str] = None

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
    """加载配置，失败抛 ConfigError；报错按来源归因（命令行参数或环境变量）。"""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        lines: list[str] = []
        for err in exc.errors():
            field = "_".join(str(p) for p in err["loc"])
            if field in overrides:
                lines.append(f"命令行参数 {field} 取值不合法：{err['msg']}")
            else:
                lines.append(f"{ENV_PREFIX}{field.upper()} 取值不合法：{err['msg']}")
        raise ConfigError("配置有误：\n" + "\n".join(lines)) from None