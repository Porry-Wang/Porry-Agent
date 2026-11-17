"""工具层：契约、注册表、首批文件工具与循环执行器。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from porry_agent.messages import ToolCall


class ToolResult(BaseModel):
    """工具执行的统一回执。"""

    ok: bool
    content: str
    error_type: Optional[str] = None
    retryable: bool = False


class ToolError(Exception):
    """工具内部主动抛出的错误，带错误分类与可重试标记。"""

    def __init__(self, error_type: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


class ToolSpec(BaseModel):
    """一个工具的完整描述：名字、说明、参数 schema 与执行函数。"""

    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], ToolResult]
    read_only: bool = False

def _resolve_in(root: Path, raw: object) -> Path:
    """把相对路径解析成工作目录内的绝对路径，参数非法或越界抛 ToolError。"""
    if not isinstance(raw, str) or not raw:
        raise ToolError("invalid_args", "缺少 path 参数。")
    candidate = (root / raw).resolve()
    if not candidate.is_relative_to(root):
        raise ToolError("outside_workspace", f"路径越界：{raw} 不在工作目录内。")
    return candidate


def _atomic_write(path: Path, text: str) -> None:
    """先写临时文件再原子替换；失败清理临时文件，原文件不受影响。"""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".porry-tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_default_registry(workspace_root: Path) -> dict[str, ToolSpec]:
    """造默认工具注册表（read_file / edit_file），先读后写的账本随每套注册表独立。"""
    root = workspace_root.resolve()
    read_paths: set[Path] = set()

    def read_file(args: dict) -> ToolResult:
        """读取文本文件，并把路径记入先读后写账本。"""
        path = _resolve_in(root, args.get("path"))
        if not path.is_file():
            raise ToolError("not_found", f"文件不存在：{path.relative_to(root)}")
        text = path.read_text(encoding="utf-8")
        read_paths.add(path)
        return ToolResult(ok=True, content=text)

    def edit_file(args: dict) -> ToolResult:
        """唯一匹配替换：查账本、重读现场、原子写、写后校验失败则回滚。"""
        path = _resolve_in(root, args.get("path"))
        old, new = args.get("old_string"), args.get("new_string")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ToolError("invalid_args", "需要非空 old_string 与字符串 new_string。")
        if path not in read_paths:
            raise ToolError("stale_read", "先 read_file 再编辑。")
        if not path.is_file():
            raise ToolError("not_found", f"文件不存在：{path.relative_to(root)}")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolError("no_match", "old_string 在文件中未找到。")
        if count > 1:
            lines, pos = [], text.find(old)
            while pos != -1:
                # 数一下它前面有多少个换行符，就是它前面有多少行
                lines.append(str(text[:pos].count("\n") + 1))
                pos = text.find(old, pos + len(old))
            raise ToolError(
                "multiple_matches",
                f"old_string 匹配 {count} 处（行 {', '.join(lines)}），请加长上下文使其唯一。")
        updated = text.replace(old, new, 1)
        _atomic_write(path, updated)
        if path.read_text(encoding="utf-8") != updated:
            _atomic_write(path, text)
            raise ToolError("io_error", "写后校验失败，已回滚原内容。", retryable=True)
        return ToolResult(ok=True, content=f"已编辑 {path.relative_to(root)}，替换 1 处。")

    return {
        "read_file": ToolSpec(
            name="read_file", description="读取工作目录内的文本文件",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "相对工作目录的文件路径"}},
                "required": ["path"]},
            handler=read_file, read_only=True),
        "edit_file": ToolSpec(
            name="edit_file", description="把文件中唯一匹配的 old_string 替换为 new_string",
            parameters={"type": "object", "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"}},
                "required": ["path", "old_string", "new_string"]},
            handler=edit_file),
    }


def make_executor(registry: dict[str, ToolSpec]) -> Callable[[ToolCall], str]:
    """把注册表包装成主循环的执行回调：查工具、解析参数、执行，失败一律转成错误文本。"""
    def execute(call: ToolCall) -> str:
        """执行单个工具调用，任何失败都转成错误文本，不向主循环抛异常。"""
        spec = registry.get(call.name)
        if spec is None:
            return f"[工具错误 unknown_tool] 未知工具：{call.name}"
        try:
            args = json.loads(call.arguments) if call.arguments else {}
            if not isinstance(args, dict):
                raise ValueError("参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            return f"[工具错误 invalid_args] 参数解析失败：{exc}"
        try:
            result = spec.handler(args)
        except ToolError as exc:
            result = ToolResult(ok=False, content=str(exc),
                                error_type=exc.error_type, retryable=exc.retryable)
        except Exception as exc:
            result = ToolResult(ok=False, content=f"工具执行异常：{exc}", error_type="tool_error")
        if result.ok:
            return result.content
        return f"[工具错误 {result.error_type}] {result.content}"
    return execute


def render_system_prompt(registry: dict[str, ToolSpec]) -> str:
    """把注册表渲染成 system prompt：工具清单、参数 schema 与编辑规矩。"""
    lines = ["你是编程助手，通过工具操作文件。可用工具："]
    for spec in registry.values():
        lines.append(f"- {spec.name}：{spec.description}。"
                     f"参数 schema：{json.dumps(spec.parameters, ensure_ascii=False)}")
    lines.append("编辑前必须先 read_file；old_string 必须唯一匹配；路径相对工作目录。")
    return "\n".join(lines)