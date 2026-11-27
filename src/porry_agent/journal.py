"""工作区文件事务日志：持久备份、状态机、锁与崩溃恢复。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

if os.name == "posix":
    import fcntl
elif os.name == "nt":
    import msvcrt

JOURNAL_DIRECTORY = Path(".porry_agent") / "journal"
TRANSACTION_DIRECTORY = JOURNAL_DIRECTORY / "transactions"
LOCK_FILENAME = "journal.lock"
MANIFEST_FILENAME = "manifest.json"
JOURNAL_VERSION = 1

JournalState = Literal["prepared", "applying", "committed", "needs_attention"]
JournalKind = Literal["update", "add", "delete"]


class JournalError(Exception):
    """事务日志失败，携带错误分类与可重试标记。"""

    def __init__(
        self, error_type: str, message: str, *, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


@dataclass(frozen=True)
class JournalFile:
    """一个路径在事务中的旧值、新值和权限。"""

    path: str
    kind: JournalKind
    old: bytes | None
    mode: int | None
    new: bytes | None
    new_mode: int | None


@dataclass(frozen=True)
class RecoveryReport:
    rolled_back: tuple[str, ...] = ()
    cleaned: tuple[str, ...] = ()
    needs_attention: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return bool(self.needs_attention or self.errors)


def fsync_directory(path: Path) -> None:
    """同步目录元数据；Windows 不支持目录 fsync 时跳过。"""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    """原子写入字节，保留或指定权限，并同步文件和父目录。"""
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    effective_mode = mode if mode is not None else existing_mode
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".porry-tmp-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if effective_mode is not None:
            os.chmod(temporary, effective_mode)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def _lock_descriptor(descriptor: int) -> None:
    try:
        if os.name == "posix":
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif os.name == "nt":
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            raise JournalError(
                "unsupported_platform", "当前平台不支持工作区事务锁。"
            )
    except OSError as exc:
        raise JournalError(
            "journal_locked", "工作区事务锁被占用；请稍后重试。", retryable=True
        ) from exc


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    elif os.name == "nt":
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


@contextmanager
def journal_lock(root: Path) -> Iterator[None]:
    """持有工作区事务锁；进程崩溃后操作系统会释放锁。"""
    base = root / JOURNAL_DIRECTORY
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        _lock_descriptor(descriptor)
        locked = True
        yield
    finally:
        if locked:
            _unlock_descriptor(descriptor)
        os.close(descriptor)


def _transaction_directory(root: Path, transaction_id: str) -> Path:
    return root / TRANSACTION_DIRECTORY / transaction_id


def _manifest_path(root: Path, transaction_id: str) -> Path:
    return _transaction_directory(root, transaction_id) / MANIFEST_FILENAME


def _resolve_entry_path(root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or "\x00" in raw_path:
        raise JournalError("journal_corrupt", f"事务日志路径非法：{raw_path}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise JournalError("journal_corrupt", f"事务日志路径越界：{raw_path}")
    return path


def _validate_manifest_entry(entry: object, transaction_id: str) -> None:
    """校验单条清单记录的形状，损坏归类为 journal_corrupt 而非运行时异常。"""
    invalid = JournalError("journal_corrupt", f"事务日志条目非法：{transaction_id}")
    if not isinstance(entry, dict):
        raise invalid
    if not isinstance(entry.get("path"), str) or entry.get("kind") not in {
        "update",
        "add",
        "delete",
    }:
        raise invalid
    for field in ("backup_name", "new_sha256"):
        value = entry.get(field)
        if value is not None and not isinstance(value, str):
            raise invalid
    for field in ("mode", "new_mode"):
        value = entry.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise invalid


def _read_manifest(root: Path, transaction_id: str) -> dict:
    path = _manifest_path(root, transaction_id)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JournalError(
            "journal_corrupt", f"无法读取事务日志 {transaction_id}：{exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("version") != JOURNAL_VERSION:
        raise JournalError("journal_corrupt", f"事务日志版本非法：{transaction_id}")
    if manifest.get("state") not in {
        "prepared",
        "applying",
        "committed",
        "needs_attention",
    }:
        raise JournalError("journal_corrupt", f"事务日志状态非法：{transaction_id}")
    if not isinstance(manifest.get("entries"), list):
        raise JournalError("journal_corrupt", f"事务日志条目非法：{transaction_id}")
    for entry in manifest["entries"]:
        _validate_manifest_entry(entry, transaction_id)
    return manifest


def _write_manifest(root: Path, transaction_id: str, manifest: dict) -> None:
    data = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    atomic_write_bytes(_manifest_path(root, transaction_id), data, mode=0o600)


def _validate_file(file: JournalFile) -> None:
    relative = Path(file.path)
    if (
        not file.path
        or relative.is_absolute()
        or ".." in relative.parts
        or "\x00" in file.path
    ):
        raise JournalError("invalid_args", f"事务日志路径非法：{file.path}")
    if file.kind == "update" and (file.old is None or file.new is None):
        raise JournalError("invalid_args", "update 事务必须包含旧值和新值。")
    if file.kind == "add" and (file.old is not None or file.new is None):
        raise JournalError("invalid_args", "add 事务必须没有旧值且包含新值。")
    if file.kind == "delete" and (file.old is None or file.new is not None):
        raise JournalError("invalid_args", "delete 事务必须包含旧值且没有新值。")


def begin_transaction(root: Path, files: Sequence[JournalFile]) -> str:
    """写入持久备份和 prepared 清单；调用方应持有事务锁。"""
    for file in files:
        _validate_file(file)
    transaction_id = uuid.uuid4().hex
    transactions = root / TRANSACTION_DIRECTORY
    transactions.mkdir(parents=True, exist_ok=True)
    directory = _transaction_directory(root, transaction_id)
    directory.mkdir()
    fsync_directory(transactions)
    backups = directory / "backups"
    backups.mkdir()
    entries: list[dict] = []
    for index, file in enumerate(files):
        backup_name = None
        if file.old is not None:
            backup_name = f"{index}.bak"
            atomic_write_bytes(backups / backup_name, file.old, mode=0o600)
        entries.append(
            {
                "path": file.path,
                "kind": file.kind,
                "existed": file.old is not None,
                "mode": file.mode,
                "new_sha256": (
                    hashlib.sha256(file.new).hexdigest()
                    if file.new is not None
                    else None
                ),
                "new_mode": file.new_mode,
                "backup_name": backup_name,
            }
        )
    manifest = {
        "version": JOURNAL_VERSION,
        "state": "prepared",
        "entries": entries,
    }
    _write_manifest(root, transaction_id, manifest)
    fsync_directory(backups)
    fsync_directory(directory)
    return transaction_id


def mark_state(root: Path, transaction_id: str, state: JournalState) -> None:
    manifest = _read_manifest(root, transaction_id)
    manifest["state"] = state
    _write_manifest(root, transaction_id, manifest)


def cleanup_transaction(root: Path, transaction_id: str) -> None:
    directory = _transaction_directory(root, transaction_id)
    if not directory.exists():
        return
    shutil.rmtree(directory)
    fsync_directory(directory.parent)


def _backup_bytes(directory: Path, entry: dict) -> bytes | None:
    backup_name = entry.get("backup_name")
    if backup_name is None:
        return None
    return (directory / "backups" / backup_name).read_bytes()


def _rollback_entry(root: Path, directory: Path, entry: dict) -> bool:
    """返回 True 表示已回到旧值；False 表示发现外部冲突。"""
    path = _resolve_entry_path(root, entry["path"])
    if path.exists() and not path.is_file():
        return False
    current = path.read_bytes() if path.is_file() else None
    old = _backup_bytes(directory, entry)
    if current == old:
        return True
    new_sha256 = entry.get("new_sha256")
    if new_sha256 is None:
        if current is None and old is not None:
            atomic_write_bytes(path, old, mode=entry.get("mode"))
            return True
        return False
    if current is not None and hashlib.sha256(current).hexdigest() == new_sha256:
        if entry["kind"] == "add":
            path.unlink()
            fsync_directory(path.parent)
        else:
            if old is None:
                return False
            atomic_write_bytes(path, old, mode=entry.get("mode"))
        return True
    return False


def _rollback_transaction_unlocked(root: Path, transaction_id: str) -> str:
    manifest = _read_manifest(root, transaction_id)
    state = manifest["state"]
    if state == "committed":
        cleanup_transaction(root, transaction_id)
        return "cleaned"
    if state == "needs_attention":
        return "needs_attention"
    if state not in {"prepared", "applying"}:
        raise JournalError("journal_corrupt", f"无法回滚事务状态：{state}")
    directory = _transaction_directory(root, transaction_id)
    conflicts: list[str] = []
    errors: list[str] = []
    for entry in reversed(manifest["entries"]):
        try:
            if not _rollback_entry(root, directory, entry):
                conflicts.append(entry["path"])
                break
        except (OSError, JournalError) as exc:
            errors.append(f"{entry.get('path', '?')}: {exc}")
            break
    if conflicts or errors:
        manifest["state"] = "needs_attention"
        manifest["conflicts"] = conflicts
        manifest["errors"] = errors
        _write_manifest(root, transaction_id, manifest)
        return "needs_attention"
    cleanup_transaction(root, transaction_id)
    return "rolled_back"


def recover_journal_unlocked(root: Path) -> RecoveryReport:
    """恢复所有遗留事务；调用方应持有事务锁。"""
    transactions = root / TRANSACTION_DIRECTORY
    if not transactions.exists():
        return RecoveryReport()
    rolled_back: list[str] = []
    cleaned: list[str] = []
    needs_attention: list[str] = []
    errors: list[str] = []
    for directory in sorted(transactions.iterdir()):
        if not directory.is_dir():
            continue
        transaction_id = directory.name
        try:
            manifest = _read_manifest(root, transaction_id)
            state = manifest["state"]
            if state == "committed":
                cleanup_transaction(root, transaction_id)
                cleaned.append(transaction_id)
            elif state in {"prepared", "applying"}:
                result = _rollback_transaction_unlocked(root, transaction_id)
                if result == "rolled_back":
                    rolled_back.append(transaction_id)
                elif result == "cleaned":
                    cleaned.append(transaction_id)
                else:
                    needs_attention.append(transaction_id)
            else:
                needs_attention.append(transaction_id)
        except (OSError, JournalError) as exc:
            errors.append(f"{transaction_id}: {exc}")
    return RecoveryReport(
        rolled_back=tuple(rolled_back),
        cleaned=tuple(cleaned),
        needs_attention=tuple(needs_attention),
        errors=tuple(errors),
    )


def recover_journal(root: Path) -> RecoveryReport:
    """获取事务锁并恢复遗留事务。"""
    with journal_lock(root):
        return recover_journal_unlocked(root)
