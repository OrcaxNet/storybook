"""Storybook setup 的 Agent adapter 基础设施。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class AdapterError(RuntimeError):
    """Agent 配置不可安全读取或合并。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Launcher:
    """可跨 cwd 启动 Storybook 的稳定命令。"""

    command: str
    args: tuple[str, ...] = ()

    def mcp_node(self) -> dict[str, Any]:
        return {"command": self.command, "args": [*self.args, "mcp"], "env": {}}


@dataclass(frozen=True)
class AdapterContext:
    home: Path
    environ: Mapping[str, str]
    launcher: Launcher


@dataclass(frozen=True)
class AdapterPlan:
    adapter: str
    display_name: str
    detected: bool
    selected: bool
    changed: bool
    targets: tuple[str, ...]
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "display_name": self.display_name,
            "detected": self.detected,
            "selected": self.selected,
            "changed": self.changed,
            "targets": list(self.targets),
            "summary": self.summary,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(
            "SB_SETUP_CONFIG_INVALID", f"无法解析 {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AdapterError(
            "SB_SETUP_CONFIG_INVALID", f"{path} 的 JSON 根节点必须是 object"
        )
    return value


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    """同目录临时文件 + replace，避免配置只写一半。"""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temp.chmod(0o600)
        os.replace(temp, path)
        if os.name != "nt":
            path.chmod(0o600)
    finally:
        if temp.exists():
            temp.unlink()


def backup_file(path: Path, backup_dir: Path, label: str) -> dict[str, Any]:
    """备份完整原文件，并返回可审计的 hash 元数据。"""

    record: dict[str, Any] = {"target": str(path), "existed": path.exists()}
    if not path.exists():
        record["before_sha256"] = None
        record["backup"] = None
        return record
    data = path.read_bytes()
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup = backup_dir / f"{label}-{path.name}.bak"
    shutil.copy2(path, backup)
    if os.name != "nt":
        backup.chmod(0o600)
    record.update({"before_sha256": sha256_bytes(data), "backup": str(backup)})
    return record


class AgentAdapter(ABC):
    name: str
    display_name: str

    @abstractmethod
    def detected(self, context: AdapterContext) -> bool:
        """当前用户是否安装/配置了该 Agent。"""

    @abstractmethod
    def plan(self, context: AdapterContext, *, selected: bool) -> AdapterPlan:
        """只读计算计划。"""

    @abstractmethod
    def apply(self, context: AdapterContext, backup_dir: Path) -> dict[str, Any]:
        """合并 Storybook 节点并返回卸载所需状态。"""

    @abstractmethod
    def uninstall(self, context: AdapterContext, state: Mapping[str, Any]) -> dict[str, Any]:
        """仅恢复本 adapter 写入的节点。"""

    @abstractmethod
    def verify(self, context: AdapterContext) -> tuple[bool, str]:
        """验证配置中的 Storybook 节点。"""
