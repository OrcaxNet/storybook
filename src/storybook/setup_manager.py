"""一键 setup / uninstall 编排。"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

from . import config, embeddings, health, search, store
from .profiles import PlatformRoots, ProfileError
from .setup_adapters import (
    AdapterContext,
    AdapterError,
    AgentAdapter,
    Launcher,
    get_adapters,
)
from .setup_adapters.base import AdapterPlan, atomic_write


STATE_SCHEMA_VERSION = 1
Progress = Callable[[dict[str, Any]], None]


class SetupError(RuntimeError):
    """带稳定 error code 的 setup 失败。"""

    def __init__(self, code: str, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "hint": self.hint}


def default_launcher() -> Launcher:
    """优先使用安装后的 console script，源码运行时回退到 ``python -m``。"""

    executable = shutil.which("storybook")
    if executable:
        return Launcher(str(Path(executable).resolve()))
    return Launcher(sys.executable, ("-m", "storybook.cli"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_size(size: int | None) -> str | None:
    if size is None:
        return None
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return None


def _ollama_tags() -> dict[str, dict[str, Any]]:
    response = requests.get(f"{config.EMBED_BASE_URL}/api/tags", timeout=3)
    response.raise_for_status()
    payload = response.json()
    return {
        str(item.get("name")): item
        for item in payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }


def _pull_model(model: str, progress: Progress | None = None) -> None:
    with requests.post(
        f"{config.EMBED_BASE_URL}/api/pull",
        json={"name": model, "stream": True},
        stream=True,
        timeout=(3, None),
    ) as response:
        response.raise_for_status()
        for raw in response.iter_lines():
            if not raw:
                continue
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            total = event.get("total")
            completed = event.get("completed")
            percent = (
                round(float(completed) / float(total) * 100, 1)
                if total and completed is not None
                else None
            )
            if progress:
                progress(
                    {
                        "phase": "model",
                        "model": model,
                        "status": event.get("status", "downloading"),
                        "size_bytes": total,
                        "size": _format_size(total),
                        "completed_bytes": completed,
                        "percent": percent,
                    }
                )


@dataclass(frozen=True)
class SetupPlan:
    profile: dict[str, Any]
    embedding: dict[str, Any]
    adapters: tuple[dict[str, Any], ...]
    models: tuple[str, ...]
    legacy_databases: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "embedding": self.embedding,
            "adapters": list(self.adapters),
            "models": list(self.models),
            "legacy_databases": list(self.legacy_databases),
            "writes": [
                target
                for adapter in self.adapters
                if adapter["selected"] and adapter["changed"]
                for target in adapter["targets"]
            ],
        }


@dataclass(frozen=True)
class _FileSnapshot:
    """事务开始前的文件内容与元数据，用于精确回滚用户配置。"""

    path: Path
    existed: bool
    data: bytes | None
    mode: int | None
    atime_ns: int | None
    mtime_ns: int | None
    missing_parents: tuple[Path, ...]

    @classmethod
    def capture(cls, path: Path) -> _FileSnapshot:
        missing_parents: list[Path] = []
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            missing_parents.append(parent)
            parent = parent.parent
        if not path.exists():
            return cls(path, False, None, None, None, None, tuple(missing_parents))
        metadata = path.stat()
        return cls(
            path,
            True,
            path.read_bytes(),
            metadata.st_mode & 0o7777,
            metadata.st_atime_ns,
            metadata.st_mtime_ns,
            tuple(missing_parents),
        )

    def restore_content(self) -> None:
        if not self.existed:
            self.path.unlink(missing_ok=True)
            return
        assert self.data is not None
        atomic_write(self.path, self.data)
        if self.mode is not None:
            self.path.chmod(self.mode)
        if self.atime_ns is not None and self.mtime_ns is not None:
            os.utime(self.path, ns=(self.atime_ns, self.mtime_ns))

    def remove_created_parents(self) -> None:
        for parent in self.missing_parents:
            try:
                parent.rmdir()
            except FileNotFoundError:
                continue
            except OSError:
                break


class SetupManager:
    def __init__(
        self,
        *,
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        launcher: Launcher | None = None,
        adapters: Sequence[AgentAdapter] | None = None,
        roots: PlatformRoots | None = None,
    ) -> None:
        self.home = Path.home() if home is None else Path(home)
        self.environ = os.environ if environ is None else environ
        self.launcher = launcher or default_launcher()
        self.adapters = tuple(get_adapters() if adapters is None else adapters)
        self.roots = roots or config.PROFILE_REGISTRY.roots
        self.context = AdapterContext(self.home, self.environ, self.launcher)

    @property
    def state_path(self) -> Path:
        return self.roots.state / "setup-state.json"

    def _selected_names(self, requested: Iterable[str] | None) -> set[str]:
        if requested:
            selected = set(requested)
            known = {adapter.name for adapter in self.adapters}
            unknown = selected - known
            if unknown:
                raise SetupError(
                    "SB_SETUP_AGENT_UNKNOWN",
                    f"未知 adapter: {', '.join(sorted(unknown))}",
                    hint=f"可选: {', '.join(sorted(known))}",
                )
            return selected
        return {
            adapter.name for adapter in self.adapters if adapter.detected(self.context)
        }

    def _legacy_databases(self) -> tuple[str, ...]:
        current = config.DB_PATH.resolve(strict=False)
        candidates = {
            Path.cwd() / "data" / "memory.db",
            config.BASE_DIR / "data" / "memory.db",
            Path.cwd() / "memory.db",
        }
        return tuple(
            str(path)
            for path in sorted(candidates, key=str)
            if path.is_file() and path.resolve(strict=False) != current
        )

    def plan(self, requested_agents: Iterable[str] | None = None) -> SetupPlan:
        """只读生成完整计划；不得创建 Profile、目录或网络请求。"""

        selected = self._selected_names(requested_agents)
        adapter_plans: list[dict[str, Any]] = []
        try:
            for adapter in self.adapters:
                if adapter.name not in selected:
                    adapter_plans.append(
                        AdapterPlan(
                            adapter=adapter.name,
                            display_name=adapter.display_name,
                            detected=adapter.detected(self.context),
                            selected=False,
                            changed=False,
                            targets=(),
                            summary="not selected",
                        ).as_dict()
                    )
                    continue
                adapter_plans.append(
                    adapter.plan(self.context, selected=True).as_dict()
                )
        except AdapterError as exc:
            raise SetupError(exc.code, str(exc), hint="修复配置语法后重试") from exc
        try:
            profile = config.PROFILE_REGISTRY.peek_active_profile()
        except ProfileError as exc:
            raise SetupError(
                "SB_SETUP_PROFILE_INVALID",
                str(exc),
                hint="修复或从备份恢复 Profile registry 后重试",
            ) from exc
        profile_plan = {
            "action": "reuse" if profile else "create",
            "id": profile.id if profile else None,
            "display_name": profile.display_name if profile else "default",
            "data_root": str(
                self.roots.data / "profiles" / (profile.id if profile else "<profile-uuid>")
            ),
            "sync_state": profile.sync_state if profile else "local_only",
        }
        return SetupPlan(
            profile=profile_plan,
            embedding={
                "type": config.EMBED_TYPE,
                "preset": config.EMBED_PRESET,
                "adapter": config.EMBED_ADAPTER,
                "base_url": config.EMBED_BASE_URL,
                "model": config.EMBED_MODEL,
                "dimension": config.EMBED_DIM,
                "version": config.EMBED_VERSION,
                "config_source": config.EMBED_CONFIG_SOURCE,
                "config_normalized": config.EMBED_CONFIG_NORMALIZED,
                "remote_text_disclosure": config.embedding_text_leaves_device(),
            },
            adapters=tuple(adapter_plans),
            models=(config.EMBED_MODEL,),
            legacy_databases=self._legacy_databases(),
        )

    def _load_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SetupError(
                "SB_SETUP_STATE_INVALID",
                f"无法读取 {self.state_path}: {exc}",
                hint="从 setup-backups 恢复 setup-state.json 后重试；不要覆盖损坏的 state",
            ) from exc
        self._validate_state(state)
        return state

    def _invalid_state(self, detail: str) -> SetupError:
        return SetupError(
            "SB_SETUP_STATE_INVALID",
            f"{self.state_path} 的 setup state 无效: {detail}",
            hint="从 setup-backups 恢复 setup-state.json 后重试；不要覆盖损坏的 state",
        )

    def _validate_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise self._invalid_state("根节点必须是 object")
        if type(state.get("schema_version")) is not int or state.get(
            "schema_version"
        ) != STATE_SCHEMA_VERSION:
            raise self._invalid_state(
                f"schema_version 必须是整数 {STATE_SCHEMA_VERSION}"
            )
        for field in ("installed_at", "updated_at", "profile_id"):
            if not isinstance(state.get(field), str) or not state[field]:
                raise self._invalid_state(f"{field} 必须是非空 string")

        launcher = state.get("launcher")
        if not isinstance(launcher, dict):
            raise self._invalid_state("launcher 必须是 object")
        if not isinstance(launcher.get("command"), str) or not launcher["command"]:
            raise self._invalid_state("launcher.command 必须是非空 string")
        args = launcher.get("args")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise self._invalid_state("launcher.args 必须是 string array")

        embedding = state.get("embedding")
        if embedding is not None:
            if not isinstance(embedding, dict):
                raise self._invalid_state("embedding 必须是 object")
            expected_strings = (
                "type", "preset", "adapter", "base_url", "model", "version",
                "api_key_env",
            )
            if any(not isinstance(embedding.get(name), str) for name in expected_strings):
                raise self._invalid_state("embedding 字段类型无效")
            if embedding["type"] != "api":
                raise self._invalid_state("embedding.type 必须是 api")
            if embedding["preset"] not in {"ollama", "custom"}:
                raise self._invalid_state("embedding.preset 无效")
            if embedding["adapter"] not in {"ollama", "openai_compatible"}:
                raise self._invalid_state("embedding.adapter 无效")
            if type(embedding.get("dimension")) is not int or embedding["dimension"] < 1:
                raise self._invalid_state("embedding.dimension 必须是正整数")
            forbidden = {"api_key", "token", "credential", "authorization"}
            if forbidden.intersection(embedding):
                raise self._invalid_state("embedding 不得持久化明文凭据")

        adapter_states = state.get("adapters")
        if not isinstance(adapter_states, dict):
            raise self._invalid_state("adapters 必须是 object")
        known = {adapter.name for adapter in self.adapters}
        selected_adapters = state.get("selected_adapters")
        if selected_adapters is not None:
            if (
                not isinstance(selected_adapters, list)
                or not all(isinstance(name, str) for name in selected_adapters)
                or len(selected_adapters) != len(set(selected_adapters))
            ):
                raise self._invalid_state("selected_adapters 必须是无重复 string array")
            unknown_selected = set(selected_adapters) - known
            if unknown_selected:
                raise self._invalid_state(
                    "selected_adapters 包含未知 adapter: "
                    + ", ".join(sorted(unknown_selected))
                )
        for name, adapter_state in adapter_states.items():
            if name not in known:
                raise self._invalid_state(f"adapters.{name} 是未知 adapter")
            self._validate_adapter_state(name, adapter_state)

    def _validate_adapter_state(self, name: str, state: Any) -> None:
        field = f"adapters.{name}"
        if not isinstance(state, dict):
            raise self._invalid_state(f"{field} 必须是 object")
        if state.get("adapter") != name:
            raise self._invalid_state(f"{field}.adapter 必须是 {name!r}")
        if type(state.get("changed")) is not bool:
            raise self._invalid_state(f"{field}.changed 必须是 boolean")
        files = state.get("files")
        if not isinstance(files, list):
            raise self._invalid_state(f"{field}.files 必须是 array")
        for index, record in enumerate(files):
            self._validate_backup_record(f"{field}.files[{index}]", record)

        if name in {"claude", "cursor"}:
            self._validate_json_adapter_state(field, state)
        elif name == "codex":
            if not isinstance(state.get("previous_blocks"), str):
                raise self._invalid_state(f"{field}.previous_blocks 必须是 string")
            if not isinstance(state.get("managed_block"), str):
                raise self._invalid_state(f"{field}.managed_block 必须是 string")

        if name == "claude":
            if not isinstance(state.get("managed_hook"), dict):
                raise self._invalid_state(f"{field}.managed_hook 必须是 object")
            if type(state.get("hook_added")) is not bool:
                raise self._invalid_state(f"{field}.hook_added 必须是 boolean")

    def _validate_json_adapter_state(
        self, field: str, state: Mapping[str, Any]
    ) -> None:
        previous = state.get("previous")
        if not isinstance(previous, dict):
            raise self._invalid_state(f"{field}.previous 必须是 object")
        if type(previous.get("present")) is not bool:
            raise self._invalid_state(f"{field}.previous.present 必须是 boolean")
        managed = state.get("managed")
        if not isinstance(managed, dict):
            raise self._invalid_state(f"{field}.managed 必须是 object")
        if not isinstance(managed.get("command"), str) or not managed["command"]:
            raise self._invalid_state(f"{field}.managed.command 必须是非空 string")
        args = managed.get("args")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise self._invalid_state(f"{field}.managed.args 必须是 string array")
        environ = managed.get("env")
        if not isinstance(environ, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environ.items()
        ):
            raise self._invalid_state(f"{field}.managed.env 必须是 string object")

    def _validate_backup_record(self, field: str, record: Any) -> None:
        if not isinstance(record, dict):
            raise self._invalid_state(f"{field} 必须是 object")
        if not isinstance(record.get("target"), str) or not record["target"]:
            raise self._invalid_state(f"{field}.target 必须是非空 string")
        if type(record.get("existed")) is not bool:
            raise self._invalid_state(f"{field}.existed 必须是 boolean")
        for name in ("before_sha256", "backup"):
            value = record.get(name)
            if value is not None and not isinstance(value, str):
                raise self._invalid_state(f"{field}.{name} 必须是 string 或 null")
        if not isinstance(record.get("after_sha256"), str):
            raise self._invalid_state(f"{field}.after_sha256 必须是 string")

    def _write_state(self, state: Mapping[str, Any]) -> None:
        atomic_write(
            self.state_path,
            (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )

    def _transaction_snapshots(self, plan: SetupPlan) -> tuple[_FileSnapshot, ...]:
        paths = {self.state_path}
        paths.update(
            Path(target)
            for adapter in plan.adapters
            if adapter["selected"]
            for target in adapter["targets"]
        )
        return tuple(_FileSnapshot.capture(path) for path in sorted(paths, key=str))

    @staticmethod
    def _rollback_transaction(
        snapshots: Sequence[_FileSnapshot], backup_dir: Path
    ) -> list[str]:
        errors: list[str] = []
        for snapshot in reversed(snapshots):
            try:
                snapshot.restore_content()
            except Exception as exc:  # noqa: BLE001 -- 汇总所有回滚失败
                errors.append(f"{snapshot.path}: {exc}")
        if errors:
            return errors
        try:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            backup_root = backup_dir.parent
            if backup_root.exists():
                backup_root.rmdir()
        except OSError:
            # 旧安装的其他备份会让父目录非空；本次目录已删除即不影响回滚。
            pass
        for snapshot in reversed(snapshots):
            snapshot.remove_created_parents()
        return errors

    def _ensure_models(
        self, *, download: bool, progress: Progress | None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        required = (config.EMBED_MODEL,)
        if config.EMBED_ADAPTER != "ollama":
            # 通用 API 的模型生命周期由服务端管理；setup 不得调用
            # Ollama 的 tags/pull 端点。可用性由后续 embedding smoke 验证。
            return (
                [{
                    "name": name,
                    "status": "configured",
                    "provider": "api",
                    "adapter": config.EMBED_ADAPTER,
                } for name in required],
                [],
            )
        try:
            installed = _ollama_tags()
        except Exception as exc:  # noqa: BLE001 -- 网络/daemon 统一降级
            return (
                [
                    {"name": name, "status": "unavailable", "size": None}
                    for name in required
                ],
                [f"Ollama unavailable: {exc}"],
            )
        results: list[dict[str, Any]] = []
        degraded: list[str] = []
        for name in required:
            current = installed.get(name)
            if current:
                results.append(
                    {
                        "name": name,
                        "status": "cached",
                        "size_bytes": current.get("size"),
                        "size": _format_size(current.get("size")),
                    }
                )
                continue
            if not download:
                results.append({"name": name, "status": "skipped", "size": None})
                degraded.append(f"model missing: {name}")
                continue
            try:
                if progress:
                    progress({"phase": "model", "model": name, "status": "starting"})
                _pull_model(name, progress)
                results.append({"name": name, "status": "downloaded", "size": None})
            except Exception as exc:  # noqa: BLE001 -- 可离线安装
                results.append(
                    {"name": name, "status": "failed", "error": str(exc), "size": None}
                )
                degraded.append(f"model download failed ({name}): {exc}")
        return results, degraded

    @staticmethod
    def _schema_smoke() -> tuple[bool, str]:
        try:
            with sqlite3.connect(config.DB_PATH) as db:
                tables = {
                    row[0]
                    for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                    )
                }
            missing = {"sessions", "stories", "edges", "story_vectors"} - tables
            if missing:
                return False, f"missing tables: {', '.join(sorted(missing))}"
            return True, "schema ready"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def _smoke_tests(self, selected: set[str]) -> list[dict[str, Any]]:
        tests: list[dict[str, Any]] = []
        schema_ok, schema_detail = self._schema_smoke()
        tests.append({"name": "schema", "ok": schema_ok, "detail": schema_detail})

        vector: list[float] | None = None
        try:
            vector = embeddings.embed("storybook setup smoke test")
            embedding_ok = bool(vector) and len(vector) == config.EMBED_DIM
            detail = f"dimension={len(vector) if vector else 0}"
        except Exception as exc:  # noqa: BLE001
            embedding_ok = False
            detail = str(exc)
        tests.append({"name": "embedding", "ok": embedding_ok, "detail": detail})

        for adapter in self.adapters:
            if adapter.name not in selected:
                continue
            try:
                ok, detail = adapter.verify(self.context)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, str(exc)
            tests.append({"name": f"adapter:{adapter.name}", "ok": ok, "detail": detail})

        if selected:
            try:
                from . import mcp_server

                mcp_server.create_server()
                mcp_ok, mcp_detail = True, "FastMCP server ready"
            except Exception as exc:  # noqa: BLE001
                mcp_ok, mcp_detail = False, str(exc)
            tests.append({"name": "mcp-runtime", "ok": mcp_ok, "detail": mcp_detail})

        if embedding_ok:
            try:
                result = search.search(
                    "storybook setup smoke test", top_k=1, record_diagnostics=False
                )
                recall_ok = not result.get("error")
                recall_detail = f"mode={result.get('mode')}; matches={len(result.get('top_matches', []))}"
            except Exception as exc:  # noqa: BLE001
                recall_ok, recall_detail = False, str(exc)
        else:
            recall_ok, recall_detail = False, "skipped: embedding unavailable"
        tests.append({"name": "recall", "ok": recall_ok, "detail": recall_detail})
        return tests

    def _history_ingestion_status(self, selected: set[str]) -> list[dict[str, Any]]:
        """Report local history independently from MCP/config integration."""

        from .history_adapters.claude import ClaudeAdapter as ClaudeHistoryAdapter
        from .history_adapters.codex import CodexAdapter as CodexHistoryAdapter
        from .history_adapters.cursor import CursorAdapter as CursorHistoryAdapter

        codex_root = self.context.environ.get("CODEX_HOME", "").strip()
        probes = {
            "claude": ClaudeHistoryAdapter(self.context.home / ".claude" / "projects"),
            "cursor": CursorHistoryAdapter(
                self.context.home / "Library" / "Application Support" / "Cursor"
                / "User" / "workspaceStorage"
            ),
            "codex": CodexHistoryAdapter(
                Path(codex_root).expanduser() if codex_root else self.context.home / ".codex"
            ),
        }
        result = []
        for name in sorted(selected):
            probe = probes.get(name)
            if probe is None:
                continue
            detection = probe.detect()
            result.append({
                "name": name,
                "available": bool(detection.get("available")),
                "status": detection.get("status", "unknown"),
                "adapter_version": probe.version,
            })
        return result

    def execute(
        self,
        *,
        requested_agents: Iterable[str] | None = None,
        download_models: bool = True,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        plan = self.plan(requested_agents)
        selected = {
            item["adapter"] for item in plan.adapters if item["selected"]
        }
        # 受管 state 决定升级与卸载恢复行为，必须在任何 Profile/DB 写入前校验。
        existing_state = self._load_state() or {}
        try:
            config.refresh_profile(create=True)
            store.init_db()
        except Exception as exc:  # noqa: BLE001
            raise SetupError(
                "SB_SETUP_SCHEMA_FAILED",
                f"无法初始化用户级数据库: {exc}",
                hint="检查目录权限与 sqlite-vec 安装后重试",
            ) from exc

        adapter_states = dict(existing_state.get("adapters", {}))
        backup_dir = self.roots.state / "setup-backups" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        snapshots = self._transaction_snapshots(plan)
        adapter_results: list[dict[str, Any]] = []
        planned_changes = {
            item["adapter"]: bool(item["changed"]) for item in plan.adapters
        }
        phase = "config"
        try:
            for adapter in self.adapters:
                if adapter.name not in selected:
                    continue
                prior_state = adapter_states.get(adapter.name)
                if prior_state and planned_changes.get(adapter.name):
                    restored = adapter.uninstall(self.context, prior_state)
                    if restored.get("status") == "drifted":
                        raise AdapterError(
                            "SB_SETUP_CONFIG_DRIFT",
                            f"{adapter.display_name} 的 Storybook 节点已被修改，拒绝覆盖",
                        )
                state = adapter.apply(self.context, backup_dir)
                adapter_results.append(
                    {"name": adapter.name, "changed": bool(state.get("changed"))}
                )
                if state.get("changed"):
                    adapter_states[adapter.name] = state
            state = {
                "schema_version": STATE_SCHEMA_VERSION,
                "installed_at": existing_state.get("installed_at", _utc_now()),
                "updated_at": _utc_now(),
                "profile_id": config.PROFILE_ID,
                # A partial re-run does not uninstall adapters managed by an
                # earlier setup, so keep checking the full managed union.
                "selected_adapters": sorted(
                    set(existing_state.get("selected_adapters", ()))
                    | set(adapter_states)
                    | selected
                ),
                "launcher": {
                    "command": self.launcher.command,
                    "args": list(self.launcher.args),
                },
                "embedding": {
                    "type": config.EMBED_TYPE,
                    "preset": config.EMBED_PRESET,
                    "adapter": config.EMBED_ADAPTER,
                    "base_url": config.EMBED_BASE_URL,
                    "model": config.EMBED_MODEL,
                    "dimension": config.EMBED_DIM,
                    "version": config.EMBED_VERSION,
                    "api_key_env": config.EMBED_API_KEY_ENV,
                },
                "adapters": adapter_states,
            }
            phase = "state"
            self._write_state(state)
        except Exception as exc:
            rollback_errors = self._rollback_transaction(snapshots, backup_dir)
            if rollback_errors:
                raise SetupError(
                    "SB_SETUP_ROLLBACK_FAILED",
                    f"setup 失败且无法完整回滚: {'; '.join(rollback_errors)}",
                    hint="从 setup-backups 恢复配置并检查磁盘空间或目录权限",
                ) from exc
            if phase == "state":
                raise SetupError(
                    "SB_SETUP_STATE_WRITE_FAILED",
                    f"无法持久化 setup state: {exc}",
                    hint="配置与旧 setup state 已回滚；检查磁盘空间和 state 目录权限后重试",
                ) from exc
            code = exc.code if isinstance(exc, AdapterError) else "SB_SETUP_CONFIG_WRITE_FAILED"
            raise SetupError(code, str(exc), hint="配置与旧 setup state 已回滚；修复后重试") from exc

        models, degraded = self._ensure_models(
            download=download_models, progress=progress
        )
        smoke = self._smoke_tests(selected)
        failed_smoke = [test["name"] for test in smoke if not test["ok"]]
        degraded.extend(f"smoke failed: {name}" for name in failed_smoke)
        status = "ready" if not degraded else "degraded"
        return {
            "status": status,
            "profile": {
                "id": config.PROFILE_ID,
                "data_dir": str(config.DATA_DIR),
                "database": str(config.DB_PATH),
                "sync_state": config.SYNC_STATE,
            },
            "adapters": adapter_results,
            "history_ingestion": self._history_ingestion_status(selected),
            "models": models,
            "smoke_tests": smoke,
            "legacy_databases": list(plan.legacy_databases),
            "degraded_reasons": degraded,
            "state_file": str(self.state_path),
        }

    def runtime_status(self) -> dict[str, Any]:
        """Return a stable, live snapshot of Profile, model and adapter readiness.

        The setup state records which adapters Storybook is expected to manage,
        while model and adapter readiness are probed live so recovery does not
        require running setup again.  Degraded reasons are stable machine-readable
        codes; volatile exception text is deliberately excluded from the contract.
        """

        degraded_reasons: list[str] = []
        profile = config.PROFILE_REGISTRY.peek_active_profile()
        if profile is None:
            profile_payload: dict[str, Any] = {
                "status": "missing",
                "id": None,
                "display_name": None,
                "mode": None,
            }
            sync_state = config.SYNC_STATE
            degraded_reasons.append("profile_uninitialized")
        else:
            profile_payload = {
                "status": "ready",
                "id": profile.id,
                "display_name": profile.display_name,
                "mode": profile.mode,
            }
            sync_state = profile.sync_state

        llm_ready = bool(config.LLM_API_KEY)
        llm_payload = {
            "provider": config.LLM_PROVIDER,
            "name": config.LLM_MODEL,
            "status": "ready" if llm_ready else "credentials_missing",
        }
        if not llm_ready:
            degraded_reasons.append("llm_credentials_missing")

        actual_dimension = None
        if config.EMBED_ADAPTER == "ollama":
            endpoint_ok, tags, _ = health._check_ollama_reachable()
            embedding_ready = endpoint_ok and health._model_pulled(
                tags, config.EMBED_MODEL
            )
            if not endpoint_ok:
                embedding_status = "unavailable"
                degraded_reasons.append("endpoint_unreachable:embedding")
            elif not embedding_ready:
                embedding_status = "missing"
                degraded_reasons.append("model_unavailable:embedding")
            else:
                probe = embeddings.probe()
                actual_dimension = int(probe["dimension"])
                embedding_ready = bool(probe["ok"])
                reason = str(probe["reason"] or "")
                embedding_status = "ready" if embedding_ready else reason
                if not embedding_ready:
                    degraded_reasons.append(f"{reason}:embedding")
        else:
            probe = embeddings.probe()
            actual_dimension = int(probe["dimension"])
            embedding_ready = bool(probe["ok"])
            reason = str(probe["reason"] or "")
            embedding_status = "ready" if embedding_ready else reason
            if not embedding_ready:
                degraded_reasons.append(f"{reason}:embedding")

        compatibility = health.serving_index_compatibility(actual_dimension)
        if embedding_ready and not compatibility["ok"]:
            embedding_ready = False
            embedding_status = "serving_index_mismatch"
            degraded_reasons.append("serving_index_mismatch:embedding")

        model_payload = {
            "provider": "hybrid",
            "status": "ready" if llm_ready and embedding_ready else "degraded",
            "llm": llm_payload,
            "embedding": {
                "type": config.EMBED_TYPE,
                "provider": "api",
                "preset": config.EMBED_PRESET,
                "adapter": config.EMBED_ADAPTER,
                "base_url": config.EMBED_BASE_URL,
                "name": config.EMBED_MODEL,
                "dimension": config.EMBED_DIM,
                "actual_dimension": actual_dimension,
                "serving_dimension": compatibility["serving_dimension"],
                "active_model": compatibility["active_model"],
                "active_version": compatibility["active_version"],
                "version": config.EMBED_VERSION,
                "config_source": config.EMBED_CONFIG_SOURCE,
                "config_normalized": config.EMBED_CONFIG_NORMALIZED,
                "remote_text_disclosure": config.embedding_text_leaves_device(),
                "model_state": (
                    embeddings.model_state()
                    if config.EMBED_ADAPTER == "ollama"
                    else None
                ),
                "status": embedding_status,
            },
        }

        try:
            state = self._load_state()
        except SetupError:
            # Status must remain the stable diagnostic entry point even when
            # setup-state.json itself needs recovery.
            state = None
            degraded_reasons.append("setup_state_invalid")
        adapter_by_name = {adapter.name: adapter for adapter in self.adapters}
        if state is None:
            selected_names: list[str] = []
        else:
            selected_names = list(
                state.get("selected_adapters", state.get("adapters", {}).keys())
            )
        adapter_checks: list[dict[str, str]] = []
        for name in sorted(selected_names):
            adapter = adapter_by_name[name]
            try:
                ready, _ = adapter.verify(self.context)
            except Exception:  # noqa: BLE001 -- invalid/missing config is degraded
                ready = False
            adapter_checks.append(
                {"name": name, "status": "ready" if ready else "missing"}
            )
            if not ready:
                degraded_reasons.append(f"adapter_unavailable:{name}")

        adapter_payload = {
            "status": (
                "not_configured"
                if not adapter_checks
                else (
                    "ready"
                    if all(item["status"] == "ready" for item in adapter_checks)
                    else "degraded"
                )
            ),
            "checks": adapter_checks,
        }
        stable_reasons = list(dict.fromkeys(degraded_reasons))
        return {
            "status": "ready_degraded" if stable_reasons else "ready",
            "profile": profile_payload,
            "model": model_payload,
            "adapter": adapter_payload,
            "sync": {
                "state": sync_state,
                "enabled": sync_state != "local_only",
            },
            "sync_state": sync_state,
            "degraded_reasons": stable_reasons,
        }

    @staticmethod
    def _safe_purge(paths: Iterable[Path], home: Path) -> list[str]:
        removed: list[str] = []
        seen: set[Path] = set()
        for raw in paths:
            path = raw.resolve(strict=False)
            if path in seen or not path.exists():
                continue
            seen.add(path)
            if path in {Path("/"), home.resolve(strict=False)} or len(path.parts) < 3:
                raise SetupError(
                    "SB_UNINSTALL_PURGE_UNSAFE", f"拒绝删除过宽路径: {path}"
                )
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
        return removed

    def uninstall(self, *, purge_data: bool = False) -> dict[str, Any]:
        state = self._load_state()
        if state is None:
            removed = (
                self._safe_purge(
                    (
                        self.roots.data,
                        self.roots.cache,
                        self.roots.logs,
                        self.roots.config,
                        self.roots.state,
                    ),
                    self.home,
                )
                if purge_data
                else []
            )
            return {
                "status": "already_uninstalled",
                "data": "purged" if purge_data else "kept",
                "adapters": [],
                "removed": removed,
            }
        adapter_state = state.get("adapters", {})
        results: list[dict[str, Any]] = []
        drifted = False
        for adapter in self.adapters:
            saved = adapter_state.get(adapter.name)
            if not saved:
                continue
            try:
                result = adapter.uninstall(self.context, saved)
            except AdapterError as exc:
                raise SetupError(exc.code, str(exc)) from exc
            results.append(result)
            drifted = drifted or result.get("status") == "drifted"

        removed: list[str] = []
        if purge_data:
            removed = self._safe_purge(
                (
                    self.roots.data,
                    self.roots.cache,
                    self.roots.logs,
                    self.roots.config,
                    self.roots.state,
                ),
                self.home,
            )
        elif not drifted and self.state_path.exists():
            self.state_path.unlink()

        return {
            "status": "degraded" if drifted else "uninstalled",
            "data": "purged" if purge_data else "kept",
            "adapters": results,
            "removed": removed,
            "degraded_reasons": (
                ["adapter config drift detected; setup state retained"] if drifted else []
            ),
        }
