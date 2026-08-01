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

from . import config, embeddings, search, store
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
    response = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
    response.raise_for_status()
    payload = response.json()
    return {
        str(item.get("name")): item
        for item in payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }


def _pull_model(model: str, progress: Progress | None = None) -> None:
    with requests.post(
        f"{config.OLLAMA_HOST}/api/pull",
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
    adapters: tuple[dict[str, Any], ...]
    models: tuple[str, ...]
    legacy_databases: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
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
            adapters=tuple(adapter_plans),
            models=(config.LLM_MODEL, config.EMBED_MODEL),
            legacy_databases=self._legacy_databases(),
        )

    def _load_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SetupError(
                "SB_SETUP_STATE_INVALID", f"无法读取 {self.state_path}: {exc}"
            ) from exc
        if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise SetupError("SB_SETUP_STATE_INVALID", "setup state schema 无效")
        return state

    def _write_state(self, state: Mapping[str, Any]) -> None:
        atomic_write(
            self.state_path,
            (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )

    def _ensure_models(
        self, *, download: bool, progress: Progress | None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        required = (config.LLM_MODEL, config.EMBED_MODEL)
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
        try:
            config.refresh_profile(create=True)
            store.init_db()
        except Exception as exc:  # noqa: BLE001
            raise SetupError(
                "SB_SETUP_SCHEMA_FAILED",
                f"无法初始化用户级数据库: {exc}",
                hint="检查目录权限与 sqlite-vec 安装后重试",
            ) from exc

        existing_state = self._load_state() or {}
        adapter_states = dict(existing_state.get("adapters", {}))
        backup_dir = self.roots.state / "setup-backups" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        applied: list[tuple[AgentAdapter, dict[str, Any]]] = []
        retired_prior: set[str] = set()
        adapter_results: list[dict[str, Any]] = []
        planned_changes = {
            item["adapter"]: bool(item["changed"]) for item in plan.adapters
        }
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
                    retired_prior.add(adapter.name)
                state = adapter.apply(self.context, backup_dir)
                adapter_results.append(
                    {"name": adapter.name, "changed": bool(state.get("changed"))}
                )
                if state.get("changed"):
                    adapter_states[adapter.name] = state
                    applied.append((adapter, state))
        except Exception as exc:
            for adapter, state in reversed(applied):
                try:
                    adapter.uninstall(self.context, state)
                except Exception:
                    pass
            if retired_prior and existing_state:
                recovery = dict(existing_state)
                recovery["updated_at"] = _utc_now()
                recovery["adapters"] = {
                    name: saved
                    for name, saved in existing_state.get("adapters", {}).items()
                    if name not in retired_prior
                }
                self._write_state(recovery)
            code = exc.code if isinstance(exc, AdapterError) else "SB_SETUP_CONFIG_WRITE_FAILED"
            raise SetupError(code, str(exc), hint="配置已回滚；修复后重试") from exc

        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "installed_at": existing_state.get("installed_at", _utc_now()),
            "updated_at": _utc_now(),
            "profile_id": config.PROFILE_ID,
            "launcher": {"command": self.launcher.command, "args": list(self.launcher.args)},
            "adapters": adapter_states,
        }
        self._write_state(state)

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
            "models": models,
            "smoke_tests": smoke,
            "legacy_databases": list(plan.legacy_databases),
            "degraded_reasons": degraded,
            "state_file": str(self.state_path),
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
