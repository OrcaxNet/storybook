"""用户级 Profile registry 与平台数据目录解析。

Profile 是 Storybook 的本地记忆所有者。registry 只保存随机 UUID、显示名和
模式等可移植元数据；绝对路径始终按当前平台推导，不会成为跨设备对象主键。
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator, Mapping

try:  # Unix/macOS
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - Unix/macOS
    msvcrt = None


REGISTRY_SCHEMA_VERSION = 1
PROFILE_MODES = frozenset({"local", "isolated"})
SYNC_STATES = frozenset(
    {"local_only", "synced", "pending", "conflict", "paused", "error"}
)
DEFAULT_SYNC_STATE = "local_only"
DEFAULT_DATABASE_REF = "db/memory.db"


class ProfileError(RuntimeError):
    """Profile registry 无法读取、校验或更新。"""


@dataclass(frozen=True)
class PlatformRoots:
    """当前 OS 用户的规范配置、数据、缓存和状态目录。"""

    config: Path
    data: Path
    cache: Path
    state: Path
    logs: Path


@dataclass(frozen=True)
class Profile:
    """registry 中持久化的 Profile 元数据。"""

    id: str
    display_name: str
    mode: str
    sync_state: str
    created_at: str
    # Profile-local relative pointer.  Migrations switch this value atomically
    # instead of replacing an SQLite file which another process may still have
    # open.  Absolute paths never enter the portable registry.
    database_ref: str = DEFAULT_DATABASE_REF


@dataclass(frozen=True)
class ProfilePaths:
    """由 Profile UUID 与当前平台目录推导出的本地路径。"""

    root: Path
    database_dir: Path
    database: Path
    index_dir: Path
    cache_dir: Path
    log_dir: Path


def _env_path(environ: Mapping[str, str], key: str) -> Path | None:
    value = environ.get(key, "").strip()
    return Path(value).expanduser() if value else None


def platform_roots(
    platform_name: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> PlatformRoots:
    """解析 macOS、Linux、Windows 的用户级目录。

    ``STORYBOOK_HOME`` 可把所有运行态目录收拢到一个自定义根目录，适合测试、
    便携安装或显式隔离。也可分别用 ``STORYBOOK_{CONFIG,DATA,CACHE,STATE}_HOME``
    覆盖单个目录。
    """

    env = os.environ if environ is None else environ
    user_home = Path.home() if home is None else Path(home)
    system = (platform_name or sys.platform).lower()

    common_home = _env_path(env, "STORYBOOK_HOME")
    if common_home is not None:
        config_root = common_home / "config"
        data_root = common_home / "data"
        cache_root = common_home / "cache"
        state_root = common_home / "state"
        log_root = common_home / "logs"
    elif system == "darwin":
        application_support = (
            user_home / "Library" / "Application Support" / "Storybook"
        )
        config_root = application_support
        data_root = application_support
        cache_root = user_home / "Library" / "Caches" / "Storybook"
        state_root = application_support / "state"
        log_root = user_home / "Library" / "Logs" / "Storybook"
    elif system.startswith("win"):
        local_app_data = _env_path(env, "LOCALAPPDATA")
        if local_app_data is None:
            local_app_data = user_home / "AppData" / "Local"
        base = local_app_data / "Storybook"
        config_root = base / "config"
        data_root = base
        cache_root = base / "cache"
        state_root = base / "state"
        log_root = base / "logs"
    else:
        config_root = (
            _env_path(env, "XDG_CONFIG_HOME") or user_home / ".config"
        ) / "storybook"
        data_root = (
            _env_path(env, "XDG_DATA_HOME") or user_home / ".local" / "share"
        ) / "storybook"
        cache_root = (
            _env_path(env, "XDG_CACHE_HOME") or user_home / ".cache"
        ) / "storybook"
        state_root = (
            _env_path(env, "XDG_STATE_HOME") or user_home / ".local" / "state"
        ) / "storybook"
        log_root = state_root / "logs"

    config_override = _env_path(env, "STORYBOOK_CONFIG_HOME")
    data_override = _env_path(env, "STORYBOOK_DATA_HOME")
    cache_override = _env_path(env, "STORYBOOK_CACHE_HOME")
    state_override = _env_path(env, "STORYBOOK_STATE_HOME")
    config_root = config_override or config_root
    data_root = data_override or data_root
    cache_root = cache_override or cache_root
    state_root = state_override or state_root
    if state_override is not None:
        log_root = state_root / "logs"

    return PlatformRoots(
        config=config_root,
        data=data_root,
        cache=cache_root,
        state=state_root,
        logs=log_root,
    )


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)


def _private_file(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o600)


class ProfileRegistry:
    """原子管理同一 OS 用户的 Profile registry。"""

    def __init__(
        self,
        registry_path: Path | None = None,
        *,
        roots: PlatformRoots | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.environ = os.environ if environ is None else environ
        self.roots = roots or platform_roots(environ=self.environ)
        configured = _env_path(self.environ, "STORYBOOK_REGISTRY")
        self.path = Path(
            registry_path or configured or (self.roots.config / "profiles.json")
        )
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        _private_dir(self.path.parent)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _private_file(self.lock_path)
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)

    @staticmethod
    def _new_profile(display_name: str, mode: str) -> Profile:
        name = display_name.strip()
        if not name:
            raise ProfileError("Profile display_name 不能为空")
        if mode not in PROFILE_MODES:
            raise ProfileError(
                f"不支持的 Profile mode: {mode!r}；可选 {sorted(PROFILE_MODES)}"
            )
        return Profile(
            id=str(uuid.uuid4()),
            display_name=name,
            mode=mode,
            sync_state=DEFAULT_SYNC_STATE,
            created_at=datetime.now(timezone.utc).isoformat(),
            database_ref=DEFAULT_DATABASE_REF,
        )

    @staticmethod
    def _validate_database_ref(value: object) -> str:
        ref = str(value or DEFAULT_DATABASE_REF).replace("\\", "/")
        path = PurePosixPath(ref)
        if (
            not ref
            or path.is_absolute()
            or ref.startswith("//")
            or (path.parts and path.parts[0].endswith(":"))
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix != ".db"
        ):
            raise ProfileError(
                "Profile database_ref 必须是 Profile 内的安全相对 .db 路径"
            )
        return path.as_posix()

    @staticmethod
    def _validate_profile(raw: object) -> Profile:
        if not isinstance(raw, dict):
            raise ProfileError("Profile 记录必须是 JSON object")
        try:
            profile = Profile(
                id=str(raw["id"]),
                display_name=str(raw["display_name"]),
                mode=str(raw["mode"]),
                sync_state=str(raw["sync_state"]),
                created_at=str(raw["created_at"]),
                database_ref=ProfileRegistry._validate_database_ref(
                    raw.get("database_ref", DEFAULT_DATABASE_REF)
                ),
            )
            uuid.UUID(profile.id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError(f"Profile 记录无效: {exc}") from exc
        if not profile.display_name.strip():
            raise ProfileError("Profile display_name 不能为空")
        if profile.mode not in PROFILE_MODES:
            raise ProfileError(f"Profile mode 无效: {profile.mode!r}")
        if profile.sync_state not in SYNC_STATES:
            raise ProfileError(f"Profile sync_state 无效: {profile.sync_state!r}")
        return profile

    def _read_unlocked(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"无法读取 Profile registry {self.path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ProfileError("Profile registry 根节点必须是 JSON object")
        if raw.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ProfileError(
                f"不支持的 Profile registry schema_version: "
                f"{raw.get('schema_version')!r}"
            )
        profiles = [self._validate_profile(item) for item in raw.get("profiles", [])]
        if not profiles:
            raise ProfileError("Profile registry 至少需要一个 Profile")
        ids = [profile.id for profile in profiles]
        names = [profile.display_name.casefold() for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ProfileError("Profile registry 含重复 UUID")
        if len(names) != len(set(names)):
            raise ProfileError("Profile registry 含重复 display_name")
        if raw.get("active_profile_id") not in ids:
            raise ProfileError("active_profile_id 未指向有效 Profile")
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "active_profile_id": raw["active_profile_id"],
            "profiles": profiles,
        }

    def _write_unlocked(self, state: dict) -> None:
        payload = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "active_profile_id": state["active_profile_id"],
            "profiles": [asdict(profile) for profile in state["profiles"]],
        }
        _private_dir(self.path.parent)
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            _private_file(self.path)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(
                    self.path.parent, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if tmp.exists():
                tmp.unlink()

    def ensure(self) -> dict:
        """确保 registry 和默认 local Profile 存在并返回当前快照。"""

        with self._locked():
            state = self._read_unlocked()
            if state is None:
                profile = self._new_profile("default", "local")
                state = {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "active_profile_id": profile.id,
                    "profiles": [profile],
                }
                self._write_unlocked(state)
        self.ensure_profile_directories(
            self.profile_by_id(state["active_profile_id"], state)
        )
        return state

    def list_profiles(self) -> list[Profile]:
        return list(self.ensure()["profiles"])

    def profile_by_id(self, profile_id: str, state: dict | None = None) -> Profile:
        snapshot = state or self.ensure()
        for profile in snapshot["profiles"]:
            if profile.id == profile_id:
                return profile
        raise ProfileError(f"Profile 不存在: {profile_id}")

    def resolve(self, ref: str, state: dict | None = None) -> Profile:
        snapshot = state or self.ensure()
        needle = ref.strip()
        for profile in snapshot["profiles"]:
            if (
                profile.id == needle
                or profile.display_name.casefold() == needle.casefold()
            ):
                return profile
        raise ProfileError(f"Profile 不存在: {ref}")

    def active_profile(self) -> Profile:
        state = self.ensure()
        override = self.environ.get("STORYBOOK_PROFILE", "").strip()
        if override:
            return self.resolve(override, state)
        return self.profile_by_id(state["active_profile_id"], state)

    def peek_active_profile(self) -> Profile | None:
        """只读返回当前 Profile；registry 不存在时不创建任何文件。

        ``storybook setup --dry-run`` 必须做到零写入，因此配置模块导入阶段不能
        隐式调用 :meth:`ensure`。普通命令仍通过 :meth:`active_profile` 保持原有的
        首次使用自动初始化行为。
        """

        state = self._read_unlocked()
        if state is None:
            return None
        override = self.environ.get("STORYBOOK_PROFILE", "").strip()
        if override:
            return self.resolve(override, state)
        return self.profile_by_id(state["active_profile_id"], state)

    def create_profile(
        self,
        display_name: str,
        *,
        mode: str = "isolated",
        activate: bool = False,
    ) -> Profile:
        with self._locked():
            state = self._read_unlocked()
            if state is None:
                default = self._new_profile("default", "local")
                state = {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "active_profile_id": default.id,
                    "profiles": [default],
                }
            if any(
                p.display_name.casefold() == display_name.strip().casefold()
                for p in state["profiles"]
            ):
                raise ProfileError(f"Profile display_name 已存在: {display_name}")
            profile = self._new_profile(display_name, mode)
            state["profiles"].append(profile)
            if activate:
                state["active_profile_id"] = profile.id
            self._write_unlocked(state)
        self.ensure_profile_directories(profile)
        return profile

    def switch_profile(self, ref: str) -> Profile:
        with self._locked():
            state = self._read_unlocked()
            if state is None:
                raise ProfileError("Profile registry 尚未初始化")
            profile = self.resolve(ref, state)
            state["active_profile_id"] = profile.id
            self._write_unlocked(state)
        self.ensure_profile_directories(profile)
        return profile

    def set_profile_database(self, ref: str, database_ref: str) -> Profile:
        """Atomically point one Profile at a managed database generation."""

        validated = self._validate_database_ref(database_ref)
        with self._locked():
            state = self._read_unlocked()
            if state is None:
                raise ProfileError("Profile registry 尚未初始化")
            selected = self.resolve(ref, state)
            updated = replace(selected, database_ref=validated)
            state["profiles"] = [
                updated if profile.id == selected.id else profile
                for profile in state["profiles"]
            ]
            self._write_unlocked(state)
        self.ensure_profile_directories(updated)
        return updated

    def paths_for(self, profile: Profile) -> ProfilePaths:
        root = self.roots.data / "profiles" / profile.id
        database = root.joinpath(*PurePosixPath(profile.database_ref).parts)
        return ProfilePaths(
            root=root,
            database_dir=database.parent,
            database=database,
            index_dir=root / "indexes",
            cache_dir=self.roots.cache / "profiles" / profile.id,
            log_dir=self.roots.logs / "profiles" / profile.id,
        )

    def ensure_profile_directories(self, profile: Profile) -> ProfilePaths:
        paths = self.paths_for(profile)
        for directory in (
            paths.root,
            paths.database_dir,
            paths.index_dir,
            paths.cache_dir,
            paths.log_dir,
        ):
            _private_dir(directory)
        return paths

    def active_paths(self) -> ProfilePaths:
        profile = self.active_profile()
        return self.ensure_profile_directories(profile)


def default_registry() -> ProfileRegistry:
    """返回按当前 OS 与环境变量配置的 registry。"""

    return ProfileRegistry()
