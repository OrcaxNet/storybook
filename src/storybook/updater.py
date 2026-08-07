"""``book update``：检测最新正式版本并安全升级。

设计要点
--------
- 版本检测复用官方发布机制：默认 ``STORYBOOK_INSTALL_REPOSITORY``（与 ``install.sh``
  一致）指向 GitHub 仓库，检测端点 ``{repository}/releases/latest`` 遵循 GitHub
  Releases 的“重定向到最新 tag 页”约定；镜像/测试只需实现同样的重定向即可。
- 下载、sha256 校验、原子安装（backup + swap + rollback）**完整委托**官方
  ``install.sh``（随包分发在 ``storybook/data/install.sh``），避免双份安装逻辑漂移；
  本模块只做检测、版本比较、确认与结果呈现。
- 错误码沿用 ``SB_*`` 约定：本模块自有错误 ``SB_UPDATE_*``；``install.sh`` 的
  ``SB_INSTALL_*`` 错误原样透传给上层。
- 本模块不触碰 Profile / 数据库数据；只与安装 prefix 交互。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import requests

DEFAULT_REPOSITORY = "https://github.com/OrcaxNet/storybook"
BUNDLED_INSTALLER = Path(__file__).resolve().parent / "data" / "install.sh"

_VERSION_RE = re.compile(
    r"^v?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?(?:[-+].*)?$"
)
_TAG_URL_RE = re.compile(r"/releases/tag/([^/?#\"']+)")
_SB_CODE_RE = re.compile(r"\[(SB_[A-Z0-9_]+)\]")


class UpdateError(Exception):
    """带稳定错误码（``SB_*``）的更新失败。"""

    def __init__(self, code: str, message: str, detail: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


def parse_version(version: str) -> tuple[int, int, int] | None:
    """把 ``v0.1.4`` / ``0.1.4`` 解析为 (major, minor, patch)；无法解析返回 None。"""
    match = _VERSION_RE.match(version.strip())
    if not match:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor") or 0),
        int(match.group("patch") or 0),
    )


def is_update_available(current: str, latest: str) -> bool:
    """``latest`` 严格大于已装版本时返回 True（已是最新则 False）。"""
    current_parsed = parse_version(current)
    latest_parsed = parse_version(latest)
    if current_parsed is None or latest_parsed is None:
        raise UpdateError(
            "SB_UPDATE_VERSION_INVALID",
            f"无法比较版本: current={current!r}, latest={latest!r}",
        )
    return latest_parsed > current_parsed


def validate_release_url(url: str) -> None:
    """与 ``install.sh`` 相同的 URL 安全校验：https 或 http-loopback，无凭据/query/fragment。"""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    safe_scheme = parsed.scheme == "https"
    safe_loopback = parsed.scheme == "http" and host in {"127.0.0.1", "localhost"}
    if (
        not url
        or any(ord(character) < 33 or ord(character) == 127 for character in url)
        or not parsed.netloc
        or not host
        or not (safe_scheme or safe_loopback)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise UpdateError(
            "SB_UPDATE_URL_UNSAFE",
            "检测/下载 URL 必须使用 HTTPS（或本机 loopback）且不含凭据、query 或 fragment",
        )


def repository() -> str:
    return os.environ.get("STORYBOOK_INSTALL_REPOSITORY", DEFAULT_REPOSITORY).rstrip("/")


def detection_url() -> str:
    return f"{repository()}/releases/latest"


def detect_latest_version(timeout: float = 30.0) -> str:
    """返回最新正式版本号（去 ``v`` 前缀），例如 ``0.1.4``。"""
    url = detection_url()
    validate_release_url(url)
    try:
        response = requests.get(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise UpdateError(
            "SB_UPDATE_LATEST_UNREACHABLE",
            f"无法获取最新版本信息: {exc}",
            detail=str(exc),
        ) from exc

    tag = _extract_tag(response)
    if tag is None:
        raise UpdateError(
            "SB_UPDATE_LATEST_INVALID",
            f"无法从 {response.url} 解析最新版本标签",
        )
    version = tag[1:] if tag.startswith("v") else tag
    if parse_version(version) is None:
        raise UpdateError(
            "SB_UPDATE_VERSION_INVALID",
            f"无法解析最新版本号: {tag!r}",
        )
    return version


def _extract_tag(response: requests.Response) -> str | None:
    match = _TAG_URL_RE.search(response.url)
    if match:
        return match.group(1)
    # 兜底：某些镜像直接返回 tag 页内容而非 302 重定向
    match = _TAG_URL_RE.search(response.text)
    if match:
        return match.group(1)
    return None


def infer_prefix() -> Path | None:
    """从当前安装布局推断 prefix（``<prefix>/lib/storybook/...``）。

    标准安装（``install.sh``）布局为
    ``<prefix>/lib/storybook/{current,releases}/...``；同时考虑包路径、入口脚本与
    ``sys.executable``。dev checkout（未按该布局安装）时返回 None，由上层回退到
    ``$HOME/.local``。
    """
    candidates: list[Path] = [Path(__file__).resolve().parent]
    for source in (sys.argv[0], sys.executable):
        try:
            candidates.append(Path(source).resolve())
        except (OSError, RuntimeError):
            continue
    seen: set[Path] = set()
    for start in candidates:
        for ancestor in (start, *start.parents):
            if ancestor in seen:
                continue
            seen.add(ancestor)
            if (ancestor / "lib" / "storybook").is_dir():
                return ancestor
    return None


def resolve_installer(flag: Path | None = None) -> Path:
    """定位 install.sh：``--installer`` > ``STORYBOOK_INSTALL_SCRIPT`` > 随包副本。"""
    if flag is not None:
        path = Path(flag)
    else:
        override = os.environ.get("STORYBOOK_INSTALL_SCRIPT")
        path = Path(override) if override else BUNDLED_INSTALLER
    if not path.is_file():
        raise UpdateError("SB_UPDATE_INSTALLER_MISSING", f"安装脚本不存在: {path}")
    return path


def run_installer(
    installer: Path,
    version: str,
    prefix: Path,
    timeout: float | None = 1800.0,
) -> subprocess.CompletedProcess[str]:
    """以官方 ``install.sh`` 执行下载 → 校验 → 原子安装。

    强制 ``STORYBOOK_INSTALL_PYTHON=sys.executable``，保证新 release 与当前安装使用
    同一 Python 家族；``STORYBOOK_INSTALL_ARCHIVE_URL``/``CHECKSUM_URL``（若已设置）
    原样透传，与 ``install.sh`` 的镜像约定一致。
    """
    env = dict(os.environ)
    env["STORYBOOK_INSTALL_PYTHON"] = sys.executable
    command = [
        "sh",
        str(installer),
        "--version",
        version,
        "--prefix",
        str(prefix),
        "--no-init",
    ]
    try:
        return subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            "SB_UPDATE_INSTALL_TIMEOUT",
            "安装超时；现有安装不受影响",
            detail=str(exc),
        ) from exc


def extract_sb_code(stderr: str) -> str | None:
    """从 install.sh 的 stderr 提取稳定错误码（``[SB_*]``）。"""
    match = _SB_CODE_RE.search(stderr)
    return match.group(1) if match else None


def last_stderr_line(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else ""
