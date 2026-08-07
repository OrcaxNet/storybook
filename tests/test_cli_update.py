"""FLO-209 功能二：``book update`` 子命令测试。

覆盖验收标准（PRD 第 4 节）：
1. ``book update --help`` 正常。
2. 已是最新版本：不下载、输出 up-to-date、退出 0。
3. 存在新版：下载并校验 sha256 通过后安装；安装后新版本生效。
4. ``--dry-run``：零写入，输出含当前版本、最新版本、目标 prefix。
5. 下载失败：稳定错误、退出非 0、无部分安装、现有安装不受影响。
6. checksum 不匹配：中止安装、退出非 0、现有安装不受影响。
7. 更新不修改 Profile 数据 / 数据库（隔离 HOME 验证）。
8. 本文件即为 ``tests/test_cli_update.py``。

测试手法参考 ``tests/test_installer.py`` 的 fake release server / fake archive：
- ``ReleaseServer`` 实现 ``{base}/releases/latest`` 重定向 + 静态 release 文件下载；
- 安装路径用真实 ``install.sh``（随包副本），完整验证下载 → 校验 → 原子安装语义；
- 升级场景用 probe wheel（``book``/``storybook`` 入口打印固定 tag），避免引入真实依赖。
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from storybook import __version__, cli


ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════
#  测试基础设施
# ═══════════════════════════════════════════════


class ReleaseServer:
    """最小 GitHub Releases 镜像。

    - ``{base}/releases/latest`` → 302 到 ``/releases/tag/<latest_tag>``（版本检测）；
    - ``{base}/releases/tag/<tag>`` → tag 页；
    - ``{base}/releases/download/<ver>/<name>`` 与静态文件路径均从 directory 读取
      （install.sh 的下载端点，可被 ``STORYBOOK_INSTALL_ARCHIVE_URL`` 覆盖）。
    - 统计各类请求次数，用于断言“已是最新不下载”等零写入行为。
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.latest_tag = "v" + __version__
        self.fail_archive = False
        self.counts = {"latest": 0, "archive": 0, "checksum": 0}
        self.httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), self._make_handler()
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address
        self.url = "http://{}:{}".format(host, port)

    def _make_handler(self):
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/releases/latest":
                    server.counts["latest"] += 1
                    self.send_response(302)
                    self.send_header("Location", "/releases/tag/" + server.latest_tag)
                    self.end_headers()
                    return
                if re.match(r"^/releases/tag/[^/]+/?$", self.path):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"latest release page")
                    return
                match = re.match(r"^/releases/download/([^/]+)/([^/]+)$", self.path)
                if match:
                    _, name = match.groups()
                    self._serve_file(name)
                    return
                # 其它路径（例如 STORYBOOK_INSTALL_ARCHIVE_URL 直接指向的 wheel）
                self._serve_file(self.path.lstrip("/"))

            def _serve_file(self, name):
                if name.endswith(".sha256"):
                    server.counts["checksum"] += 1
                else:
                    server.counts["archive"] += 1
                    if server.fail_archive:
                        self.send_response(404)
                        self.end_headers()
                        return
                path = (server.directory / name).resolve()
                base = server.directory.resolve()
                if not str(path).startswith(str(base)) or not path.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def close(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()


def _write_probe_wheel(path: Path, tag: str) -> None:
    """生成最小 probe wheel：安装后 ``book``/``storybook`` 打印固定 tag。"""
    dist_info = "storybook-0.0.0.dist-info"
    module = (
        "import sys\n\n"
        "def main():\n"
        "    print(%r)\n" % (tag,)
    )
    entrypoints = (
        "[console_scripts]\n"
        "book = storybook_probe:main\n"
        "storybook = storybook_probe:main\n"
    )
    files = {
        "storybook_probe.py": module,
        dist_info + "/METADATA": (
            "Metadata-Version: 2.1\nName: storybook\nVersion: 0.0.0\n"
        ),
        dist_info + "/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: storybook-tests\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        dist_info + "/entry_points.txt": entrypoints,
    }
    record = "".join(name + ",,\n" for name in files)
    record += dist_info + "/RECORD,,\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, content in files.items():
            wheel.writestr(name, content)
        wheel.writestr(dist_info + "/RECORD", record)


def _write_checksum(path: Path, archive: Path) -> None:
    path.write_text(
        hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name + "\n",
        encoding="ascii",
    )


def _env(
    server: ReleaseServer,
    *,
    archive_url=None,
    checksum_url=None,
    home: Path | None = None,
    **extra,
):
    isolated = home or (server.directory / "isolated-home")
    env = {
        **os.environ,
        "STORYBOOK_INSTALL_REPOSITORY": server.url,
        "HOME": str(isolated),
        "XDG_CONFIG_HOME": str(isolated / "config"),
        "XDG_DATA_HOME": str(isolated / "data"),
        "XDG_CACHE_HOME": str(isolated / "cache"),
        "XDG_STATE_HOME": str(isolated / "state"),
    }
    if archive_url:
        env["STORYBOOK_INSTALL_ARCHIVE_URL"] = archive_url
    if checksum_url:
        env["STORYBOOK_INSTALL_CHECKSUM_URL"] = checksum_url
    env.update(extra)
    return env


def _invoke(args, *, env=None, input=None):
    return CliRunner().invoke(cli.cli, args, prog_name="book", env=env, input=input)


def _install_probe(prefix: Path, archive_url: str, checksum_url: str) -> None:
    """用真实 install.sh 把 probe wheel 装进 prefix（模拟“已装旧版本”）。"""
    env = {
        **os.environ,
        "STORYBOOK_INSTALL_ARCHIVE_URL": archive_url,
        "STORYBOOK_INSTALL_CHECKSUM_URL": checksum_url,
        "STORYBOOK_INSTALL_PYTHON": sys.executable,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
    }
    completed = subprocess.run(
        [
            "sh", str(ROOT / "install.sh"),
            "--version", "0.1.3",
            "--prefix", str(prefix),
            "--no-init",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _book_tag(prefix: Path) -> str:
    return subprocess.check_output([str(prefix / "bin" / "book")], text=True).strip()


def _assert_no_storybook_artifacts(isolated: Path) -> None:
    """隔离 HOME 下不得出现 storybook Profile / 数据库产物。

    允许通用工具链（pip/python 等）在 HOME 下写缓存目录；只校验 storybook 自身的
    数据产物：profile 注册表（profiles.json）、SQLite 数据库（.db/.sqlite）以及
    XDG 下的 storybook 平台数据目录。
    """
    if not isolated.exists():
        return
    for path in isolated.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in (".db", ".sqlite", ".sqlite3"), path
            assert path.name != "profiles.json", path
        elif path.is_dir() and path.name == "storybook":
            raise AssertionError(f"unexpected storybook data dir: {path}")


@pytest.fixture
def release_server(tmp_path):
    server = ReleaseServer(tmp_path)
    try:
        yield server
    finally:
        server.close()


# ═══════════════════════════════════════════════
#  验收 1：--help
# ═══════════════════════════════════════════════


def test_update_help():
    result = CliRunner().invoke(cli.cli, ["update", "--help"], prog_name="book")
    assert result.exit_code == 0, result.output
    assert "update" in result.output
    assert "--dry-run" in result.output
    assert "--yes" in result.output
    assert "--prefix" in result.output


# ═══════════════════════════════════════════════
#  验收 2：已是最新 → up-to-date，退出 0，零下载
# ═══════════════════════════════════════════════


def test_update_up_to_date_does_not_download(tmp_path, release_server):
    release_server.latest_tag = "v" + __version__
    prefix = tmp_path / "prefix"

    result = _invoke(["update", "--yes", "--prefix", str(prefix)], env=_env(release_server))

    assert result.exit_code == 0, result.output
    assert "Already up to date (v" + __version__ + ")" in result.stdout
    assert not prefix.exists()
    assert release_server.counts["archive"] == 0
    assert release_server.counts["checksum"] == 0
    assert release_server.counts["latest"] == 1
    assert not (tmp_path / "isolated-home").exists()


def test_update_up_to_date_json(tmp_path, release_server):
    release_server.latest_tag = "v" + __version__
    result = _invoke(
        ["update", "--yes", "--prefix", str(tmp_path / "prefix"), "--json"],
        env=_env(release_server),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "up_to_date"
    assert payload["current_version"] == __version__
    assert payload["latest_version"] == __version__


# ═══════════════════════════════════════════════
#  验收 4：--dry-run 零写入，输出当前/最新/prefix
# ═══════════════════════════════════════════════


def test_update_dry_run_zero_writes(tmp_path, release_server):
    release_server.latest_tag = "v9.9.9"
    prefix = tmp_path / "prefix"

    result = _invoke(["update", "--dry-run", "--prefix", str(prefix)], env=_env(release_server))

    assert result.exit_code == 0, result.output
    assert __version__ in result.stdout
    assert "9.9.9" in result.stdout
    assert str(prefix) in result.stdout
    assert release_server.url + "/releases/latest" in result.stdout
    assert "no writes performed" in result.stdout
    assert not prefix.exists()
    assert release_server.counts["archive"] == 0
    assert release_server.counts["checksum"] == 0
    assert not (tmp_path / "isolated-home").exists()


def test_update_dry_run_json(tmp_path, release_server):
    release_server.latest_tag = "v9.9.9"
    prefix = tmp_path / "prefix"
    result = _invoke(
        ["update", "--dry-run", "--prefix", str(prefix), "--json"],
        env=_env(release_server),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["current_version"] == __version__
    assert payload["latest_version"] == "9.9.9"
    assert payload["prefix"] == str(prefix)


# ═══════════════════════════════════════════════
#  验收：默认确认
# ═══════════════════════════════════════════════


def test_update_non_interactive_requires_yes(tmp_path, release_server, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.BytesIO(b""))
    release_server.latest_tag = "v9.9.9"
    prefix = tmp_path / "prefix"

    result = _invoke(["update", "--prefix", str(prefix)], env=_env(release_server))

    assert result.exit_code != 0
    assert "SB_UPDATE_CONFIRM_REQUIRED" in result.stderr
    assert not prefix.exists()
    assert release_server.counts["archive"] == 0


# ═══════════════════════════════════════════════
#  验收 5：下载失败 → 稳定错误、退出非 0、无部分安装
# ═══════════════════════════════════════════════


def test_update_download_failure_no_partial_install(tmp_path, release_server):
    release_server.latest_tag = "v9.9.9"
    release_server.fail_archive = True
    prefix = tmp_path / "prefix"

    result = _invoke(["update", "--yes", "--prefix", str(prefix)], env=_env(release_server))

    assert result.exit_code != 0
    assert "SB_INSTALL_DOWNLOAD_FAILED" in result.stderr
    assert not (prefix / "lib" / "storybook" / "current").exists()
    assert not (prefix / "bin" / "book").exists()
    assert not (tmp_path / "isolated-home").exists()


# ═══════════════════════════════════════════════
#  验收 6：checksum 不匹配 → 中止安装、退出非 0、无部分安装
# ═══════════════════════════════════════════════


def test_update_checksum_mismatch_aborts(tmp_path, release_server):
    release_server.latest_tag = "v9.9.9"
    archive = tmp_path / "storybook.tar.gz"
    archive.write_bytes(b"payload")
    (tmp_path / "storybook.tar.gz.sha256").write_text(
        "0" * 64 + "  storybook.tar.gz\n", encoding="ascii"
    )
    prefix = tmp_path / "prefix"

    result = _invoke(["update", "--yes", "--prefix", str(prefix)], env=_env(release_server))

    assert result.exit_code != 0
    assert "SB_INSTALL_CHECKSUM_MISMATCH" in result.stderr
    assert not (prefix / "lib" / "storybook" / "current").exists()
    assert not (prefix / "bin" / "book").exists()
    assert not (tmp_path / "isolated-home").exists()


# ═══════════════════════════════════════════════
#  验收 7：不修改 Profile / 数据库（隔离 HOME）
# ═══════════════════════════════════════════════


def test_update_does_not_touch_profile_or_database(tmp_path, release_server):
    release_server.latest_tag = "v" + __version__
    isolated = tmp_path / "isolated"
    env = {
        **os.environ,
        "STORYBOOK_INSTALL_REPOSITORY": release_server.url,
        "HOME": str(isolated / "home"),
        "XDG_CONFIG_HOME": str(isolated / "config"),
        "XDG_DATA_HOME": str(isolated / "data"),
        "XDG_CACHE_HOME": str(isolated / "cache"),
        "XDG_STATE_HOME": str(isolated / "state"),
    }
    result = _invoke(
        ["update", "--yes", "--prefix", str(tmp_path / "prefix")], env=env
    )
    assert result.exit_code == 0, result.output
    assert "Already up to date" in result.stdout
    assert not isolated.exists()


# ═══════════════════════════════════════════════
#  验收 3：端到端升级（下载 → sha256 校验 → 原子安装 → 新版本生效）
# ═══════════════════════════════════════════════


def test_update_installs_new_version_end_to_end(tmp_path, release_server):
    v2 = tmp_path / "storybook-v2.whl"
    _write_probe_wheel(v2, "v2")
    _write_checksum(tmp_path / "storybook-v2.whl.sha256", v2)
    release_server.latest_tag = "v0.1.4"
    prefix = tmp_path / "prefix"
    isolated = tmp_path / "isolated-home"

    result = _invoke(
        ["update", "--yes", "--prefix", str(prefix)],
        env=_env(
            release_server,
            archive_url=release_server.url + "/storybook-v2.whl",
            checksum_url=release_server.url + "/storybook-v2.whl.sha256",
            home=isolated,
            PIP_CACHE_DIR=str(tmp_path / "pip-cache"),
            PIP_NO_CACHE_DIR="1",
        ),
    )

    assert result.exit_code == 0, result.stderr
    assert "Installed Storybook 0.1.4." in result.stdout
    assert release_server.counts["archive"] >= 1
    assert release_server.counts["checksum"] >= 1
    # 新版本已激活：current → 0.1.4 release，book 入口输出新 probe tag
    current = prefix / "lib" / "storybook" / "current"
    assert current.is_symlink()
    assert "0.1.4-" in current.resolve().name
    assert _book_tag(prefix) == "v2"
    assert (prefix / "bin" / "storybook").is_symlink()
    # 隔离 HOME 下不产生 storybook Profile / 数据库产物（允许通用工具链缓存）
    _assert_no_storybook_artifacts(isolated)


# ═══════════════════════════════════════════════
#  验收 5/6：升级失败时既有安装不受影响
# ═══════════════════════════════════════════════


def test_update_failure_keeps_existing_install(tmp_path, release_server):
    v1 = tmp_path / "storybook-v1.whl"
    _write_probe_wheel(v1, "v1")
    _write_checksum(tmp_path / "storybook-v1.whl.sha256", v1)
    prefix = tmp_path / "prefix"
    _install_probe(
        prefix,
        release_server.url + "/storybook-v1.whl",
        release_server.url + "/storybook-v1.whl.sha256",
    )
    assert _book_tag(prefix) == "v1"
    previous_current = (prefix / "lib" / "storybook" / "current").resolve()

    # 新版可用但 checksum 错误 → 升级中止
    v2 = tmp_path / "storybook-v2.whl"
    _write_probe_wheel(v2, "v2")
    (tmp_path / "storybook-v2.whl.sha256").write_text(
        "0" * 64 + "  storybook-v2.whl\n", encoding="ascii"
    )
    release_server.latest_tag = "v0.1.4"

    result = _invoke(
        ["update", "--yes", "--prefix", str(prefix)],
        env=_env(
            release_server,
            archive_url=release_server.url + "/storybook-v2.whl",
            checksum_url=release_server.url + "/storybook-v2.whl.sha256",
        ),
    )

    assert result.exit_code != 0
    assert "SB_INSTALL_CHECKSUM_MISMATCH" in result.stderr
    assert _book_tag(prefix) == "v1"
    assert (prefix / "lib" / "storybook" / "current").resolve() == previous_current
    releases = prefix / "lib" / "storybook" / "releases"
    targets = [entry for entry in releases.iterdir() if entry.is_dir()]
    assert len(targets) == 1


# ═══════════════════════════════════════════════
#  安装器单一来源：随包副本与仓库根 install.sh 一致
# ═══════════════════════════════════════════════


def test_bundled_installer_matches_repo_root():
    bundled = Path(cli.__file__).resolve().parent / "data" / "install.sh"
    assert bundled.read_bytes() == (ROOT / "install.sh").read_bytes()
