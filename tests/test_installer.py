"""Black-box contracts for the POSIX user-local installer."""
from __future__ import annotations

import errno
import hashlib
import http.server
import os
import pty
import select
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "install.sh"


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _fake_tools(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    tools = tmp_path / "tools"
    tools.mkdir()
    archive = tmp_path / "release.tar.gz"
    archive.write_bytes(b"verified storybook release")
    checksum = tmp_path / "release.tar.gz.sha256"
    checksum.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  storybook.tar.gz\n",
        encoding="ascii",
    )
    _executable(
        tools / "curl",
        """#!/bin/sh
set -eu
[ "${FAKE_DOWNLOAD_FAIL:-0}" = 1 ] && exit 22
destination=
url=
while [ "$#" -gt 0 ]; do
  case $1 in -o) destination=$2; shift 2;; -*) shift;; *) url=$1; shift;; esac
done
case $url in *.sha256) cp "$FAKE_CHECKSUM" "$destination";; *) cp "$FAKE_ARCHIVE" "$destination";; esac
""",
    )
    _executable(
        tools / "python3",
        """#!/bin/sh
set -eu
if [ "${1:-}" = - ]; then
  case ${2:-} in https://mirror.invalid/*) exit 0;; *) exit 1;; esac
fi
if [ "${1:-}" = -c ] && [ "${3:-}" = storybook-sqlite-extension-check ]; then
  [ "${FAKE_SQLITE_EXTENSION_MISSING:-0}" = 1 ] && exit 1
  exit 0
fi
if [ "${1:-}" = -c ] && [ "$#" -gt 2 ]; then rm -f "$4"; mv "$3" "$4"; exit 0; fi
if [ "${1:-}" = -c ]; then echo 3.11; exit 0; fi
if [ "${1:-}" = -m ] && [ "${2:-}" = venv ] && [ "${3:-}" = --help ]; then
  [ "${FAKE_VENV_MISSING:-0}" = 1 ] && exit 1
  exit 0
fi
if [ "${1:-}" = -m ] && [ "${2:-}" = venv ]; then
  target=$3
  mkdir -p "$target/bin"
  cat >"$target/bin/pip" <<'PIP'
#!/bin/sh
set -eu
[ "${FAKE_PIP_FAIL:-0}" = 1 ] && exit 42
bin=$(dirname "$0")
for name in book storybook; do
  cat >"$bin/$name" <<ENTRY
#!/bin/sh
printf '%s\\n' "${FAKE_INSTALL_TAG:-installed}"
ENTRY
  chmod +x "$bin/$name"
done
PIP
  chmod +x "$target/bin/pip"
  exit 0
fi
exit 2
""",
    )
    _executable(
        tools / "uname",
        """#!/bin/sh
case ${1:-} in -s) printf '%s\n' "${FAKE_OS:-Darwin}";; -m) printf '%s\n' "${FAKE_ARCH:-arm64}";; *) exit 2;; esac
""",
    )
    env = {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "FAKE_ARCHIVE": str(archive),
        "FAKE_CHECKSUM": str(checksum),
        "STORYBOOK_INSTALL_ARCHIVE_URL": "https://mirror.invalid/storybook.tar.gz",
        "STORYBOOK_INSTALL_CHECKSUM_URL": "https://mirror.invalid/storybook.tar.gz.sha256",
        "STORYBOOK_INSTALL_PYTHON": "python3",
        "FAKE_OS": "Darwin",
        "FAKE_ARCH": "arm64",
    }
    return tools, env


def _run(prefix: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(INSTALLER), "--prefix", str(prefix), "--no-init", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_pty(fd: int, size: int) -> bytes:
    """Read a PTY, treating Linux EIO after slave close as portable EOF."""
    try:
        return os.read(fd, size)
    except OSError as exc:
        if exc.errno == errno.EIO:
            return b""
        raise


def _write_probe_wheel(path: Path, tag: str, *, both_entrypoints: bool = True) -> None:
    dist_info = "storybook-0.0.0.dist-info"
    module = f'''import os
import pathlib
import sys

def main():
    if sys.argv[1:] == ["init"] and os.environ.get("PROBE_FILE"):
        pathlib.Path(os.environ["PROBE_FILE"]).write_text(
            os.environ.get("STORYBOOK_LAUNCHER", ""), encoding="utf-8"
        )
    print({tag!r})
'''
    entrypoints = "[console_scripts]\nbook = storybook_probe:main\n"
    if both_entrypoints:
        entrypoints += "storybook = storybook_probe:main\n"
    files = {
        "storybook_probe.py": module,
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\nName: storybook\nVersion: 0.0.0\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: storybook-tests\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": entrypoints,
    }
    record = "".join(f"{name},,\n" for name in files)
    record += f"{dist_info}/RECORD,,\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, content in files.items():
            wheel.writestr(name, content)
        wheel.writestr(f"{dist_info}/RECORD", record)


def _use_real_python_wheel(
    tmp_path: Path, release_url: str
) -> tuple[Path, dict[str, str], Path]:
    wheel = tmp_path / "storybook-0.0.0-py3-none-any.whl"
    _write_probe_wheel(wheel, "v1")
    checksum = tmp_path / f"{wheel.name}.sha256"
    checksum.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n",
        encoding="ascii",
    )
    env = {
        **os.environ,
        "STORYBOOK_INSTALL_ARCHIVE_URL": f"{release_url}/{wheel.name}",
        "STORYBOOK_INSTALL_CHECKSUM_URL": f"{release_url}/{checksum.name}",
        "STORYBOOK_INSTALL_PYTHON": sys.executable,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    return wheel, env, checksum


@pytest.fixture
def mutable_release_url(tmp_path: Path):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    handler = lambda *args, **kwargs: QuietHandler(  # noqa: E731
        *args, directory=str(tmp_path), **kwargs
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture(scope="session")
def official_release_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("official-release")
    env = {**os.environ, "PYTHON": sys.executable}
    completed = subprocess.run(
        [ROOT / "scripts" / "build_release_assets.sh", output],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "storybook.tar.gz").is_file()
    assert (output / "storybook.tar.gz.sha256").is_file()
    return output


@pytest.fixture
def official_release_url(official_release_assets: Path):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    handler = lambda *args, **kwargs: QuietHandler(  # noqa: E731
        *args, directory=str(official_release_assets), **kwargs
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize(("system", "architecture"), [("Darwin", "arm64"), ("Linux", "x86_64")])
def test_dry_run_with_space_in_prefix_performs_zero_writes(
    tmp_path, system, architecture
):
    _, env = _fake_tools(tmp_path)
    env.update(FAKE_OS=system, FAKE_ARCH=architecture)
    prefix = tmp_path / "prefix with spaces"

    completed = _run(prefix, env, "--dry-run")

    assert completed.returncode == 0, completed.stderr
    assert "no writes performed" in completed.stdout
    assert not prefix.exists()


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "https://SECRET-SENTINEL@example.invalid/storybook.tar.gz",
        "https://user:SECRET-SENTINEL@example.invalid/storybook.tar.gz",
        "https://SECRET-SENTINEL%40name@example.invalid/storybook.tar.gz",
        "https://example.invalid/storybook.tar.gz?token=SECRET-SENTINEL",
        "https://example.invalid/storybook.tar.gz#SECRET-SENTINEL",
    ],
)
def test_unsafe_download_url_is_rejected_without_echoing_secrets(
    tmp_path, unsafe_url
):
    _, env = _fake_tools(tmp_path)
    env.update(
        STORYBOOK_INSTALL_ARCHIVE_URL=unsafe_url,
        STORYBOOK_INSTALL_PYTHON=sys.executable,
    )
    prefix = tmp_path / "must-not-exist"

    completed = _run(prefix, env, "--dry-run")

    assert completed.returncode == 1
    assert "SB_INSTALL_URL_UNSAFE" in completed.stderr
    assert "SECRET-SENTINEL" not in completed.stdout
    assert "SECRET-SENTINEL" not in completed.stderr
    assert not prefix.exists()


def test_safe_https_download_url_passes_dry_run(tmp_path):
    _, env = _fake_tools(tmp_path)
    env["STORYBOOK_INSTALL_PYTHON"] = sys.executable

    completed = _run(tmp_path / "prefix", env, "--dry-run")

    assert completed.returncode == 0, completed.stderr
    assert "https://mirror.invalid/storybook.tar.gz" in completed.stdout


def test_missing_sqlite_extension_fails_before_writes_with_macos_repair(tmp_path):
    _, env = _fake_tools(tmp_path)
    env["FAKE_SQLITE_EXTENSION_MISSING"] = "1"
    prefix = tmp_path / "must-not-exist"

    failed = _run(prefix, env)

    assert failed.returncode == 1
    assert "SB_INSTALL_SQLITE_EXTENSION_UNAVAILABLE" in failed.stderr
    assert "brew install python@3.11" in failed.stderr
    assert (
        "STORYBOOK_INSTALL_PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11"
        in failed.stderr
    )
    assert not prefix.exists()


def test_generated_official_asset_installs_with_real_downloader_and_rolls_back(
    tmp_path, official_release_assets, official_release_url
):
    prefix = tmp_path / "official prefix"
    archive_url = f"{official_release_url}/storybook.tar.gz"
    checksum_url = f"{archive_url}.sha256"
    env = {
        **os.environ,
        "STORYBOOK_INSTALL_ARCHIVE_URL": archive_url,
        "STORYBOOK_INSTALL_CHECKSUM_URL": checksum_url,
        "STORYBOOK_INSTALL_PYTHON": sys.executable,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }

    installed = _run(prefix, env, "--version", "0.1.3")
    assert installed.returncode == 0, installed.stderr
    for entrypoint in ("book", "storybook"):
        checked = subprocess.run(
            [prefix / "bin" / entrypoint, "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr

    repeated = _run(prefix, env, "--version", "0.1.3")
    assert repeated.returncode == 0, repeated.stderr
    upgraded = _run(prefix, env, "--version", "0.1.4")
    assert upgraded.returncode == 0, upgraded.stderr
    active_before = (prefix / "lib" / "storybook" / "current").resolve()

    checksum = official_release_assets / "storybook.tar.gz.sha256"
    original_checksum = checksum.read_text(encoding="ascii")
    checksum.write_text("0" * 64 + "  storybook.tar.gz\n", encoding="ascii")
    try:
        failed = _run(prefix, env, "--version", "0.1.5")
    finally:
        checksum.write_text(original_checksum, encoding="ascii")

    assert failed.returncode == 1
    assert "SB_INSTALL_CHECKSUM_MISMATCH" in failed.stderr
    assert (prefix / "lib" / "storybook" / "current").resolve() == active_before
    assert subprocess.run(
        [prefix / "bin" / "book", "--help"], check=False
    ).returncode == 0


def test_clean_repeat_and_version_upgrade_atomically_switch_release(tmp_path):
    _, env = _fake_tools(tmp_path)
    prefix = tmp_path / "prefix with spaces"
    env["FAKE_INSTALL_TAG"] = "first"

    first = _run(prefix, env, "--version", "1.2.3")

    assert first.returncode == 0, first.stderr
    assert subprocess.check_output([prefix / "bin" / "book"], text=True).strip() == "first"
    assert (prefix / "bin" / "storybook").is_symlink()

    env["FAKE_INSTALL_TAG"] = "second"
    second = _run(prefix, env, "--version", "1.2.3")

    assert second.returncode == 0, second.stderr
    assert subprocess.check_output([prefix / "bin" / "book"], text=True).strip() == "first"

    archive = Path(env["FAKE_ARCHIVE"])
    archive.write_bytes(b"verified storybook release v2")
    Path(env["FAKE_CHECKSUM"]).write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  storybook.tar.gz\n",
        encoding="ascii",
    )
    upgraded = _run(prefix, env, "--version", "1.2.4")

    assert upgraded.returncode == 0, upgraded.stderr
    assert subprocess.check_output([prefix / "bin" / "book"], text=True).strip() == "second"


def test_checksum_failure_keeps_previous_release_running(tmp_path):
    _, env = _fake_tools(tmp_path)
    prefix = tmp_path / "prefix"
    env["FAKE_INSTALL_TAG"] = "known-good"
    assert _run(prefix, env, "--version", "1.2.3").returncode == 0
    Path(env["FAKE_CHECKSUM"]).write_text("0" * 64 + "  storybook.tar.gz\n")
    env["FAKE_INSTALL_TAG"] = "bad"

    failed = _run(prefix, env, "--version", "1.2.3")

    assert failed.returncode == 1
    assert "SB_INSTALL_CHECKSUM_MISMATCH" in failed.stderr
    assert subprocess.check_output([prefix / "bin" / "book"], text=True).strip() == "known-good"


@pytest.mark.parametrize(
    ("environment", "code"),
    [
        ({"STORYBOOK_INSTALL_PYTHON": "missing-python"}, "SB_INSTALL_PYTHON_MISSING"),
        ({"FAKE_VENV_MISSING": "1"}, "SB_INSTALL_VENV_MISSING"),
        ({"FAKE_DOWNLOAD_FAIL": "1"}, "SB_INSTALL_DOWNLOAD_FAILED"),
    ],
)
def test_preflight_and_download_failures_have_stable_codes_and_zero_writes(
    tmp_path, environment, code
):
    _, env = _fake_tools(tmp_path)
    env.update(environment)
    prefix = tmp_path / "prefix"

    failed = _run(prefix, env)

    assert failed.returncode == 1
    assert code in failed.stderr
    assert not prefix.exists()


def test_package_install_failure_keeps_previous_release_running(tmp_path):
    _, env = _fake_tools(tmp_path)
    prefix = tmp_path / "prefix"
    env["FAKE_INSTALL_TAG"] = "known-good"
    assert _run(prefix, env, "--version", "1.2.3").returncode == 0
    archive = Path(env["FAKE_ARCHIVE"])
    archive.write_bytes(b"broken next release")
    Path(env["FAKE_CHECKSUM"]).write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  storybook.tar.gz\n",
        encoding="ascii",
    )
    env["FAKE_PIP_FAIL"] = "1"

    failed = _run(prefix, env, "--version", "1.2.4")

    assert failed.returncode == 1
    assert "SB_INSTALL_PACKAGE_FAILED" in failed.stderr
    assert subprocess.check_output([prefix / "bin" / "book"], text=True).strip() == "known-good"


def test_real_venv_entrypoints_survive_clean_install_upgrade_and_bad_release(
    tmp_path, mutable_release_url,
):
    wheel, env, checksum = _use_real_python_wheel(tmp_path, mutable_release_url)
    prefix = tmp_path / "real prefix"

    installed = _run(prefix, env, "--version", "1.2.3")

    assert installed.returncode == 0, installed.stderr
    for name in ("book", "storybook"):
        completed = subprocess.run(
            [prefix / "bin" / name, "--help"], text=True,
            capture_output=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "v1"

    _write_probe_wheel(wheel, "v2")
    checksum.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n",
        encoding="ascii",
    )
    upgraded = _run(prefix, env, "--version", "1.2.4")
    assert upgraded.returncode == 0, upgraded.stderr
    assert subprocess.check_output(
        [prefix / "bin" / "book", "--help"], text=True
    ).strip() == "v2"

    _write_probe_wheel(wheel, "broken", both_entrypoints=False)
    checksum.write_text(
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n",
        encoding="ascii",
    )
    failed = _run(prefix, env, "--version", "1.2.5")
    assert failed.returncode == 1
    assert "SB_INSTALL_ENTRYPOINT_MISSING" in failed.stderr
    assert subprocess.check_output(
        [prefix / "bin" / "book", "--help"], text=True
    ).strip() == "v2"


@pytest.mark.skipif(os.name == "nt", reason="installer onboarding TTY is POSIX-only")
def test_path_missing_onboarding_exports_stable_launcher_to_book_init(
    tmp_path, mutable_release_url,
):
    _, env, _ = _use_real_python_wheel(tmp_path, mutable_release_url)
    prefix = tmp_path / "prefix"
    probe = tmp_path / "launcher.txt"
    env["PROBE_FILE"] = str(probe)
    assert str(prefix / "bin") not in env["PATH"].split(os.pathsep)
    master, slave = pty.openpty()
    process = subprocess.Popen(
        ["sh", str(INSTALLER), "--prefix", str(prefix), "--version", "1.2.3"],
        cwd=ROOT, env=env, stdin=slave, stdout=slave, stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    deadline = time.monotonic() + 30
    try:
        while b"Run onboarding now?" not in output:
            if time.monotonic() > deadline or process.poll() is not None:
                pytest.fail(output.decode(errors="replace"))
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                chunk = _read_pty(master, 8192)
                if not chunk:
                    process.wait(timeout=5)
                    pytest.fail(output.decode(errors="replace"))
                output.extend(chunk)
        os.write(master, b"\n")
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
        os.close(master)

    assert probe.read_text(encoding="utf-8") == str(prefix / "bin" / "book")
