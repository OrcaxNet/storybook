#!/bin/sh
# Storybook user-local installer. POSIX sh; never requires sudo.

set -eu

PROGRAM=storybook-install
VERSION=latest
PREFIX=${HOME:-}/.local
DRY_RUN=0
RUN_INIT=1
PYTHON=${STORYBOOK_INSTALL_PYTHON:-python3}
REPOSITORY=${STORYBOOK_INSTALL_REPOSITORY:-https://github.com/OrcaxNet/storybook}
TEMP_DIR=
TARGET_CREATED=0
TARGET=

usage() {
    cat <<'EOF'
Usage: install.sh [options]

Options:
  --version VERSION  Install an official release (default: latest)
  --prefix PATH      User-owned install prefix (default: $HOME/.local)
  --dry-run          Validate and print the plan without writing
  --no-init          Do not offer to run book init after installation
  --help             Show this help

Environment for mirrors/testing:
  STORYBOOK_INSTALL_ARCHIVE_URL   Override the release archive URL
  STORYBOOK_INSTALL_CHECKSUM_URL  Override the SHA-256 file URL
  STORYBOOK_INSTALL_PYTHON        Override the Python executable
EOF
}

fail() {
    code=$1
    shift
    printf '%s: error [%s]: %s\n' "$PROGRAM" "$code" "$*" >&2
    exit 1
}

cleanup() {
    if [ "$TARGET_CREATED" -eq 1 ] && [ -n "$TARGET" ] && [ -d "$TARGET" ]; then
        rm -rf "$TARGET"
    fi
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT HUP INT TERM

while [ "$#" -gt 0 ]; do
    case $1 in
        --version)
            [ "$#" -ge 2 ] || fail SB_INSTALL_USAGE "--version requires a value"
            VERSION=$2
            shift 2
            ;;
        --prefix)
            [ "$#" -ge 2 ] || fail SB_INSTALL_USAGE "--prefix requires a value"
            PREFIX=$2
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-init) RUN_INIT=0; shift ;;
        --help|-h) usage; exit 0 ;;
        *) fail SB_INSTALL_USAGE "unknown option: $1 (run with --help)" ;;
    esac
done

[ -n "$PREFIX" ] || fail SB_INSTALL_PREFIX_INVALID "prefix cannot be empty"
case $PREFIX in /*) ;; *) PREFIX=$(pwd)/$PREFIX ;; esac
case $VERSION in
    *[!A-Za-z0-9._-]*|'') fail SB_INSTALL_VERSION_INVALID "invalid version: $VERSION" ;;
esac

OS=$(uname -s 2>/dev/null || true)
ARCH=$(uname -m 2>/dev/null || true)
case $OS in
    Darwin|Linux) ;;
    *) fail SB_INSTALL_OS_UNSUPPORTED "supported systems are macOS and Linux (found ${OS:-unknown})" ;;
esac
case $ARCH in
    x86_64|amd64|arm64|aarch64) ;;
    *) fail SB_INSTALL_ARCH_UNSUPPORTED "unsupported architecture: ${ARCH:-unknown}" ;;
esac

command -v "$PYTHON" >/dev/null 2>&1 || fail SB_INSTALL_PYTHON_MISSING "Python 3.11+ is required; install it and retry"
PYTHON_VERSION=$(
    "$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2]); raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null
) || fail SB_INSTALL_PYTHON_TOO_OLD "Python 3.11+ is required (found ${PYTHON_VERSION:-unknown})"
"$PYTHON" -m venv --help >/dev/null 2>&1 || fail SB_INSTALL_VENV_MISSING "Python venv is unavailable; install python3-venv and retry"
if ! "$PYTHON" -c '
import sqlite3
connection = sqlite3.connect(":memory:")
connection.enable_load_extension(True)
connection.enable_load_extension(False)
connection.close()
' storybook-sqlite-extension-check >/dev/null 2>&1; then
    case $OS:$ARCH in
        Darwin:arm64|Darwin:aarch64)
            fail SB_INSTALL_SQLITE_EXTENSION_UNAVAILABLE 'Python SQLite must support loadable extensions; run: brew install python@3.11; then retry with STORYBOOK_INSTALL_PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11'
            ;;
        Darwin:*)
            fail SB_INSTALL_SQLITE_EXTENSION_UNAVAILABLE 'Python SQLite must support loadable extensions; run: brew install python@3.11; then retry with STORYBOOK_INSTALL_PYTHON=/usr/local/opt/python@3.11/bin/python3.11'
            ;;
        *)
            fail SB_INSTALL_SQLITE_EXTENSION_UNAVAILABLE 'Python SQLite must support loadable extensions; install a Python build compiled with --enable-loadable-sqlite-extensions and retry'
            ;;
    esac
fi

if command -v curl >/dev/null 2>&1; then
    DOWNLOADER=curl
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER=wget
else
    fail SB_INSTALL_DOWNLOADER_MISSING "install curl or wget and retry"
fi

if [ "$VERSION" = latest ]; then
    ARCHIVE_URL=${STORYBOOK_INSTALL_ARCHIVE_URL:-$REPOSITORY/releases/latest/download/storybook.tar.gz}
    CHECKSUM_URL=${STORYBOOK_INSTALL_CHECKSUM_URL:-$REPOSITORY/releases/latest/download/storybook.tar.gz.sha256}
else
    ARCHIVE_URL=${STORYBOOK_INSTALL_ARCHIVE_URL:-$REPOSITORY/releases/download/v$VERSION/storybook.tar.gz}
    CHECKSUM_URL=${STORYBOOK_INSTALL_CHECKSUM_URL:-$ARCHIVE_URL.sha256}
fi

validate_download_url() {
    "$PYTHON" - "$1" >/dev/null 2>&1 <<'PY'
import sys
from urllib.parse import urlsplit

try:
    raw = sys.argv[1]
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower()
    safe_scheme = parsed.scheme == "https"
    safe_loopback = parsed.scheme == "http" and host in {"127.0.0.1", "localhost"}
    if (
        not raw
        or any(ord(character) < 33 or ord(character) == 127 for character in raw)
        or not parsed.netloc
        or not host
        or not (safe_scheme or safe_loopback)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError
    parsed.port
except (ValueError, UnicodeError):
    raise SystemExit(1)
PY
}

validate_download_url "$ARCHIVE_URL" || \
    fail SB_INSTALL_URL_UNSAFE "download URLs must use HTTPS without credentials, query, or fragment"
validate_download_url "$CHECKSUM_URL" || \
    fail SB_INSTALL_URL_UNSAFE "download URLs must use HTTPS without credentials, query, or fragment"

INSTALL_ROOT=$PREFIX/lib/storybook
BIN_DIR=$PREFIX/bin
RELEASE_NAME=$VERSION

printf 'Storybook install plan\n'
printf '  Platform  %s/%s\n' "$OS" "$ARCH"
printf '  Python    %s (%s)\n' "$PYTHON" "$PYTHON_VERSION"
printf '  Version   %s\n' "$VERSION"
printf '  Prefix    %s\n' "$PREFIX"
printf '  Source    %s\n' "$ARCHIVE_URL"

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'Dry-run complete: no writes performed.\n'
    exit 0
fi

download() {
    source_url=$1
    destination=$2
    case $source_url in
        http://127.0.0.1/*|http://127.0.0.1:*|http://localhost/*|http://localhost:*)
            allowed_protocols='=http,https'
            loopback_http=1
            ;;
        *)
            allowed_protocols='=https'
            loopback_http=0
            ;;
    esac
    if [ "$DOWNLOADER" = curl ]; then
        curl -fsSL --proto "$allowed_protocols" --tlsv1.2 "$source_url" -o "$destination"
    elif [ "$loopback_http" -eq 1 ]; then
        wget -q -O "$destination" "$source_url"
    else
        wget -q --https-only -O "$destination" "$source_url"
    fi
}

TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/storybook-install.XXXXXX") || fail SB_INSTALL_TEMP_FAILED "cannot create temporary directory"
case $ARCHIVE_URL in
    *.whl) ARCHIVE=$TEMP_DIR/storybook-0.0.0-py3-none-any.whl ;;
    *) ARCHIVE=$TEMP_DIR/storybook.tar.gz ;;
esac
CHECKSUM=$TEMP_DIR/storybook.tar.gz.sha256
download "$ARCHIVE_URL" "$ARCHIVE" || fail SB_INSTALL_DOWNLOAD_FAILED "release download failed; the previous installation is unchanged"
download "$CHECKSUM_URL" "$CHECKSUM" || fail SB_INSTALL_CHECKSUM_DOWNLOAD_FAILED "checksum download failed; the previous installation is unchanged"

EXPECTED=$(awk 'NF {print $1; exit}' "$CHECKSUM")
[ "${#EXPECTED}" -eq 64 ] || fail SB_INSTALL_CHECKSUM_INVALID "official checksum file is invalid"
case $EXPECTED in *[!0-9A-Fa-f]*) fail SB_INSTALL_CHECKSUM_INVALID "official checksum file is invalid" ;; esac
EXPECTED=$(printf '%s' "$EXPECTED" | tr 'A-F' 'a-f')
if command -v shasum >/dev/null 2>&1; then
    ACTUAL=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
elif command -v sha256sum >/dev/null 2>&1; then
    ACTUAL=$(sha256sum "$ARCHIVE" | awk '{print $1}')
else
    ACTUAL=$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ARCHIVE")
fi
[ "$EXPECTED" = "$ACTUAL" ] || fail SB_INSTALL_CHECKSUM_MISMATCH "release checksum mismatch; the previous installation is unchanged"

mkdir -p "$INSTALL_ROOT/releases" "$BIN_DIR"
TARGET_NAME=$RELEASE_NAME-$(printf '%s' "$ACTUAL" | cut -c1-16)-venv2
TARGET=$INSTALL_ROOT/releases/$TARGET_NAME
if [ ! -f "$TARGET/.storybook-ready" ]; then
    [ ! -e "$TARGET" ] || fail SB_INSTALL_TARGET_INCOMPLETE "release target exists without a ready marker; the previous installation is unchanged"
    mkdir "$TARGET" || fail SB_INSTALL_TARGET_FAILED "could not reserve the release target; the previous installation is unchanged"
    TARGET_CREATED=1
    VENV_LOG=$TEMP_DIR/venv-create.log
    if ! "$PYTHON" -m venv "$TARGET" >"$VENV_LOG" 2>&1; then
        # Surface the real venv error verbatim, then give targeted repair guidance.
        if [ -s "$VENV_LOG" ]; then
            printf '%s: venv creation failed with the following output:\n' "$PROGRAM" >&2
            cat "$VENV_LOG" >&2
        fi
        if grep -qi 'ensurepip' "$VENV_LOG" 2>/dev/null; then
            case $OS in
                Darwin)
                    PYTHON_EXEC=$("$PYTHON" -c 'import sys; print(sys.executable)' 2>/dev/null || printf '%s' "$PYTHON")
                    case $PYTHON_EXEC in
                        /opt/homebrew/*|/usr/local/*)
                            fail SB_INSTALL_VENV_FAILED "could not create the isolated environment (ensurepip failed to bootstrap pip); run: brew update && brew upgrade python@$PYTHON_VERSION (or brew reinstall python@$PYTHON_VERSION), then retry"
                            ;;
                        *)
                            fail SB_INSTALL_VENV_FAILED "could not create the isolated environment (ensurepip failed to bootstrap pip); repair or reinstall your Python installation and retry"
                            ;;
                    esac
                    ;;
                Linux)
                    if command -v apt-get >/dev/null 2>&1 || [ -f /etc/debian_version ]; then
                        fail SB_INSTALL_VENV_FAILED "could not create the isolated environment (ensurepip failed to bootstrap pip); install python3-venv and retry (e.g. sudo apt install python3-venv on Debian/Ubuntu)"
                    else
                        fail SB_INSTALL_VENV_FAILED "could not create the isolated environment (ensurepip failed to bootstrap pip); install your distribution's python3-venv package (or repair the Python installation) and retry"
                    fi
                    ;;
                *)
                    fail SB_INSTALL_VENV_FAILED "could not create the isolated environment (ensurepip failed to bootstrap pip); repair or reinstall your Python installation and retry"
                    ;;
            esac
        fi
        fail SB_INSTALL_VENV_FAILED "could not create the isolated environment"
    fi
    [ -x "$TARGET/bin/pip" ] || fail SB_INSTALL_PIP_MISSING "venv did not provide pip; repair the Python installation"
    "$TARGET/bin/pip" install --disable-pip-version-check "$ARCHIVE" >/dev/null || fail SB_INSTALL_PACKAGE_FAILED "package installation failed; the previous installation is unchanged"
    if [ ! -x "$TARGET/bin/book" ] || [ ! -x "$TARGET/bin/storybook" ]; then
        fail SB_INSTALL_ENTRYPOINT_MISSING "installed package did not provide book and storybook"
    fi
    if ! "$TARGET/bin/book" --help >/dev/null 2>&1 || ! "$TARGET/bin/storybook" --help >/dev/null 2>&1; then
        fail SB_INSTALL_ENTRYPOINT_BROKEN "installed entrypoints failed before activation; the previous installation is unchanged"
    fi
    : >"$TARGET/.storybook-ready"
    TARGET_CREATED=0
else
    if ! "$TARGET/bin/book" --help >/dev/null 2>&1 || ! "$TARGET/bin/storybook" --help >/dev/null 2>&1; then
        fail SB_INSTALL_ENTRYPOINT_BROKEN "cached release entrypoints are invalid; the previous installation is unchanged"
    fi
fi

CURRENT_TMP=$INSTALL_ROOT/.current.$$
BOOK_TMP=$BIN_DIR/.book.$$
STORYBOOK_TMP=$BIN_DIR/.storybook.$$
ln -s "$INSTALL_ROOT/current/bin/book" "$BOOK_TMP"
ln -s "$INSTALL_ROOT/current/bin/storybook" "$STORYBOOK_TMP"
mv -f "$BOOK_TMP" "$BIN_DIR/book"
mv -f "$STORYBOOK_TMP" "$BIN_DIR/storybook"
if ! ln -s "releases/$TARGET_NAME" "$CURRENT_TMP"; then
    fail SB_INSTALL_SWITCH_FAILED "could not prepare the release switch; the previous installation is unchanged"
fi
if ! "$PYTHON" -c 'import os,sys; os.replace(sys.argv[1], sys.argv[2])' \
    "$CURRENT_TMP" "$INSTALL_ROOT/current"; then
    fail SB_INSTALL_SWITCH_FAILED "could not activate the new release; the previous installation is unchanged"
fi

printf 'Installed Storybook %s.\n' "$VERSION"
case :$PATH: in
    *:"$BIN_DIR":*) ;;
    *)
        shell_name=$(basename "${SHELL:-sh}")
        case $shell_name in
            fish) printf 'Add book to PATH: fish_add_path %s\n' "$BIN_DIR" ;;
            *) printf "Add book to PATH: export PATH=\"%s:\$PATH\"\n" "$BIN_DIR" ;;
        esac
        ;;
esac

if [ "$RUN_INIT" -eq 1 ]; then
    if [ -t 0 ] && [ -t 1 ]; then
        printf 'Run onboarding now? [Y/n] '
        read -r answer || answer=n
        case $answer in
            n|N|no|NO) printf 'Next: book init\n' ;;
            *) STORYBOOK_LAUNCHER="$BIN_DIR/book" "$BIN_DIR/book" init ;;
        esac
    else
        printf 'Next: book init\n'
    fi
fi
