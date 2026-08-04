#!/bin/sh
# Build and verify the two fixed-name assets consumed by install.sh.

set -eu

PROGRAM=storybook-release
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=${1:-$ROOT/dist/release}
PYTHON=${PYTHON:-python3}
TEMP_DIR=

fail() {
    printf '%s: error: %s\n' "$PROGRAM" "$*" >&2
    exit 1
}

cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT HUP INT TERM

command -v "$PYTHON" >/dev/null 2>&1 || fail "Python is required"
"$PYTHON" -m build --help >/dev/null 2>&1 || \
    fail "the Python 'build' package is required (python -m pip install build)"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(CDPATH='' cd -- "$OUTPUT_DIR" && pwd)
TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/storybook-release.XXXXXX") || \
    fail "could not create a temporary directory"

"$PYTHON" -m build --sdist --outdir "$TEMP_DIR/dist" "$ROOT"
set -- "$TEMP_DIR"/dist/storybook-*.tar.gz
[ "$#" -eq 1 ] && [ -f "$1" ] || fail "expected exactly one Storybook source archive"
cp "$1" "$TEMP_DIR/storybook.tar.gz"

"$PYTHON" - "$TEMP_DIR/storybook.tar.gz" "$TEMP_DIR/storybook.tar.gz.sha256" <<'PY'
import hashlib
import pathlib
import sys

archive = pathlib.Path(sys.argv[1])
checksum = pathlib.Path(sys.argv[2])
checksum.write_text(
    f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  storybook.tar.gz\n",
    encoding="ascii",
)
PY

"$PYTHON" -m venv "$TEMP_DIR/verify"
"$TEMP_DIR/verify/bin/pip" install --disable-pip-version-check \
    "$TEMP_DIR/storybook.tar.gz" >/dev/null
"$TEMP_DIR/verify/bin/book" --help >/dev/null
"$TEMP_DIR/verify/bin/storybook" --help >/dev/null

mv "$TEMP_DIR/storybook.tar.gz" "$OUTPUT_DIR/storybook.tar.gz"
mv "$TEMP_DIR/storybook.tar.gz.sha256" "$OUTPUT_DIR/storybook.tar.gz.sha256"
printf 'Verified release assets:\n  %s\n  %s\n' \
    "$OUTPUT_DIR/storybook.tar.gz" "$OUTPUT_DIR/storybook.tar.gz.sha256"
