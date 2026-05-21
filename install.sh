#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="/usr/local/bin"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_ARGS=(--no-cache-dir)

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "missing required interpreter: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install "${PIP_ARGS[@]}" --upgrade pip setuptools wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  "$VENV_DIR/bin/python" -m pip install "${PIP_ARGS[@]}" "torch==2.6.0" "torchaudio==2.6.0"
else
  "$VENV_DIR/bin/python" -m pip install \
    "${PIP_ARGS[@]}" \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0+cpu" \
    "torchaudio==2.6.0+cpu"
fi

"$VENV_DIR/bin/python" -m pip install "${PIP_ARGS[@]}" -r "$ROOT_DIR/requirements.txt"

cargo build --release --bin parakeet
install -m 0755 "$ROOT_DIR/target/release/parakeet" "$ROOT_DIR/parakeet"

install -d "$BIN_DIR"
ln -sfn "$ROOT_DIR/parakeet" "$BIN_DIR/parakeet"
ln -sfn "$ROOT_DIR/parakeet" "$BIN_DIR/parakeetd"
ln -sfn "$ROOT_DIR/parakeet" "$BIN_DIR/paraket"

echo "installed:"
echo "  $ROOT_DIR/parakeet"
echo "  $BIN_DIR/parakeet"
echo "  $BIN_DIR/parakeetd"
echo "  $BIN_DIR/paraket"
echo "  $VENV_DIR/bin/python"
