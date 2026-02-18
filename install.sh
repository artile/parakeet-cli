#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="/usr/local/bin"

cd "$ROOT_DIR"
cargo build --release --bin parakeet
install -m 0755 "$ROOT_DIR/target/release/parakeet" "$ROOT_DIR/parakeet"

install -d "$BIN_DIR"
ln -sfn "$ROOT_DIR/parakeet" "$BIN_DIR/parakeet"
ln -sfn "$ROOT_DIR/parakeet" "$BIN_DIR/parakeetd"

echo "installed:"
echo "  $ROOT_DIR/parakeet"
echo "  $BIN_DIR/parakeet"
echo "  $BIN_DIR/parakeetd"
