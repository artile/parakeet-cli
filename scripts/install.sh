#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p .cache/hf .cache/torch .cache/nemo .cache/pip output tmp

export PARAKEET_HOME="$ROOT_DIR"
export HF_HOME="$ROOT_DIR/.cache/hf"
export TRANSFORMERS_CACHE="$ROOT_DIR/.cache/hf"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export NEMO_HOME="$ROOT_DIR/.cache/nemo"
export PIP_CACHE_DIR="$ROOT_DIR/.cache/pip"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

cargo build --release

echo "Installed Parakeet CLI in $ROOT_DIR"
echo "Binary: $ROOT_DIR/target/release/parakeet-cli"
