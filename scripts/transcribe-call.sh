#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARAKEET="$ROOT_DIR/scripts/parakeet"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input-audio-or-video> [output-md] [vocab-file]" >&2
  exit 1
fi

INPUT="$1"
OUT="${2:-}"
VOCAB="${3:-}"

if [[ -z "$OUT" ]]; then
  base="$(basename "$INPUT")"
  stem="${base%.*}"
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT="$ROOT_DIR/output/${stem}_${ts}.md"
fi

args=(--input "$INPUT" --format md --out "$OUT")
if [[ -n "$VOCAB" ]]; then
  args+=(--vocab "$VOCAB")
fi

"$PARAKEET" "${args[@]}" --emit json
