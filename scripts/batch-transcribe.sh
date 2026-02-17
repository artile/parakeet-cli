#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARAKEET="$ROOT_DIR/scripts/parakeet"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input-dir> <output-dir> [vocab-file]" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
VOCAB="${3:-}"

mkdir -p "$OUTPUT_DIR"

find "$INPUT_DIR" -type f \
  \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.flac" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" \) \
  | while read -r file; do
      base="$(basename "$file")"
      stem="${base%.*}"
      out="$OUTPUT_DIR/${stem}.md"
      args=(--input "$file" --format md --out "$out")
      if [[ -n "$VOCAB" ]]; then
        args+=(--vocab "$VOCAB")
      fi
      "$PARAKEET" "${args[@]}" --emit json
    done
