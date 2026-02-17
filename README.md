# parakeet-cli

Local transcription CLI around `nvidia/parakeet-tdt-0.6b-v3` with:

- Rust command-line frontend
- Python NeMo backend
- All caches and runtime files kept under this project folder
- Optional vocabulary correction file for domain terms

## Install

```bash
cd /root/.parakeet
chmod +x scripts/install.sh scripts/parakeet
./scripts/install.sh
```

## Usage

Print transcript to stdout:

```bash
/root/.parakeet/scripts/parakeet \
  --input /path/to/audio.wav
```

Write markdown output file:

```bash
/root/.parakeet/scripts/parakeet \
  --input /path/to/audio.m4a \
  --format md \
  --out /root/.parakeet/output/call_001.md
```

Use custom vocabulary:

```bash
/root/.parakeet/scripts/parakeet \
  --input /path/to/audio.wav \
  --vocab /root/.parakeet/vocab.example.txt
```

## Main flags

- `--input`: source audio/video file
- `--out`: optional output file path
- `--format text|md`: output format (default `text`)
- `--model`: default `nvidia/parakeet-tdt-0.6b-v3`
- `--device auto|cpu|cuda`: default `auto`
- `--vocab`: one term per line
- `--no-fuzzy-vocab`: disable fuzzy vocabulary correction
- `--verbose`: print backend diagnostics

## Notes

- If the input is not a common audio extension, backend uses `ffmpeg` to convert it.
- First run downloads model files, cached inside `/root/.parakeet/.cache`.
