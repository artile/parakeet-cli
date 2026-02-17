# parakeet-cli

Local transcription CLI around `nvidia/parakeet-tdt-0.6b-v3` with:

- Rust command-line frontend
- Python NeMo backend
- All caches and runtime files kept under this project folder
- Optional vocabulary correction file for domain terms

## Install

```bash
cd /root/.parakeet
chmod +x scripts/install.sh scripts/parakeet scripts/transcribe-call.sh scripts/batch-transcribe.sh

# smaller footprint, CPU runtime (recommended default)
./scripts/install.sh --cpu

# or CUDA runtime
./scripts/install.sh --cuda
```

## Usage

Print transcript to stdout:

```bash
/root/.parakeet/scripts/parakeet \
  --input /path/to/audio.wav
```

Machine-readable output for Rust/webhook integration:

```bash
/root/.parakeet/scripts/parakeet \
  --input /path/to/audio.wav \
  --emit json
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
- `--emit text|json`: stdout mode (default `text`)

## Helper scripts

Single call-style transcription to markdown:

```bash
/root/.parakeet/scripts/transcribe-call.sh \
  /path/to/call_audio_or_video.mp4 \
  /root/.parakeet/output/call_001.md \
  /root/.parakeet/vocab.example.txt
```

Batch folder transcription:

```bash
/root/.parakeet/scripts/batch-transcribe.sh \
  /path/to/inbox_media \
  /root/.parakeet/output \
  /root/.parakeet/vocab.example.txt
```

Webhook CLI style invocation (stdout JSON):

```bash
/root/.parakeet/scripts/parakeet \
  --input /tmp/call.m4a \
  --vocab /root/.parakeet/vocab.example.txt \
  --emit json
```

## Notes

- If the input is not a common audio extension, backend uses `ffmpeg` to convert it.
- First run downloads model files, cached inside `/root/.parakeet/.cache`.
- Install/run/write artifacts remain inside `/root/.parakeet` unless you pass external `--out` paths.
