# parakeet-cli

Local transcription CLI around `nvidia/parakeet-tdt-0.6b-v3` with:

- Rust command-line frontend
- Python NeMo backend
- All caches and runtime files kept under this project folder
- Terms library pipeline for domain vocabulary ingestion and typo cleanup

## Install

```bash
cd /root/.parakeet
chmod +x scripts/install.sh scripts/parakeet scripts/transcribe-call.sh scripts/batch-transcribe.sh scripts/terms scripts/terms-sync

# smaller footprint, CPU runtime (recommended default)
./scripts/install.sh --cpu

# or CUDA runtime
./scripts/install.sh --cuda
```

Install command launchers (direct bash calls):

```bash
/root/.parakeet/scripts/install-links
```

This installs:
- `/usr/local/bin/parakeet`
- `/usr/local/bin/parakeetd`
- `/usr/local/bin/parakeet-terms`
- `/usr/local/bin/parakeet-terms-sync`

## Usage

Print transcript to stdout:

```bash
parakeet \
  --input /path/to/audio.wav
```

Machine-readable output for Rust/webhook integration:

```bash
parakeet \
  --input /path/to/audio.wav \
  --emit json
```

Low-latency mode for repeated transcriptions:

```bash
# start persistent model process once
parakeetd start

# normal parakeet calls will auto-use daemon socket when available
parakeet --input /path/to/audio.wav --emit json

# stop when done
parakeetd stop
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

Sync terms library from channels and rebuild ASR vocabulary:

```bash
/root/.parakeet/scripts/terms-sync
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
- `--no-daemon`: force one-shot backend (disable persistent daemon socket)

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

## Terms Library (MVP)

Goal: improve domain-word recognition quality over time.

Storage under `/root/.parakeet/terms`:
- `sources.json`: channel ingestion config (wacli, telegram, openclaw, gmail placeholder)
- `manual.txt`: curated canonical terms
- `library.json`: learned terms with counts/channel stats (generated)
- `vocab.auto.txt`: generated from learned terms (generated)
- `vocab.txt`: merged final vocabulary used by transcriber (generated)

Core commands:

```bash
# Ingest from configured channels and build vocab
/root/.parakeet/scripts/terms-sync

# Ingest one message text payload (for webhook handlers)
/root/.parakeet/scripts/terms ingest-text --channel telegram --text \"new term from message\"

# Ingest webhook/message body from stdin
cat /tmp/message.txt | /root/.parakeet/scripts/terms ingest-stdin --channel telegram

# Ingest one file
/root/.parakeet/scripts/terms ingest-file --channel wacli --path /path/to/file.md

# Show top learned terms
/root/.parakeet/scripts/terms stats --top 50

# Build stricter "hard terms" vocabulary (default: max 300, min count 2)
/root/.parakeet/scripts/terms build-vocab --max-terms 300 --min-count 2
```

Auto-usage in transcription:
- If `/root/.parakeet/terms/vocab.txt` exists, `parakeet` automatically applies it.
- If `--vocab` is provided, both vocab sources are merged for the run.

Hard-term selection policy:
- prioritizes names, product terms, acronyms, mixed-case tokens, versions, and short domain phrases
- excludes metadata keys, domains/emails/URLs, and generic frequent lowercase words
- requires recurrence (`--min-count`) unless manually curated in `terms/manual.txt`

Future channels:
- `gmail` is already present in `terms/sources.json` as disabled placeholder.
- Enable and point paths when channel ingestion is added.

## Notes

- If the input is not a common audio extension, backend uses `ffmpeg` to convert it.
- First run downloads model files, cached inside `/root/.parakeet/.cache`.
- Install/run/write artifacts remain inside `/root/.parakeet` unless you pass external `--out` paths.
- Bottleneck: one-shot runs reload the model every call; use `scripts/parakeetd` for low latency.
