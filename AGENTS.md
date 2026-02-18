# AGENTS

## Purpose

`parakeet-cli` is a local transcription engine around NVIDIA Parakeet (`nvidia/parakeet-tdt-0.6b-v3`) with:
- Rust CLI frontend (`parakeet-cli`)
- Python inference backend (NeMo)
- Optional persistent daemon mode for low-latency repeated calls
- Terms library + generated vocabulary for domain-specific recognition improvements

Project root: `/root/.parakeet`

Installed launchers:
- `parakeet` -> `/root/.parakeet/scripts/parakeet`
- `parakeetd` -> `/root/.parakeet/scripts/parakeetd`
- `parakeet-terms` -> `/root/.parakeet/scripts/terms`
- `parakeet-terms-sync` -> `/root/.parakeet/scripts/terms-sync`

## Main Components

- `src/main.rs`
  - CLI entrypoint (`scripts/parakeet` wrapper)
  - Builds backend request JSON
  - Merges vocab sources (`terms/vocab.txt` + optional `--vocab`) into `tmp/merged_vocab.txt`
  - Uses daemon socket when available (unless `--no-daemon`)

- `python/parakeet_backend.py`
  - Loads/runs Parakeet model via NeMo
  - Supports one-shot mode (`--json`) and daemon service mode (`--serve`)
  - Emits transcript + timing metrics JSON

- `scripts/parakeet`
  - Normal CLI wrapper

- `scripts/parakeetd`
  - Daemon lifecycle wrapper: `start|stop|restart|status|logs`
  - Keeps model loaded in memory to remove per-request model load overhead

- `python/terms_lib.py`
  - Terms ingestion/build pipeline
  - Produces `terms/vocab.txt` for hard domain terms

- `scripts/terms`, `scripts/terms-sync`
  - Terms management wrappers

## Transcription Flow

1. User/app calls:
   - `/root/.parakeet/scripts/parakeet --input <audio> [flags]`
2. Rust CLI prepares request and vocab merge.
3. Rust CLI tries daemon socket (`/root/.parakeet/tmp/parakeet.sock`) unless `--no-daemon`.
4. If daemon not available, Rust CLI runs one-shot Python backend.
5. Python backend:
   - Normalizes input (ffmpeg if required)
   - Runs NeMo Parakeet transcription
   - Applies vocabulary correction rules
   - Returns JSON (`transcript`, metadata, timing metrics)

## Performance Notes

Primary bottleneck in one-shot mode is model load time.

- One-shot: each call loads model
- Daemon: model stays resident, only inference per request

Use daemon for production/high-throughput:

```bash
parakeetd start
parakeet --input /path/audio.wav --emit json
parakeetd stop
```

## Terms Library (Quality Improvement)

Folder: `/root/.parakeet/terms`

- `sources.json`: channel sources (wacli, telegram, openclaw, gmail placeholder)
- `manual.txt`: manually curated critical terms
- `library.json`: learned term stats (generated)
- `vocab.auto.txt`: generated auto vocab (generated)
- `vocab.txt`: merged final vocab used by transcriber (generated)

### Update Terms

```bash
parakeet-terms-sync
```

### Ingest from Runtime Message Streams

```bash
cat /tmp/message.txt | parakeet-terms ingest-stdin --channel telegram
```

### Build Hard-Term Vocab Manually

```bash
parakeet-terms build-vocab --max-terms 300 --min-count 2
```

## Integration with webhook-cli

Related project: `/root/.webhook-cli`

`fatom transcribe-latest` integration uses Parakeet as transcription engine.

Command:

```bash
cd /root/.webhook-cli
cargo run --release --bin fatom -- --config /root/.webhook-cli/config.toml \
  transcribe-latest \
  --output-dir /root/.parakeet/fathom-test \
  --parakeet-bin /usr/local/bin/parakeet
```

What it does:
1. Fetches latest Fathom meeting/media URL.
2. Downloads/extracts raw WAV with ffmpeg.
3. Calls Parakeet CLI for transcript.
4. Fetches Fathom speaker segments.
5. Produces hybrid speaker transcript:
   - Speaker names/timestamps from Fathom
   - Text from Parakeet

Expected artifacts in run folder:
- `latest_fathom_raw.wav`
- `latest_fathom_transcript.md`
- `latest_fathom_transcript_speakers.md`

## Output/Artifacts

- Runtime output: `/root/.parakeet/output`
- Temp files: `/root/.parakeet/tmp`
- Cached models/libs: `/root/.parakeet/.cache`
- Test runs from webhook flow: `/root/.parakeet/fathom-test`

## Operational Guidance

- Keep `scripts/parakeetd` running for best latency under repeated requests.
- Refresh terms regularly (`terms-sync`) to improve domain-name/product-term recall.
- Maintain `terms/manual.txt` for high-priority canonical terms.
- Use `--emit json` in automation/webhook pipelines.

## Troubleshooting

- Backend/NeMo issues:
  - check daemon logs: `/root/.parakeet/scripts/parakeetd logs`
- Missing model env:
  - reinstall: `/root/.parakeet/scripts/install.sh --cpu` (or `--cuda`)
- Slow requests:
  - verify daemon is running (`parakeetd status`)
  - if not, requests fall back to one-shot and incur model load cost
