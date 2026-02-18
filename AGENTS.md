# AGENTS

## Purpose

`parakeet` is the single Rust CLI binary for local transcription and daemon control.

Project root: `/root/.parakeet`

## Install

Run once:
- `/root/.parakeet/install.sh`

This builds `target/release/parakeet`, copies it to:
- `/root/.parakeet/parakeet`

And installs symlinks:
- `/usr/local/bin/parakeet` -> `/root/.parakeet/parakeet`
- `/usr/local/bin/parakeetd` -> `/root/.parakeet/parakeet`

`parakeetd` works via executable-name aliasing (`argv[0]` => `daemon` mode).

## Command Surface

Transcription:
- `parakeet --input <audio> [flags]`
- `parakeet transcribe --input <audio> [flags]`

Daemon:
- `parakeet daemon start|stop|status|logs`
- `parakeetd start|stop|status|logs`

## Main Components

- `src/main.rs`
- `python/parakeet_backend.py`
- `python/terms_lib.py`

## Integration Contract

Related project: `/root/.webhook`

Webhook flow expects machine-readable CLI output:
- `parakeet transcribe ... --emit json`
- fallback compatibility: `parakeet ... --emit json`

## Documentation Policy

Operational/source-of-truth instructions are maintained in `AGENTS.md`.
