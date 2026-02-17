#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from rapidfuzz import fuzz, process

import nemo.collections.asr as nemo_asr


def patch_sampler_compat() -> None:
    """
    Lhotse in current NeMo path passes `data_source` into torch Sampler.__init__.
    Newer torch variants may reject that argument, so we ignore it safely.
    """
    from torch.utils.data import Sampler

    signature = inspect.signature(Sampler.__init__)
    if "data_source" in signature.parameters:
        return

    original_init = Sampler.__init__

    def _compat_init(self, *args, **kwargs):
        kwargs.pop("data_source", None)
        return original_init(self)

    Sampler.__init__ = _compat_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="JSON request from Rust CLI")
    return parser.parse_args()


def read_request(raw_json: str) -> dict[str, Any]:
    try:
        req = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON request: {exc}") from exc
    required = ["input", "model", "device", "format", "timestamps", "fuzzy_vocab", "verbose"]
    for key in required:
        if key not in req:
            raise RuntimeError(f"missing request key: {key}")
    return req


def ensure_runtime_dirs(parakeet_home: Path) -> None:
    for rel in [".cache/hf", ".cache/torch", ".cache/nemo", ".cache/pip", "output", "tmp"]:
        (parakeet_home / rel).mkdir(parents=True, exist_ok=True)


def normalize_audio(in_path: Path, temp_dir: Path, verbose: bool) -> Path:
    if in_path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
        return in_path

    out_path = temp_dir / f"{in_path.stem}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_path),
    ]
    if verbose:
        print(f"[parakeet] converting input via ffmpeg: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed. install ffmpeg or pass a supported audio file.\n"
            f"{proc.stderr.strip()}"
        )
    return out_path


def pick_device(requested: str) -> str:
    req = requested.lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req in {"cpu", "cuda"}:
        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return req
    raise RuntimeError("invalid --device. allowed: auto|cpu|cuda")


def load_vocab(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise RuntimeError(f"vocab file not found: {path}")
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            terms.append(s)
    return terms


def apply_vocab_rules(text: str, vocab_terms: list[str], fuzzy_enabled: bool) -> str:
    if not vocab_terms:
        return text

    updated = text
    terms_by_lower = {t.lower(): t for t in vocab_terms}

    # exact case normalization first
    for lower_term, canonical_term in terms_by_lower.items():
        pattern = re.compile(rf"(?i)\b{re.escape(lower_term)}\b")
        updated = pattern.sub(canonical_term, updated)

    if not fuzzy_enabled:
        return updated

    words = re.findall(r"\b[\w'-]+\b", updated)
    vocab_words = [t for t in vocab_terms if " " not in t]
    if not vocab_words:
        return updated

    replacements: dict[str, str] = {}
    for word in words:
        if len(word) < 5 or word.lower() in terms_by_lower:
            continue
        best = process.extractOne(word, vocab_words, scorer=fuzz.WRatio, score_cutoff=90)
        if best is None:
            continue
        candidate = best[0]
        if candidate.lower() != word.lower():
            replacements[word] = candidate

    if not replacements:
        return updated

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return replacements.get(token, token)

    return re.sub(r"\b[\w'-]+\b", repl, updated)


def to_markdown(text: str, source: Path, model_name: str, device: str) -> str:
    return (
        f"# Transcript\n\n"
        f"- Source: `{source}`\n"
        f"- Model: `{model_name}`\n"
        f"- Device: `{device}`\n\n"
        f"{text.strip()}\n"
    )


def transcribe(req: dict[str, Any]) -> dict[str, Any]:
    patch_sampler_compat()
    parakeet_home = Path(os.environ.get("PARAKEET_HOME", "/root/.parakeet")).resolve()
    ensure_runtime_dirs(parakeet_home)

    input_path = Path(req["input"]).expanduser().resolve()
    if not input_path.exists():
        raise RuntimeError(f"input file does not exist: {input_path}")

    model_name = req["model"]
    output_path = Path(req["output"]).expanduser().resolve() if req.get("output") else None
    output_format = req["format"]
    timestamps = bool(req["timestamps"])
    fuzzy_vocab = bool(req["fuzzy_vocab"])
    verbose = bool(req["verbose"])
    vocab_path = Path(req["vocab"]).expanduser().resolve() if req.get("vocab") else None
    device = pick_device(req["device"])

    vocab_terms = load_vocab(vocab_path)

    tmp_dir = parakeet_home / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        normalized = normalize_audio(input_path, Path(td), verbose)
        if verbose:
            print(f"[parakeet] loading model: {model_name} on {device}", file=sys.stderr)

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model = model.to(torch.device(device))

        # Support multiple NeMo transcribe() signatures across versions.
        audio_list = [str(normalized)]
        try:
            result = model.transcribe(paths2audio_files=audio_list, batch_size=1, num_workers=0, verbose=False)
        except TypeError:
            result = model.transcribe(audio=audio_list, batch_size=1, num_workers=0, verbose=False)
        if not result:
            raise RuntimeError("empty transcription result")

        first = result[0]
        text = first.text.strip() if hasattr(first, "text") else str(first).strip()
        text = apply_vocab_rules(text, vocab_terms, fuzzy_vocab)

        if timestamps:
            text = f"[timestamps not available in current simple mode]\n{text}"

        final_text = text if output_format == "text" else to_markdown(text, input_path, model_name, device)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final_text, encoding="utf-8")

    return {
        "transcript": final_text,
        "output_path": str(output_path) if output_path else None,
        "source": str(input_path),
        "model": model_name,
        "device": device,
        "format": output_format,
    }


def main() -> int:
    args = parse_args()
    try:
        req = read_request(args.json)
        result = transcribe(req)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
