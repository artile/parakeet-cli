#!/usr/bin/env python3
import argparse
import errno
import gc
import inspect
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from rapidfuzz import fuzz, process


def detect_parakeet_home() -> Path:
    env_home = os.environ.get("PARAKEET_HOME")
    if env_home:
        return Path(env_home).resolve()
    return Path(__file__).resolve().parent.parent


# Keep lhotse-generated ~/.lhotse under parakeet cache, not /root.
PARAKEET_HOME_DEFAULT = detect_parakeet_home()
LHOTSE_HOME = PARAKEET_HOME_DEFAULT / ".cache/home"
LHOTSE_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(LHOTSE_HOME)

PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_SPEAKER_ID_MODEL = "pyannote/embedding"
PYANNOTE_SPEAKER_ID_FALLBACK_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
HF_TOKEN_ENV_VARS = (
    "PYANNOTE_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
)


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
    parser.add_argument("--json", help="JSON request from Rust CLI")
    parser.add_argument("--speaker-enroll-json", help="JSON request for speaker enrollment")
    parser.add_argument("--speaker-identify-json", help="JSON request for standalone speaker identification")
    parser.add_argument("--serve", action="store_true", help="Run persistent backend daemon")
    parser.add_argument("--socket-path", default=str(PARAKEET_HOME_DEFAULT / "tmp/parakeet.sock"))
    parser.add_argument("--service-model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--service-device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.serve and not args.json and not args.speaker_enroll_json and not args.speaker_identify_json:
        parser.error("--json, --speaker-enroll-json, or --speaker-identify-json is required unless --serve is used")
    return args


def read_request(raw_json: str) -> dict[str, Any]:
    try:
        req = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON request: {exc}") from exc
    required = [
        "input",
        "model",
        "device",
        "format",
        "timestamps",
        "fuzzy_vocab",
        "verbose",
        "identify_speakers",
    ]
    for key in required:
        if key not in req:
            raise RuntimeError(f"missing request key: {key}")
    return req


def read_speaker_enroll_request(raw_json: str) -> dict[str, Any]:
    try:
        req = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid speaker enrollment JSON request: {exc}") from exc
    required = ["input", "name", "profile_dir", "device", "verbose"]
    for key in required:
        if key not in req:
            raise RuntimeError(f"missing speaker enrollment request key: {key}")
    return req


def read_speaker_identify_request(raw_json: str) -> dict[str, Any]:
    try:
        req = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid speaker identify JSON request: {exc}") from exc
    required = ["input", "diarization", "profile_dir", "device", "verbose", "format", "timestamps"]
    for key in required:
        if key not in req:
            raise RuntimeError(f"missing speaker identify request key: {key}")
    return req


def ensure_runtime_dirs(parakeet_home: Path) -> None:
    for rel in [
        ".cache/home",
        ".cache/hf",
        ".cache/torch",
        ".cache/nemo",
        ".cache/pip",
        "output",
        "tmp",
    ]:
        (parakeet_home / rel).mkdir(parents=True, exist_ok=True)


def normalize_audio(in_path: Path, temp_dir: Path, verbose: bool, force_wav: bool = False) -> Path:
    if not force_wav and in_path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
        return in_path

    out_path = temp_dir / f"normalized_{in_path.stem}.wav"
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


def to_speaker_markdown(
    segments: list[dict[str, Any]],
    source: Path,
    model_name: str,
    device: str,
    diarization_model: str,
    timestamps: bool,
) -> str:
    body = render_speaker_transcript(segments, timestamps)
    return (
        f"# Transcript\n\n"
        f"- Source: `{source}`\n"
        f"- Model: `{model_name}`\n"
        f"- Device: `{device}`\n"
        f"- Diarization: `{diarization_model}`\n\n"
        f"{body}\n"
    )


def safe_audio_duration_sec(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return wf.getnframes() / float(rate)
    except Exception:
        return None


def transcribe_paths(model: Any, audio_paths: list[str]) -> list[Any]:
    if not audio_paths:
        return []
    batch_size = min(8, len(audio_paths))
    try:
        return model.transcribe(paths2audio_files=audio_paths, batch_size=batch_size, num_workers=0, verbose=False)
    except TypeError:
        return model.transcribe(audio=audio_paths, batch_size=batch_size, num_workers=0, verbose=False)


def extract_transcript_text(item: Any) -> str:
    return item.text.strip() if hasattr(item, "text") else str(item).strip()


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_speaker_transcript(segments: list[dict[str, Any]], timestamps: bool) -> str:
    lines = []
    for segment in segments:
        prefix = f"[{format_timestamp(segment['start_sec'])}] " if timestamps else ""
        lines.append(f"{prefix}{segment['speaker']}: {segment['text']}")
    return "\n".join(lines).strip()


def split_transcript_units(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def align_transcript_to_speakers(
    text: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    units = split_transcript_units(text)
    if not units or not segments:
        return []

    durations = [max(0.1, float(seg["end_sec"]) - float(seg["start_sec"])) for seg in segments]
    total_duration = sum(durations)
    cumulative = []
    running = 0.0
    for duration in durations:
        running += duration
        cumulative.append(running)

    assigned_units: list[list[str]] = [[] for _ in segments]
    for idx, unit in enumerate(units):
        position = ((idx + 0.5) / len(units)) * total_duration
        segment_idx = 0
        while segment_idx < len(cumulative) - 1 and position > cumulative[segment_idx]:
            segment_idx += 1
        assigned_units[segment_idx].append(unit)

    aligned = []
    for segment, unit_group in zip(segments, assigned_units):
        if not unit_group:
            continue
        aligned.append(
            {
                "speaker": segment["speaker"],
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
                "text": " ".join(unit_group).strip(),
            }
        )
    return aligned


def persist_diarization_segments(segments: list[dict[str, Any]], runtime_dir: Path) -> None:
    payload = [
        {
            "speaker": segment["speaker"],
            "start_sec": round(float(segment["start_sec"]), 3),
            "end_sec": round(float(segment["end_sec"]), 3),
        }
        for segment in segments
    ]
    (runtime_dir / "pyannote_segments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def persist_speaker_identification(payload: dict[str, Any], runtime_dir: Path) -> None:
    (runtime_dir / "speaker_identification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def persist_json_output(payload: dict[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def resolve_hf_token() -> str:
    for name in HF_TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token

        cached_token = (get_token() or "").strip()
        if cached_token:
            return cached_token
    except Exception:
        pass
    raise RuntimeError(
        "speaker diarization requires a Hugging Face token via one of: "
        + ", ".join(HF_TOKEN_ENV_VARS)
        + ", or `huggingface-cli login`"
    )


def ensure_diarization_runtime_ready() -> None:
    resolve_hf_token()
    try:
        import torchaudio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for diarization chunking; rerun install.sh") from exc
    try:
        from pyannote.audio import Pipeline  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyannote.audio is not installed; rerun install.sh to add diarization support") from exc


def ensure_embedding_runtime_ready() -> None:
    resolve_hf_token()
    try:
        from pyannote.audio import Inference, Model  # noqa: F401
        from pyannote.core import Segment  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyannote audio embedding support is unavailable; rerun install.sh") from exc


def patch_torch_load_compat() -> None:
    if getattr(torch.load, "_parakeet_weights_only_compat", False):
        return

    original_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    _compat_load._parakeet_weights_only_compat = True
    torch.load = _compat_load


def load_diarization_pipeline(device: str, verbose: bool) -> tuple[Any, float]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("pyannote.audio is not installed; rerun install.sh to add diarization support") from exc

    patch_torch_load_compat()
    token = resolve_hf_token()
    started = time.perf_counter()
    try:
        pipeline = Pipeline.from_pretrained(PYANNOTE_DIARIZATION_MODEL, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(PYANNOTE_DIARIZATION_MODEL, use_auth_token=token)

    if device == "cuda":
        pipeline = pipeline.to(torch.device("cuda"))
    if verbose:
        print(f"[parakeet] loaded diarization pipeline: {PYANNOTE_DIARIZATION_MODEL}", file=sys.stderr)
    return pipeline, time.perf_counter() - started


def embedding_model_candidates() -> list[str]:
    preferred = os.environ.get("PARAKEET_SPEAKER_EMBEDDING_MODEL", PYANNOTE_SPEAKER_ID_MODEL).strip()
    candidates = [preferred]
    if preferred != PYANNOTE_SPEAKER_ID_FALLBACK_MODEL:
        candidates.append(PYANNOTE_SPEAKER_ID_FALLBACK_MODEL)
    return candidates


def load_embedding_inference(device: str, verbose: bool) -> tuple[Any, str, float]:
    from pyannote.audio import Inference, Model

    patch_torch_load_compat()
    token = resolve_hf_token()
    errors: list[str] = []
    started = time.perf_counter()
    for model_name in embedding_model_candidates():
        try:
            try:
                model = Model.from_pretrained(model_name, token=token)
            except TypeError:
                model = Model.from_pretrained(model_name, use_auth_token=token)
            if model is None:
                raise RuntimeError("model returned None")
            inference = Inference(model, window="whole")
            if device == "cuda":
                inference.to(torch.device("cuda"))
            if verbose:
                print(f"[parakeet] loaded speaker embedding model: {model_name}", file=sys.stderr)
            return inference, model_name, time.perf_counter() - started
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "failed to load any speaker embedding model; "
        "accept access if using gated pyannote/embedding or set PARAKEET_SPEAKER_EMBEDDING_MODEL. "
        + " | ".join(errors)
    )


def merge_diarization_turns(annotation: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    merge_gap_sec = 0.35
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start_sec = float(turn.start)
        end_sec = float(turn.end)
        if end_sec <= start_sec:
            continue
        speaker_name = str(speaker).strip() or "SPEAKER_00"
        if (
            merged
            and merged[-1]["speaker"] == speaker_name
            and start_sec - float(merged[-1]["end_sec"]) <= merge_gap_sec
        ):
            merged[-1]["end_sec"] = end_sec
            continue
        merged.append({"speaker": speaker_name, "start_sec": start_sec, "end_sec": end_sec})
    return merged


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "speaker"


def normalize_speaker_label(value: Any) -> str:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return "SPEAKER_00"
    if raw.isdigit():
        return f"SPEAKER_{int(raw):02d}"
    return raw


def embedding_to_vector(embedding: Any) -> list[float]:
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    while isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
        embedding = embedding[0]
    if not isinstance(embedding, list):
        raise RuntimeError("unexpected embedding payload shape")
    return [float(value) for value in embedding]


def l2_normalize(vector: list[float]) -> list[float]:
    tensor = torch.tensor(vector, dtype=torch.float32)
    norm = torch.linalg.vector_norm(tensor).item()
    if norm <= 0:
        raise RuntimeError("embedding norm is zero")
    return (tensor / norm).tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = torch.tensor(vec_a, dtype=torch.float32)
    b = torch.tensor(vec_b, dtype=torch.float32)
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def average_embeddings(vectors: list[list[float]], weights: list[float] | None = None) -> list[float]:
    if not vectors:
        raise RuntimeError("no embeddings to average")
    tensor = torch.tensor(vectors, dtype=torch.float32)
    if weights:
        weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        centroid = (tensor * weight_tensor).sum(dim=0) / weight_tensor.sum()
    else:
        centroid = tensor.mean(dim=0)
    return l2_normalize(centroid.tolist())


def ensure_profile_dir(profile_dir: Path) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)


def load_speaker_profiles(profile_dir: Path) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    if not profile_dir.exists():
        return profiles
    for path in sorted(profile_dir.glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        centroid = profile.get("centroid")
        name = profile.get("name")
        if not name or not isinstance(centroid, list):
            continue
        profiles.append(
            {
                "name": str(name),
                "slug": str(profile.get("slug") or path.stem),
                "path": path,
                "embedding_model": str(profile.get("embedding_model") or ""),
                "centroid": [float(value) for value in centroid],
                "sample_count": int(profile.get("sample_count") or len(profile.get("samples", []))),
            }
        )
    return profiles


def save_speaker_profile(
    profile_dir: Path,
    name: str,
    embedding_model: str,
    embedding: list[float],
    source: Path,
    duration_sec: float,
    start_sec: float | None,
    end_sec: float | None,
) -> dict[str, Any]:
    ensure_profile_dir(profile_dir)
    slug = slugify_name(name)
    path = profile_dir / f"{slug}.json"
    payload: dict[str, Any]
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": 1,
            "name": name,
            "slug": slug,
            "embedding_model": embedding_model,
            "created_at": now_iso(),
            "samples": [],
        }

    if payload.get("embedding_model") and payload["embedding_model"] != embedding_model:
        raise RuntimeError(
            f"profile {name} uses embedding model {payload['embedding_model']} but current model is {embedding_model}"
        )

    samples = list(payload.get("samples") or [])
    samples.append(
        {
            "created_at": now_iso(),
            "source": str(source),
            "duration_sec": round(float(duration_sec), 3),
            "start_sec": round(float(start_sec), 3) if start_sec is not None else None,
            "end_sec": round(float(end_sec), 3) if end_sec is not None else None,
            "embedding": embedding,
        }
    )
    payload["samples"] = samples
    payload["sample_count"] = len(samples)
    payload["embedding_model"] = embedding_model
    payload["updated_at"] = now_iso()
    payload["centroid"] = average_embeddings([sample["embedding"] for sample in samples])
    payload["dimension"] = len(payload["centroid"])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "name": name,
        "profile_path": str(path),
        "sample_count": len(samples),
        "duration_sec": float(duration_sec),
        "embedding_model": embedding_model,
    }


def extract_embedding_for_clip(
    inference: Any,
    audio_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[list[float], float]:
    if start_sec is None and end_sec is None:
        embedding = inference(str(audio_path))
        duration_sec = safe_audio_duration_sec(audio_path) or 0.0
    else:
        from pyannote.core import Segment

        clip_start = max(0.0, float(start_sec or 0.0))
        clip_end = max(clip_start + 0.1, float(end_sec if end_sec is not None else clip_start + 30.0))
        embedding = inference.crop(str(audio_path), Segment(clip_start, clip_end))
        duration_sec = clip_end - clip_start
    return l2_normalize(embedding_to_vector(embedding)), float(duration_sec)


def enroll_speaker(req: dict[str, Any]) -> dict[str, Any]:
    patch_sampler_compat()
    parakeet_home = PARAKEET_HOME_DEFAULT
    ensure_runtime_dirs(parakeet_home)
    ensure_embedding_runtime_ready()

    input_path = Path(req["input"]).expanduser().resolve()
    if not input_path.exists():
        raise RuntimeError(f"input file does not exist: {input_path}")

    profile_dir = Path(req["profile_dir"]).expanduser().resolve()
    requested_device = pick_device(req["device"])
    verbose = bool(req["verbose"])
    start_sec = req.get("start_sec")
    end_sec = req.get("end_sec")

    with tempfile.TemporaryDirectory(dir=parakeet_home / "tmp") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        normalized = normalize_audio(input_path, temp_dir, verbose, force_wav=True)
        inference, embedding_model, _ = load_embedding_inference(requested_device, verbose)
        try:
            embedding, duration_sec = extract_embedding_for_clip(inference, normalized, start_sec, end_sec)
        finally:
            del inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return save_speaker_profile(
        profile_dir=profile_dir,
        name=str(req["name"]).strip(),
        embedding_model=embedding_model,
        embedding=embedding,
        source=input_path,
        duration_sec=duration_sec,
        start_sec=float(start_sec) if start_sec is not None else None,
        end_sec=float(end_sec) if end_sec is not None else None,
    )


def select_segments_for_embedding(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    min_duration_sec = 0.8
    max_segments = 5
    max_total_duration_sec = 45.0
    selected: list[dict[str, Any]] = []
    total_duration = 0.0
    for segment in sorted(segments, key=lambda item: float(item["end_sec"]) - float(item["start_sec"]), reverse=True):
        duration = float(segment["end_sec"]) - float(segment["start_sec"])
        if duration < min_duration_sec:
            continue
        if len(selected) >= max_segments or total_duration >= max_total_duration_sec:
            break
        selected.append(segment)
        total_duration += duration
    return selected


def compute_cluster_embeddings(
    audio_path: Path,
    segments: list[dict[str, Any]],
    inference: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        grouped.setdefault(str(segment["speaker"]), []).append(segment)

    clusters: dict[str, dict[str, Any]] = {}
    for speaker, speaker_segments in grouped.items():
        chosen = select_segments_for_embedding(speaker_segments)
        if not chosen:
            continue
        embeddings: list[list[float]] = []
        weights: list[float] = []
        for segment in chosen:
            emb, duration = extract_embedding_for_clip(
                inference,
                audio_path,
                float(segment["start_sec"]),
                float(segment["end_sec"]),
            )
            embeddings.append(emb)
            weights.append(max(0.1, duration))
        clusters[speaker] = {
            "embedding": average_embeddings(embeddings, weights),
            "used_segments": [
                {
                    "start_sec": round(float(segment["start_sec"]), 3),
                    "end_sec": round(float(segment["end_sec"]), 3),
                }
                for segment in chosen
            ],
            "segment_count": len(chosen),
        }
    return clusters


def identify_speaker_clusters(
    audio_path: Path,
    raw_segments: list[dict[str, Any]],
    speaker_segments: list[dict[str, Any]],
    device: str,
    runtime_dir: Path,
    profile_dir: Path,
    verbose: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    ensure_embedding_runtime_ready()
    profiles = load_speaker_profiles(profile_dir)
    if not profiles:
        raise RuntimeError(f"no speaker profiles found in {profile_dir}")

    started = time.perf_counter()
    inference, embedding_model, load_sec = load_embedding_inference(device, verbose)
    try:
        clusters = compute_cluster_embeddings(audio_path, raw_segments, inference)
    finally:
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    similarity_threshold = float(os.environ.get("PARAKEET_SPEAKER_MATCH_THRESHOLD", "0.72"))
    similarity_margin = float(os.environ.get("PARAKEET_SPEAKER_MATCH_MARGIN", "0.04"))
    assignments: dict[str, dict[str, Any]] = {}
    claimed_profiles: set[str] = set()

    for speaker, cluster in sorted(clusters.items()):
        comparisons = []
        for profile in profiles:
            score = cosine_similarity(cluster["embedding"], profile["centroid"])
            comparisons.append(
                {
                    "name": profile["name"],
                    "slug": profile["slug"],
                    "score": round(score, 4),
                }
            )
        comparisons.sort(key=lambda item: item["score"], reverse=True)
        top = comparisons[0] if comparisons else None
        second_score = comparisons[1]["score"] if len(comparisons) > 1 else -1.0
        matched = (
            top is not None
            and top["score"] >= similarity_threshold
            and (top["score"] - second_score) >= similarity_margin
            and top["slug"] not in claimed_profiles
        )
        if matched:
            claimed_profiles.add(top["slug"])
        assignments[speaker] = {
            "speaker": speaker,
            "matched": bool(matched),
            "display_name": top["name"] if matched else speaker,
            "score": top["score"] if top else None,
            "candidates": comparisons[:3],
            "used_segments": cluster["used_segments"],
        }

    updated_segments = []
    matched_speakers: set[str] = set()
    for segment in speaker_segments:
        original = str(segment["speaker"])
        assignment = assignments.get(original)
        display_name = assignment["display_name"] if assignment else original
        if assignment and assignment["matched"]:
            matched_speakers.add(original)
        updated_segments.append({**segment, "speaker": display_name, "speaker_original": original})

    payload = {
        "profile_dir": str(profile_dir),
        "embedding_model": embedding_model,
        "threshold": similarity_threshold,
        "margin": similarity_margin,
        "assignments": list(assignments.values()),
    }
    persist_speaker_identification(payload, runtime_dir)
    identification_sec = load_sec + (time.perf_counter() - started)
    info = {
        "model": embedding_model,
        "profile_dir": str(profile_dir),
        "matched_speakers": len(matched_speakers),
        "cluster_count": len(assignments),
        "assignments": list(assignments.values()),
    }
    return updated_segments, info, identification_sec


def _segment_time_sec(item: dict[str, Any], sec_keys: tuple[str, ...], ms_keys: tuple[str, ...]) -> float | None:
    for key in sec_keys:
        value = item.get(key)
        if value is not None:
            return float(value)
    for key in ms_keys:
        value = item.get(key)
        if value is not None:
            return float(value) / 1000.0
    return None


def normalize_external_segment(item: dict[str, Any]) -> dict[str, Any] | None:
    start_sec = _segment_time_sec(item, ("start_sec", "start"), ("start_ms",))
    end_sec = _segment_time_sec(item, ("end_sec", "end"), ("end_ms",))
    if start_sec is None or end_sec is None or end_sec <= start_sec:
        return None
    speaker = normalize_speaker_label(
        item.get("speaker")
        or item.get("speaker_id")
        or item.get("speakerId")
        or item.get("speaker_label")
        or item.get("speakerLabel")
    )
    text = item.get("text")
    normalized: dict[str, Any] = {
        "speaker": speaker,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
    }
    if text is not None and str(text).strip():
        normalized["text"] = str(text).strip()
    return normalized


def segments_from_soniox_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    max_gap_sec = 0.8
    current: dict[str, Any] | None = None
    parts: list[str] = []
    ordered = sorted(tokens, key=lambda token: (float(token.get("start_ms") or 0), float(token.get("end_ms") or 0)))
    for token in ordered:
        if token.get("is_audio_event"):
            continue
        start_ms = token.get("start_ms")
        end_ms = token.get("end_ms")
        if start_ms is None or end_ms is None:
            continue
        start_sec = float(start_ms) / 1000.0
        end_sec = float(end_ms) / 1000.0
        if end_sec <= start_sec:
            continue
        speaker = normalize_speaker_label(token.get("speaker"))
        text_piece = str(token.get("text") or "")
        if (
            current is None
            or current["speaker"] != speaker
            or (start_sec - float(current["end_sec"])) > max_gap_sec
        ):
            if current is not None:
                current["text"] = "".join(parts).strip()
                if current["text"]:
                    segments.append(current)
            current = {
                "speaker": speaker,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
            parts = [text_piece]
        else:
            current["end_sec"] = end_sec
            parts.append(text_piece)
    if current is not None:
        current["text"] = "".join(parts).strip()
        if current["text"]:
            segments.append(current)
    return segments


def load_external_speaker_segments(diarization_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = json.loads(diarization_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed reading diarization JSON: {diarization_path}: {exc}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("tokens"), list):
        segments = segments_from_soniox_tokens(payload["tokens"])
        return segments, "soniox_tokens"

    segment_items: list[Any] | None = None
    source_kind = "segments"
    if isinstance(payload, list):
        segment_items = payload
    elif isinstance(payload, dict):
        for key in ("segments", "speaker_segments", "diarization_segments"):
            if isinstance(payload.get(key), list):
                segment_items = payload[key]
                source_kind = key
                break
        if segment_items is None and isinstance(payload.get("diarization"), list):
            segment_items = payload["diarization"]
            source_kind = "diarization"

    if segment_items is None:
        raise RuntimeError(
            "unsupported diarization JSON format; expected a segment list, "
            "an object with segments/speaker_segments, or Soniox tokens"
        )

    segments = []
    for item in segment_items:
        if not isinstance(item, dict):
            continue
        normalized = normalize_external_segment(item)
        if normalized is not None:
            segments.append(normalized)
    if not segments:
        raise RuntimeError(f"no usable diarization segments found in {diarization_path}")
    return segments, source_kind


def to_speaker_identify_markdown(
    segments: list[dict[str, Any]],
    source: Path,
    diarization_source: Path,
    embedding_model: str,
    timestamps: bool,
) -> str:
    body = render_speaker_transcript(segments, timestamps)
    return (
        f"# Speaker Identification\n\n"
        f"- Source: `{source}`\n"
        f"- Diarization input: `{diarization_source}`\n"
        f"- Embedding model: `{embedding_model}`\n\n"
        f"{body}\n"
    )


def identify_speakers_from_external_diarization(req: dict[str, Any]) -> dict[str, Any]:
    patch_sampler_compat()
    parakeet_home = PARAKEET_HOME_DEFAULT
    ensure_runtime_dirs(parakeet_home)
    ensure_embedding_runtime_ready()

    input_path = Path(req["input"]).expanduser().resolve()
    diarization_path = Path(req["diarization"]).expanduser().resolve()
    profile_dir = Path(req["profile_dir"]).expanduser().resolve()
    output_path = Path(req["output"]).expanduser().resolve() if req.get("output") else None
    render_output = Path(req["render_output"]).expanduser().resolve() if req.get("render_output") else None
    work_dir = Path(req["work_dir"]).expanduser().resolve() if req.get("work_dir") else None
    output_format = str(req["format"])
    timestamps = bool(req["timestamps"])
    requested_device = pick_device(req["device"])
    verbose = bool(req["verbose"])

    if not input_path.exists():
        raise RuntimeError(f"input file does not exist: {input_path}")
    if not diarization_path.exists():
        raise RuntimeError(f"diarization file does not exist: {diarization_path}")

    tmp_dir = parakeet_home / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = work_dir
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(dir=tmp_dir)
        runtime_dir = Path(temp_dir_obj.name)

    try:
        normalized = normalize_audio(input_path, runtime_dir, verbose, force_wav=True)
        audio_duration = safe_audio_duration_sec(normalized)
        speaker_segments, diarization_kind = load_external_speaker_segments(diarization_path)
        raw_segments = [
            {
                "speaker": segment["speaker"],
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
            }
            for segment in speaker_segments
        ]
        updated_segments, identification_info, identification_sec = identify_speaker_clusters(
            normalized,
            raw_segments,
            speaker_segments,
            requested_device,
            runtime_dir,
            profile_dir,
            verbose,
        )

        transcript: str | None = None
        if any(segment.get("text") for segment in updated_segments):
            transcript = (
                render_speaker_transcript(updated_segments, timestamps)
                if output_format == "text"
                else to_speaker_identify_markdown(
                    updated_segments,
                    input_path,
                    diarization_path,
                    identification_info["model"],
                    timestamps,
                )
            )
        if render_output is not None:
            if transcript is None:
                raise RuntimeError("render_output requested but diarization input contains no segment text")
            render_output.parent.mkdir(parents=True, exist_ok=True)
            render_output.write_text(transcript, encoding="utf-8")

        response = {
            "input": str(input_path),
            "diarization_source": str(diarization_path),
            "diarization_kind": diarization_kind,
            "output_path": str(output_path) if output_path is not None else None,
            "rendered_path": str(render_output) if render_output is not None else None,
            "transcript": transcript,
            "embedding_model": identification_info["model"],
            "metrics": {
                "identification_sec": identification_sec,
                "audio_sec": audio_duration,
                "cluster_count": identification_info["cluster_count"],
                "matched_speakers": identification_info["matched_speakers"],
                "segment_count": len(updated_segments),
            },
            "assignments": identification_info["assignments"],
            "segments": updated_segments,
        }
        if output_path is not None:
            persist_json_output(response, output_path)
        return response
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def diarize_audio(audio_path: Path, device: str, verbose: bool) -> tuple[list[dict[str, Any]], float, int]:
    pipeline, load_sec = load_diarization_pipeline(device, verbose)
    started = time.perf_counter()
    try:
        annotation = pipeline(str(audio_path))
        segments = merge_diarization_turns(annotation)
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    speakers = {segment["speaker"] for segment in segments}
    diarization_sec = load_sec + (time.perf_counter() - started)
    if verbose:
        print(
            f"[parakeet] diarization segments={len(segments)} speakers={len(speakers)}",
            file=sys.stderr,
        )
    return segments, diarization_sec, len(speakers)


def transcribe_speaker_segments(
    model: Any,
    audio_path: Path,
    segments: list[dict[str, Any]],
    temp_dir: Path,
    vocab_terms: list[str],
    fuzzy_vocab: bool,
) -> list[dict[str, Any]]:
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for diarization chunking; rerun install.sh") from exc

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total_frames = waveform.shape[1]
    padded_segments: list[dict[str, Any]] = []
    chunk_paths: list[str] = []
    min_padding_frames = int(sample_rate * 0.15)

    for idx, segment in enumerate(segments):
        start_frame = max(0, int(segment["start_sec"] * sample_rate))
        end_frame = min(total_frames, max(start_frame + 1, int(segment["end_sec"] * sample_rate)))
        if end_frame - start_frame < min_padding_frames:
            pad = min_padding_frames - (end_frame - start_frame)
            left = pad // 2
            right = pad - left
            start_frame = max(0, start_frame - left)
            end_frame = min(total_frames, end_frame + right)
        chunk = waveform[:, start_frame:end_frame]
        if chunk.numel() == 0:
            continue

        chunk_path = temp_dir / f"speaker_turn_{idx:04d}.wav"
        torchaudio.save(str(chunk_path), chunk, sample_rate)
        chunk_paths.append(str(chunk_path))
        padded_segments.append(
            {
                "speaker": segment["speaker"],
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
            }
        )

    results = transcribe_paths(model, chunk_paths)
    speaker_segments: list[dict[str, Any]] = []
    for segment, result in zip(padded_segments, results):
        text = apply_vocab_rules(extract_transcript_text(result), vocab_terms, fuzzy_vocab)
        if not text:
            continue
        speaker_segments.append({**segment, "text": text})
    return speaker_segments


def load_model(model_name: str, device: str, verbose: bool) -> tuple[Any, str, float]:
    import nemo.collections.asr as nemo_asr

    t0 = time.perf_counter()
    resolved_device = pick_device(device)
    if verbose:
        print(f"[parakeet] loading model: {model_name} on {resolved_device}", file=sys.stderr)

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(torch.device(resolved_device))
    return model, resolved_device, time.perf_counter() - t0


def transcribe(
    req: dict[str, Any],
    preloaded_model: Any | None = None,
    preloaded_model_name: str | None = None,
    preloaded_device: str | None = None,
) -> dict[str, Any]:
    patch_sampler_compat()
    started = time.perf_counter()
    parakeet_home = PARAKEET_HOME_DEFAULT
    ensure_runtime_dirs(parakeet_home)

    input_path = Path(req["input"]).expanduser().resolve()
    if not input_path.exists():
        raise RuntimeError(f"input file does not exist: {input_path}")

    model_name = req["model"]
    output_path = Path(req["output"]).expanduser().resolve() if req.get("output") else None
    work_dir = Path(req["work_dir"]).expanduser().resolve() if req.get("work_dir") else None
    output_format = req["format"]
    timestamps = bool(req["timestamps"])
    diarize = bool(req.get("diarize"))
    identify_speakers = bool(req.get("identify_speakers"))
    fuzzy_vocab = bool(req["fuzzy_vocab"])
    verbose = bool(req["verbose"])
    vocab_path = Path(req["vocab"]).expanduser().resolve() if req.get("vocab") else None
    speaker_profile_dir = (
        Path(req["speaker_profile_dir"]).expanduser().resolve()
        if req.get("speaker_profile_dir")
        else None
    )

    vocab_terms = load_vocab(vocab_path)

    tmp_dir = parakeet_home / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if diarize:
        ensure_diarization_runtime_ready()
    if identify_speakers:
        if not diarize:
            raise RuntimeError("speaker identification requires diarization")
        if speaker_profile_dir is None:
            raise RuntimeError("speaker identification requires speaker_profile_dir")
        ensure_embedding_runtime_ready()

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = work_dir
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(dir=tmp_dir)
        runtime_dir = Path(temp_dir_obj.name)

    try:
        normalized = normalize_audio(input_path, runtime_dir, verbose, force_wav=diarize)
        audio_duration = safe_audio_duration_sec(normalized)

        diarization_sec = None
        speaker_identification_sec = None
        diarization_info = None
        speaker_identification_info = None
        requested_device = pick_device(req["device"])
        if diarize:
            raw_segments, diarization_runtime, speaker_count = diarize_audio(normalized, requested_device, verbose)
            diarization_sec = diarization_runtime
            persist_diarization_segments(raw_segments, runtime_dir)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_load_sec = 0.0
        if (
            not diarize
            and preloaded_model is not None
            and preloaded_model_name == model_name
            and preloaded_device == requested_device
        ):
            model = preloaded_model
            resolved_device = preloaded_device
        else:
            model, resolved_device, model_load_sec = load_model(model_name, req["device"], verbose)

        if diarize:
            infer_start = time.perf_counter()
            result = transcribe_paths(model, [str(normalized)])
            if not result:
                raise RuntimeError("empty transcription result")
            text = apply_vocab_rules(extract_transcript_text(result[0]), vocab_terms, fuzzy_vocab)
            speaker_segments = align_transcript_to_speakers(text, raw_segments)
            if identify_speakers:
                speaker_segments, speaker_identification_info, speaker_identification_sec = identify_speaker_clusters(
                    normalized,
                    raw_segments,
                    speaker_segments,
                    requested_device,
                    runtime_dir,
                    speaker_profile_dir,
                    verbose,
                )
            infer_sec = time.perf_counter() - infer_start
            if not speaker_segments:
                raise RuntimeError("speaker diarization produced no aligned transcript segments")
            final_text = (
                render_speaker_transcript(speaker_segments, timestamps)
                if output_format == "text"
                else to_speaker_markdown(
                    speaker_segments,
                    input_path,
                    model_name,
                    resolved_device,
                    PYANNOTE_DIARIZATION_MODEL,
                    timestamps,
                )
            )
            diarization_info = {
                "model": PYANNOTE_DIARIZATION_MODEL,
                "speaker_count": speaker_count,
                "segment_count": len(speaker_segments),
            }
        else:
            infer_start = time.perf_counter()
            result = transcribe_paths(model, [str(normalized)])
            infer_sec = time.perf_counter() - infer_start

            if not result:
                raise RuntimeError("empty transcription result")

            text = apply_vocab_rules(extract_transcript_text(result[0]), vocab_terms, fuzzy_vocab)

            if timestamps:
                text = f"[timestamps not available in current simple mode]\n{text}"

            final_text = text if output_format == "text" else to_markdown(text, input_path, model_name, resolved_device)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final_text, encoding="utf-8")
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    total_sec = time.perf_counter() - started
    return {
        "transcript": final_text,
        "output_path": str(output_path) if output_path else None,
        "source": str(input_path),
        "model": model_name,
        "device": resolved_device,
        "format": output_format,
        "metrics": {
            "model_load_sec": model_load_sec,
            "inference_sec": infer_sec,
            "total_sec": total_sec,
            "audio_sec": audio_duration,
            "diarization_sec": diarization_sec,
            "speaker_identification_sec": speaker_identification_sec,
            "speaker_segments": diarization_info["segment_count"] if diarization_info else None,
            "identified_speakers": (
                speaker_identification_info["matched_speakers"] if speaker_identification_info else None
            ),
        },
        "diarization": diarization_info,
    }


def serve(socket_path: Path, model_name: str, device: str, verbose: bool) -> int:
    patch_sampler_compat()
    parakeet_home = PARAKEET_HOME_DEFAULT
    ensure_runtime_dirs(parakeet_home)

    model, resolved_device, load_sec = load_model(model_name, device, verbose)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if socket_path.exists():
            socket_path.unlink()
    except Exception:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(16)
    print(
        f"[parakeetd] ready socket={socket_path} model={model_name} device={resolved_device} load_sec={load_sec:.2f}",
        file=sys.stderr,
        flush=True,
    )

    while True:
        conn, _ = server.accept()
        with conn:
            try:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    continue

                req = read_request(data.decode("utf-8", errors="ignore").strip())
                result = transcribe(
                    req,
                    preloaded_model=model,
                    preloaded_model_name=model_name,
                    preloaded_device=resolved_device,
                )
                conn.sendall((json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8"))
            except Exception as exc:
                payload = {"error": str(exc)}
                try:
                    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                except OSError as send_exc:
                    if send_exc.errno not in {errno.EPIPE, errno.ECONNRESET, errno.ENOTCONN}:
                        raise


def main() -> int:
    args = parse_args()
    try:
        if args.serve:
            return serve(Path(args.socket_path), args.service_model, args.service_device, args.verbose)

        if args.speaker_enroll_json:
            req = read_speaker_enroll_request(args.speaker_enroll_json)
            result = enroll_speaker(req)
            print(json.dumps(result, ensure_ascii=False))
            return 0

        if args.speaker_identify_json:
            req = read_speaker_identify_request(args.speaker_identify_json)
            result = identify_speakers_from_external_diarization(req)
            print(json.dumps(result, ensure_ascii=False))
            return 0

        req = read_request(args.json)
        result = transcribe(req)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
