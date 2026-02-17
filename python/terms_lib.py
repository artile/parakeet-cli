#!/usr/bin/env python3
import argparse
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz import fuzz

WORD_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9][A-Za-zА-Яа-яІіЇїЄєҐґ0-9_'’.-]{2,}")
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "you", "your", "are", "was", "were",
    "what", "when", "where", "will", "would", "could", "should", "into", "about", "there", "their", "then",
    "also", "just", "like", "very", "more", "some", "than", "here", "they", "them", "our", "out", "all",
    "що", "як", "це", "так", "але", "або", "для", "про", "його", "вона", "вони", "ми", "ви", "ти", "та",
    "мене", "тобі", "дуже", "тут", "коли", "тому", "якщо", "щоб", "й", "і", "таки", "type", "json", "http",
    "none", "null", "true", "false"
}
TEXT_EXTS = {".txt", ".md", ".json", ".log", ".csv", ".yaml", ".yml"}
SKIP_DIRS = {".git", "target", "node_modules", ".venv", "__pycache__", "media", ".cache"}
MAX_FILES_PER_DIR_SCAN = 8000


@dataclass
class TermStats:
    term: str
    count: int
    channels: dict[str, int]
    last_seen: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "count": self.count,
            "channels": self.channels,
            "last_seen": self.last_seen,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parakeet terms library manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    add_text = sub.add_parser("ingest-text", help="Ingest terms from one text payload")
    add_text.add_argument("--channel", required=True)
    add_text.add_argument("--text", required=True)

    add_file = sub.add_parser("ingest-file", help="Ingest terms from one file")
    add_file.add_argument("--channel", required=True)
    add_file.add_argument("--path", required=True)

    add_stdin = sub.add_parser("ingest-stdin", help="Ingest terms from stdin text stream")
    add_stdin.add_argument("--channel", required=True)

    auto = sub.add_parser("ingest-auto", help="Ingest terms from configured channel sources")
    auto.add_argument("--config", default="/root/.parakeet/terms/sources.json")

    build = sub.add_parser("build-vocab", help="Build vocabulary files for ASR")
    build.add_argument("--max-terms", type=int, default=5000)

    stats = sub.add_parser("stats", help="Show terms library stats")
    stats.add_argument("--top", type=int, default=30)

    sub.add_parser("rebuild", help="Run ingest-auto + build-vocab")
    return p.parse_args()


def root_dir() -> Path:
    return Path(os.environ.get("PARAKEET_HOME", "/root/.parakeet")).resolve()


def terms_dir() -> Path:
    p = root_dir() / "terms"
    p.mkdir(parents=True, exist_ok=True)
    return p


def lib_path() -> Path:
    return terms_dir() / "library.json"


def manual_path() -> Path:
    return terms_dir() / "manual.txt"


def auto_vocab_path() -> Path:
    return terms_dir() / "vocab.auto.txt"


def merged_vocab_path() -> Path:
    return terms_dir() / "vocab.txt"


def load_library() -> dict[str, TermStats]:
    p = lib_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, TermStats] = {}
    for k, v in raw.items():
        out[k] = TermStats(
            term=v["term"],
            count=int(v.get("count", 0)),
            channels={str(ch): int(cnt) for ch, cnt in v.get("channels", {}).items()},
            last_seen=float(v.get("last_seen", 0)),
        )
    return out


def save_library(lib: dict[str, TermStats]) -> None:
    payload = {k: v.to_dict() for k, v in lib.items()}
    lib_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_token(tok: str) -> str:
    cleaned = tok.strip(" _-.,:;!?()[]{}\"'`“”‘’")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def key_for(tok: str) -> str:
    return normalize_token(tok).lower().replace("’", "'")


def looks_useful(tok: str) -> bool:
    t = normalize_token(tok)
    if len(t) < 3:
        return False
    k = key_for(t)
    if not k or k in STOPWORDS:
        return False
    if t.isdigit():
        return False
    if re.fullmatch(r"[0-9_.-]+", t):
        return False
    if "@" in t or t.startswith("http"):
        return False
    if re.search(r"\.(com|net|org|us|io)$", t.lower()):
        return False
    if "_" in t and t.lower() == t:
        return False
    return True


def looks_vocab_candidate(tok: str) -> bool:
    t = normalize_token(tok)
    if not looks_useful(t):
        return False
    k = key_for(t)
    if k in {"display_name", "matched_calendar_invitee_email", "speaker_name", "recording_id"}:
        return False
    if len(t) < 4 and not t.isupper():
        return False
    if t.count("-") + t.count("_") > 2:
        return False
    return True


def extract_terms(text: str) -> list[str]:
    candidates = []
    for m in WORD_RE.finditer(text):
        tok = normalize_token(m.group(0))
        if looks_useful(tok):
            candidates.append(tok)

    # Capture likely product/domain names with separators.
    for m in re.finditer(r"\b[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+\b", text):
        tok = normalize_token(m.group(0))
        if looks_useful(tok):
            candidates.append(tok)
    return candidates


def nearest_existing_key(new_key: str, existing: Iterable[str]) -> str | None:
    best_key = None
    best_score = 0
    for ex in existing:
        if abs(len(ex) - len(new_key)) > 2:
            continue
        score = fuzz.ratio(new_key, ex)
        if score >= 94 and score > best_score:
            best_score = score
            best_key = ex
    return best_key


def add_terms(lib: dict[str, TermStats], terms: list[str], channel: str) -> int:
    now = time.time()
    added = 0
    keys = set(lib.keys())

    for raw in terms:
        tok = normalize_token(raw)
        if not looks_useful(tok):
            continue

        k = key_for(tok)
        if k not in keys:
            merge_key = nearest_existing_key(k, keys)
            if merge_key is not None:
                k = merge_key

        entry = lib.get(k)
        if entry is None:
            entry = TermStats(term=tok, count=0, channels={}, last_seen=now)
            lib[k] = entry
            keys.add(k)
            added += 1

        entry.count += 1
        entry.channels[channel] = entry.channels.get(channel, 0) + 1
        entry.last_seen = now

        # Prefer canonical spelling if this variant has more uppercase/special signal.
        if signal_score(tok) > signal_score(entry.term):
            entry.term = tok

    return added


def signal_score(term: str) -> int:
    score = 0
    if any(c.isupper() for c in term):
        score += 2
    if any(c.isdigit() for c in term):
        score += 1
    if any(c in "-_/" for c in term):
        score += 1
    if len(term) >= 6:
        score += 1
    return score


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""
    except Exception:
        return ""


def extract_json_values(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            out.extend(extract_json_values(item))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(extract_json_values(v))
    return out


def ingest_file(lib: dict[str, TermStats], path: Path, channel: str) -> int:
    if not path.exists() or not path.is_file():
        return 0
    if path.suffix.lower() not in TEXT_EXTS and path.stat().st_size > 5_000_000:
        return 0
    if path.suffix.lower() == ".json":
        text = read_text_file(path)
        if not text:
            return 0
        try:
            payload = json.loads(text)
            terms: list[str] = []
            for v in extract_json_values(payload):
                terms.extend(extract_terms(v))
            return add_terms(lib, terms, channel)
        except Exception:
            pass

    text = read_text_file(path)
    if not text:
        return 0
    return add_terms(lib, extract_terms(text), channel)


def ingest_text_dir(lib: dict[str, TermStats], folder: Path, channel: str) -> int:
    if not folder.exists():
        return 0
    total = 0
    seen = 0
    for p in folder.rglob("*"):
        if seen >= MAX_FILES_PER_DIR_SCAN:
            break
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            seen += 1
            total += ingest_file(lib, p, channel)
    return total


def ingest_sqlite(lib: dict[str, TermStats], db_path: Path, channel: str) -> int:
    if not db_path.exists() or not db_path.is_file():
        return 0
    total = 0
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            cols = cur.execute(f"PRAGMA table_info('{table}')").fetchall()
            text_cols = [c[1] for c in cols if str(c[2]).upper() in {"TEXT", "VARCHAR", "CHAR", "CLOB"}]
            if not text_cols:
                continue
            for col in text_cols:
                try:
                    q = f"SELECT {col} FROM '{table}' WHERE {col} IS NOT NULL LIMIT 20000"
                    for (val,) in cur.execute(q):
                        if not isinstance(val, str) or len(val) < 3:
                            continue
                        total += add_terms(lib, extract_terms(val), channel)
                except Exception:
                    continue
        conn.close()
    except Exception:
        return total
    return total


def build_vocab(lib: dict[str, TermStats], max_terms: int) -> dict[str, Any]:
    manual_terms = []
    if manual_path().exists():
        manual_terms = [
            normalize_token(x) for x in manual_path().read_text(encoding="utf-8").splitlines()
            if normalize_token(x) and not normalize_token(x).startswith("#")
        ]

    ranked = sorted(lib.values(), key=lambda x: (x.count, x.last_seen), reverse=True)
    auto_terms = [t.term for t in ranked if looks_vocab_candidate(t.term)][:max_terms]

    auto_vocab_path().write_text("\n".join(auto_terms) + ("\n" if auto_terms else ""), encoding="utf-8")

    merged = []
    seen = set()
    for t in manual_terms + auto_terms:
        k = key_for(t)
        if k and k not in seen:
            seen.add(k)
            merged.append(t)

    merged_vocab_path().write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    return {
        "manual_terms": len(manual_terms),
        "auto_terms": len(auto_terms),
        "merged_terms": len(merged),
        "auto_vocab": str(auto_vocab_path()),
        "vocab": str(merged_vocab_path()),
    }


def ingest_auto(lib: dict[str, TermStats], cfg_path: Path) -> dict[str, int]:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    added_by_channel: dict[str, int] = {}
    for ch in cfg.get("channels", []):
        if not ch.get("enabled", True):
            continue
        name = str(ch.get("name", "unknown"))
        added = 0
        for p in ch.get("text_paths", []):
            added += ingest_text_dir(lib, Path(p), name)
        for p in ch.get("sqlite_paths", []):
            added += ingest_sqlite(lib, Path(p), name)
        added_by_channel[name] = added
    return added_by_channel


def cmd_stats(lib: dict[str, TermStats], top: int) -> dict[str, Any]:
    ranked = sorted(lib.values(), key=lambda x: (x.count, x.last_seen), reverse=True)
    return {
        "terms_total": len(lib),
        "top": [
            {"term": t.term, "count": t.count, "channels": t.channels}
            for t in ranked[:top]
        ],
    }


def main() -> int:
    args = parse_args()
    lib = load_library()

    if args.cmd == "ingest-text":
        added = add_terms(lib, extract_terms(args.text), args.channel)
        save_library(lib)
        print(json.dumps({"event": "terms_ingest_text", "added": added, "total_terms": len(lib)}, ensure_ascii=False))
        return 0

    if args.cmd == "ingest-file":
        added = ingest_file(lib, Path(args.path), args.channel)
        save_library(lib)
        print(json.dumps({"event": "terms_ingest_file", "added": added, "total_terms": len(lib)}, ensure_ascii=False))
        return 0

    if args.cmd == "ingest-stdin":
        payload = ""
        try:
            payload = os.read(0, 10_000_000).decode("utf-8", errors="ignore")
        except Exception:
            payload = ""
        added = add_terms(lib, extract_terms(payload), args.channel)
        save_library(lib)
        print(json.dumps({"event": "terms_ingest_stdin", "added": added, "total_terms": len(lib)}, ensure_ascii=False))
        return 0

    if args.cmd == "ingest-auto":
        added_map = ingest_auto(lib, Path(args.config))
        save_library(lib)
        print(json.dumps({"event": "terms_ingest_auto", "added_by_channel": added_map, "total_terms": len(lib)}, ensure_ascii=False))
        return 0

    if args.cmd == "build-vocab":
        result = build_vocab(lib, args.max_terms)
        print(json.dumps({"event": "terms_build_vocab", **result, "total_terms": len(lib)}, ensure_ascii=False))
        return 0

    if args.cmd == "rebuild":
        added_map = ingest_auto(lib, terms_dir() / "sources.json")
        save_library(lib)
        built = build_vocab(lib, 5000)
        print(json.dumps({"event": "terms_rebuild", "added_by_channel": added_map, **built, "total_terms": len(lib)}, ensure_ascii=False))
        return 0

    if args.cmd == "stats":
        print(json.dumps(cmd_stats(lib, args.top), ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
