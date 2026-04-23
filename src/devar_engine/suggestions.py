"""Live Bible verse suggestion from a local Bible JSON file.

Watches a growing transcript file (written by transcribe.py), extracts the
most recent sermon context, scores verses from the selected Bible JSON file,
and prints the top matches with full verse text.

Prerequisites
-------------
- Bible JSON file placed at data/nkjv.json or passed with --bible
- rank-bm25 and snowballstemmer installed:
    pip install rank-bm25 snowballstemmer

Usage
-----
    # One-shot from a saved transcript:
    devar-engine-suggest once --transcript-file transcripts/live.jsonl

    # Live polling loop (pair with transcribe.py in a second terminal):
    devar-engine-suggest loop --transcript-file transcripts/live.jsonl

    # Ad-hoc query:
    devar-engine-suggest query --text "Heal the sick, raise the dead"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi
from snowballstemmer import stemmer as _SnowballStemmer

_stemmer = _SnowballStemmer("english")

from .audio import (
    DEFAULT_BIBLE_PATH,
    DEFAULT_TRANSCRIPT_OUTPUT,
)


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TOP_K = 5
DEFAULT_CONTEXT_CHARS = 500         # fallback when timestamps unavailable
DEFAULT_CONTEXT_SECONDS = 15.0      # primary window: last 15 s of audio
DEFAULT_INTERVAL_SECONDS = 10.0

STOPWORDS = frozenset({
    "about", "after", "again", "also", "because", "been", "being",
    "from", "have", "into", "just", "like", "more", "over", "that",
    "their", "them", "then", "there", "these", "they", "this",
    "through", "unto", "were", "what", "when", "where", "which",
    "while", "with", "would", "your",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip stopwords, and stem each token."""
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return [_stemmer.stemWord(t) for t in tokens if t not in STOPWORDS]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BibleVerse:
    book: str
    chapter: int
    verse: int
    text: str

    @property
    def reference(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"


@dataclass
class SuggestionResult:
    reference: str           # canonical reference, e.g. "Matthew 10:8"
    verses: list[BibleVerse]

    @property
    def display_text(self) -> str:
        return " ".join(v.text for v in self.verses)


@dataclass(frozen=True)
class SuggestionCycle:
    timestamp: float
    context_chars: int
    context_hash: str
    results: list[SuggestionResult]
    context_text: str = ""
    timestamp_range: str | None = None


# ── BibleIndex ────────────────────────────────────────────────────────────────

class BibleIndex:
    """Loads a Bible JSON file and builds an index for fast verse search.

    Expected JSON format — a flat list of verse objects::

        [
          {"book": "John", "chapter": 3, "verse": 16,
           "text": "For God so loved the world..."},
          ...
        ]

    A translation field is accepted but not required.
    """

    def __init__(self, path: Path) -> None:
        self._index: dict[tuple[str, int, int], BibleVerse] = {}
        self._all: list[BibleVerse] = []
        self._load(path)
        self._bm25 = BM25Okapi([_tokenize(v.text) for v in self._all])

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Bible JSON file not found: {path}\n"
                f"Place your licensed Bible JSON at {DEFAULT_BIBLE_PATH} — see README."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Bible JSON must be a JSON array of verse objects")
        for item in data:
            if not isinstance(item, dict):
                continue
            book = str(item.get("book", "")).strip()
            chapter = item.get("chapter")
            verse_num = item.get("verse")
            text = str(item.get("text", "")).strip()
            if not book or chapter is None or verse_num is None or not text:
                continue
            v = BibleVerse(book=book, chapter=int(chapter), verse=int(verse_num), text=text)
            key = (_book_key(book), int(chapter), int(verse_num))
            self._index[key] = v
            self._all.append(v)

    def all_verses(self) -> list[BibleVerse]:
        return self._all

    def __len__(self) -> int:
        return len(self._all)


# Backward-compatible alias for older callers.
NkjvBible = BibleIndex


def _book_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ── BM25 verse search ─────────────────────────────────────────────────────────

def keyword_suggest(bible: BibleIndex, query: str, top_k: int = DEFAULT_TOP_K) -> list[SuggestionResult]:
    """Score all verses using BM25 and return the top-K matches.

    BM25 normalises for verse length and applies IDF weighting so rare,
    meaningful terms (e.g. "leprosy") score much higher than common ones.
    Tokens are stemmed so "healing", "healed", and "heal" all match.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bible._bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results: list[SuggestionResult] = []
    for idx, score in ranked:
        if score <= 0:
            break
        verse = bible.all_verses()[idx]
        results.append(SuggestionResult(reference=verse.reference, verses=[verse]))
        if len(results) >= top_k:
            break
    return results


# ── Transcript reading ────────────────────────────────────────────────────────

def _parse_time_to_seconds(s: str) -> float | None:
    parts = s.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return None


def read_recent_context(
    path: Path,
    last_chars: int = DEFAULT_CONTEXT_CHARS,
    last_seconds: float = DEFAULT_CONTEXT_SECONDS,
) -> tuple[str, str | None]:
    """Return (text, timestamp_range) for the most recent portion of a transcript.

    For JSONL files produced by transcribe.py, selects entries whose end
    timestamp falls within the last ``last_seconds`` of the recording.
    Falls back to the last ``last_chars`` characters for plain-text files or
    when timestamps cannot be parsed. timestamp_range is e.g. "00:28 - 00:53".
    """
    if not path.exists():
        return "", None

    if path.suffix.lower() == ".jsonl":
        text, ts_range = _read_jsonl_by_seconds(path, last_seconds)
        if text:
            return text.strip(), ts_range

    return path.read_text(encoding="utf-8")[-last_chars:].strip(), None


def _read_jsonl_by_seconds(path: Path, last_seconds: float) -> tuple[str, str | None]:
    entries: list[tuple[float | None, str, str]] = []  # (end_sec, text, raw_ts)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        text = obj.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        ts = obj.get("timestamp", "")
        end_sec: float | None = None
        if isinstance(ts, str) and " - " in ts:
            end_sec = _parse_time_to_seconds(ts.split(" - ")[1].strip())
        entries.append((end_sec, text.strip(), ts if isinstance(ts, str) else ""))

    if not entries:
        return "", None

    timed = [(t, txt, ts) for t, txt, ts in entries if t is not None]
    if timed:
        max_t = max(t for t, _, _ in timed)
        cutoff = max_t - last_seconds
        recent = [(t, txt, ts) for t, txt, ts in timed if t >= cutoff]
        if recent:
            first_ts = recent[0][2]
            last_ts = recent[-1][2]
            start_label = first_ts.split(" - ")[0].strip() if " - " in first_ts else ""
            end_label = last_ts.split(" - ")[1].strip() if " - " in last_ts else ""
            ts_range = f"{start_label} - {end_label}" if start_label and end_label else None
            return "\n".join(txt for _, txt, _ in recent), ts_range

    return "\n".join(txt for _, txt, _ in entries), None


# ── Run helpers ───────────────────────────────────────────────────────────────

def _context_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _print_cycle(cycle: SuggestionCycle) -> None:
    wall = time.strftime("%H:%M:%S", time.localtime(cycle.timestamp))
    ts_label = f"  audio {cycle.timestamp_range}" if cycle.timestamp_range else f"  {cycle.context_chars} chars"
    print(f"\n[{wall}]{ts_label}", flush=True)
    if cycle.context_text:
        snippet = cycle.context_text.replace("\n", " ")[:520]
        if len(cycle.context_text) > 520:
            snippet += "..."
        print(f"  Context: \"{snippet}\"", flush=True)
    print(f"  {len(cycle.results)} suggestion(s):", flush=True)
    if not cycle.results:
        print("  No suggestions found.", flush=True)
        return
    for i, r in enumerate(cycle.results, 1):
        preview = r.display_text[:560].replace("\n", " ")
        if len(r.display_text) > 560:
            preview += "..."
        print(f"  {i}. {r.reference}\n     {preview}", flush=True)


def run_once(
    transcript_file: Path,
    bible: BibleIndex,
    last_chars: int = DEFAULT_CONTEXT_CHARS,
    last_seconds: float = DEFAULT_CONTEXT_SECONDS,
    top_k: int = DEFAULT_TOP_K,
) -> SuggestionCycle | None:
    context, ts_range = read_recent_context(transcript_file, last_chars=last_chars, last_seconds=last_seconds)
    if not context:
        print(f"No transcript text found in {transcript_file}", flush=True)
        return None
    results = keyword_suggest(bible, context, top_k=top_k)
    cycle = SuggestionCycle(
        timestamp=time.time(),
        context_chars=len(context),
        context_hash=_context_hash(context),
        results=results,
        context_text=context,
        timestamp_range=ts_range,
    )
    _print_cycle(cycle)
    return cycle


def run_loop(
    transcript_file: Path,
    bible: BibleIndex,
    last_chars: int = DEFAULT_CONTEXT_CHARS,
    last_seconds: float = DEFAULT_CONTEXT_SECONDS,
    top_k: int = DEFAULT_TOP_K,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    max_cycles: int | None = None,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    print(
        f"Watching {transcript_file} every {interval_seconds:g}s "
        f"(last {last_seconds:g}s of audio / {last_chars} chars fallback)",
        flush=True,
    )
    seen_hash: str | None = None
    cycles = 0
    try:
        while True:
            context, ts_range = read_recent_context(transcript_file, last_chars=last_chars, last_seconds=last_seconds)
            h = _context_hash(context) if context else None
            if context and h != seen_hash:
                results = keyword_suggest(bible, context, top_k=top_k)
                cycle = SuggestionCycle(
                    timestamp=time.time(),
                    context_chars=len(context),
                    context_hash=h or "",
                    results=results,
                    context_text=context,
                    timestamp_range=ts_range,
                )
                _print_cycle(cycle)
                seen_hash = h
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
            elif not context:
                wall = time.strftime("%H:%M:%S")
                print(f"[{wall}] Waiting for transcript…", flush=True)
            else:
                wall = time.strftime("%H:%M:%S")
                print(f"[{wall}] Transcript unchanged — skipping.", flush=True)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    return cycles


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Suggest Bible verses from a live sermon transcript.")
    p.add_argument("--bible", type=Path, default=DEFAULT_BIBLE_PATH,
                   help=f"path to a Bible JSON file (default: {DEFAULT_BIBLE_PATH})")

    sub = p.add_subparsers(dest="command", required=True)

    # once
    once = sub.add_parser("once", help="run one suggestion cycle and exit")
    once.add_argument("--bible", type=Path, default=argparse.SUPPRESS, help="path to a Bible JSON file")
    _add_runtime_args(once)

    # loop
    loop = sub.add_parser("loop", help="watch transcript and suggest on a polling loop")
    loop.add_argument("--bible", type=Path, default=argparse.SUPPRESS, help="path to a Bible JSON file")
    _add_runtime_args(loop)
    loop.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    loop.add_argument("--max-cycles", type=int, help="stop after N cycles (useful for testing)")

    # query
    query = sub.add_parser("query", help="suggest verses for a single text query")
    query.add_argument("--bible", type=Path, default=argparse.SUPPRESS, help="path to a Bible JSON file")
    query.add_argument("--text", required=True)
    query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    return p


def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--transcript-file", type=Path, default=DEFAULT_TRANSCRIPT_OUTPUT)
    p.add_argument("--last-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    p.add_argument("--last-seconds", type=float, default=DEFAULT_CONTEXT_SECONDS)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        bible = BibleIndex(args.bible)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "once":
            run_once(
                transcript_file=args.transcript_file,
                bible=bible,
                last_chars=args.last_chars,
                last_seconds=args.last_seconds,
                top_k=args.top_k,
            )
        elif args.command == "loop":
            run_loop(
                transcript_file=args.transcript_file,
                bible=bible,
                last_chars=args.last_chars,
                last_seconds=args.last_seconds,
                top_k=args.top_k,
                interval_seconds=args.interval_seconds,
                max_cycles=getattr(args, "max_cycles", None),
            )
        elif args.command == "query":
            results = keyword_suggest(bible, args.text, top_k=args.top_k)
            cycle = SuggestionCycle(
                timestamp=time.time(),
                context_chars=len(args.text),
                context_hash=_context_hash(args.text),
                results=results,
                context_text=args.text,
            )
            _print_cycle(cycle)
        else:
            print(f"error: unknown command: {args.command}", file=sys.stderr)
            return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
