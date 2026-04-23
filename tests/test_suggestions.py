"""Tests for the local BM25 verse suggestion pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from devar_engine.suggestions import (
    BibleVerse,
    BibleIndex,
    SuggestionCycle,
    SuggestionResult,
    _tokenize,
    keyword_suggest,
    read_recent_context,
    run_loop,
    run_once,
)


SAMPLE_VERSES = [
    {
        "book": "John",
        "chapter": 3,
        "verse": 16,
        "text": "For God so loved the world that He gave His only begotten Son.",
    },
    {
        "book": "John",
        "chapter": 3,
        "verse": 17,
        "text": "For God did not send His Son into the world to condemn the world.",
    },
    {
        "book": "Romans",
        "chapter": 8,
        "verse": 28,
        "text": "All things work together for good to those who love God.",
    },
    {
        "book": "Philippians",
        "chapter": 4,
        "verse": 6,
        "text": "Be anxious for nothing, but in everything by prayer and supplication.",
    },
    {
        "book": "Philippians",
        "chapter": 4,
        "verse": 7,
        "text": "The peace of God will guard your hearts and minds.",
    },
    {
        "book": "Psalms",
        "chapter": 23,
        "verse": 1,
        "text": "The LORD is my shepherd; I shall not want.",
    },
]


def _write_bible(path: Path) -> None:
    path.write_text(json.dumps(SAMPLE_VERSES, ensure_ascii=False), encoding="utf-8")


class TestTokenize(unittest.TestCase):
    def test_stems_and_removes_stopwords(self) -> None:
        tokens = _tokenize("healing healed healer with there")

        self.assertIn("heal", tokens)
        self.assertIn("healer", tokens)
        self.assertNotIn("with", tokens)
        self.assertNotIn("there", tokens)


class TestBibleIndex(unittest.TestCase):
    def test_loads_verses(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)

            bible = BibleIndex(path)

            self.assertEqual(len(bible), len(SAMPLE_VERSES))
            self.assertEqual(bible.all_verses()[0].reference, "John 3:16")

    def test_ignores_incomplete_items(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            data = SAMPLE_VERSES + [{"book": "Missing Text", "chapter": 1, "verse": 1}]
            path.write_text(json.dumps(data), encoding="utf-8")

            bible = BibleIndex(path)

            self.assertEqual(len(bible), len(SAMPLE_VERSES))

    def test_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            BibleIndex(Path("/does/not/exist.json"))

    def test_raises_on_non_array_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            path.write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                BibleIndex(path)


class TestSuggestionResult(unittest.TestCase):
    def test_display_text_joins_verse_text(self) -> None:
        result = SuggestionResult(
            reference="John 3:16-17",
            verses=[
                BibleVerse("John", 3, 16, "First verse."),
                BibleVerse("John", 3, 17, "Second verse."),
            ],
        )

        self.assertEqual(result.display_text, "First verse. Second verse.")


class TestKeywordSuggest(unittest.TestCase):
    def test_finds_relevant_verse(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            bible = BibleIndex(path)

            results = keyword_suggest(bible, "God loved the world", top_k=3)

            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0].reference, "John 3:16")

    def test_honors_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            bible = BibleIndex(path)

            results = keyword_suggest(bible, "God world love", top_k=2)

            self.assertLessEqual(len(results), 2)

    def test_empty_query_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            bible = BibleIndex(path)

            self.assertEqual(keyword_suggest(bible, ""), [])


class TestReadRecentContext(unittest.TestCase):
    def test_reads_jsonl_by_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            path.write_text(
                '{"timestamp":"00:00 - 00:10","text":"early content"}\n'
                '{"timestamp":"01:20 - 01:30","text":"recent content"}\n',
                encoding="utf-8",
            )

            text, timestamp_range = read_recent_context(path, last_seconds=30)

            self.assertEqual(text, "recent content")
            self.assertEqual(timestamp_range, "01:20 - 01:30")

    def test_ignores_empty_jsonl_entries(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            path.write_text(
                '{"timestamp":"00:00 - 00:05","text":""}\n'
                '{"timestamp":"00:05 - 00:10","text":"usable"}\n',
                encoding="utf-8",
            )

            text, timestamp_range = read_recent_context(path, last_seconds=15)

            self.assertEqual(text, "usable")
            self.assertEqual(timestamp_range, "00:05 - 00:10")

    def test_falls_back_to_chars_for_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.txt"
            path.write_text("0123456789", encoding="utf-8")

            text, timestamp_range = read_recent_context(path, last_chars=4, last_seconds=90)

            self.assertEqual(text, "6789")
            self.assertIsNone(timestamp_range)

    def test_returns_empty_for_missing_file(self) -> None:
        text, timestamp_range = read_recent_context(Path("/no/such/file.jsonl"))

        self.assertEqual(text, "")
        self.assertIsNone(timestamp_range)


class TestRunHelpers(unittest.TestCase):
    def test_run_once_returns_none_when_no_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            bible = BibleIndex(path)

            result = run_once(Path(d) / "missing.jsonl", bible)

            self.assertIsNone(result)

    def test_run_once_returns_cycle_with_results(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bpath = Path(d) / "nkjv.json"
            tpath = Path(d) / "t.txt"
            _write_bible(bpath)
            tpath.write_text("God loved the world and gave His Son.", encoding="utf-8")
            bible = BibleIndex(bpath)

            cycle = run_once(tpath, bible, top_k=2)

            self.assertIsInstance(cycle, SuggestionCycle)
            assert cycle is not None
            self.assertEqual(cycle.context_text, "God loved the world and gave His Son.")
            self.assertLessEqual(len(cycle.results), 2)

    def test_run_loop_runs_one_cycle_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bpath = Path(d) / "nkjv.json"
            tpath = Path(d) / "t.txt"
            _write_bible(bpath)
            tpath.write_text("God loved the world", encoding="utf-8")
            bible = BibleIndex(bpath)

            cycles = run_loop(tpath, bible, interval_seconds=0.01, max_cycles=1)

            self.assertEqual(cycles, 1)


if __name__ == "__main__":
    unittest.main()
