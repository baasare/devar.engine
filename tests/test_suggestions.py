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
from devar_engine.references import SpokenReferenceTracker, reference_suggest
from devar_engine.semantic import SemanticBibleIndex, hybrid_suggest


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
    {
        "book": "Job",
        "chapter": 5,
        "verse": 7,
        "text": "Yet man is born to trouble, as the sparks fly upward.",
    },
    {
        "book": "Job",
        "chapter": 38,
        "verse": 33,
        "text": "Do you know the ordinances of the heavens? Can you set their dominion over the earth?",
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


class TestSpokenReferences(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "nkjv.json"
        _write_bible(path)
        self.bible = BibleIndex(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_resolves_reference_spoken_as_words(self) -> None:
        results = reference_suggest(self.bible, "the book of John chapter three verse sixteen")
        self.assertEqual([result.reference for result in results], ["John 3:16"])

    def test_resolves_compound_numbers_without_chapter_or_verse_cues(self) -> None:
        results = reference_suggest(self.bible, "Job thirty eight thirty three")
        self.assertEqual([result.reference for result in results], ["Job 38:33"])

    def test_carries_book_and_chapter_across_pause_updates(self) -> None:
        tracker = SpokenReferenceTracker(context_ttl=30)
        self.assertEqual(reference_suggest(self.bible, "the book of Job", tracker, now=0), [])
        self.assertEqual(reference_suggest(self.bible, "chapter five", tracker, now=4), [])
        results = reference_suggest(self.bible, "verse seven", tracker, now=12)
        self.assertEqual([result.reference for result in results], ["Job 5:7"])
        self.assertEqual(tracker.last_matches[0].reference, "Job 5:7")

    def test_keeps_both_references_when_preacher_corrects_number(self) -> None:
        tracker = SpokenReferenceTracker()
        reference_suggest(self.bible, "book of John chapter three", tracker, now=0)
        results = reference_suggest(self.bible, "verse sixteen or seventeen", tracker, now=1)
        self.assertEqual([result.reference for result in results], ["John 3:16", "John 3:17"])

    def test_discards_context_after_timeout(self) -> None:
        tracker = SpokenReferenceTracker(context_ttl=5)
        reference_suggest(self.bible, "book of John chapter three", tracker, now=0)
        results = reference_suggest(self.bible, "verse sixteen", tracker, now=6)
        self.assertEqual(results, [])


class _TinyEmbedder:
    identity = "tiny-test-embedder-v1"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            # The sample scripture's semantic concepts: love/world vs shepherd.
            vectors.append([1.0, 0.0] if "love" in lowered or "loved" in lowered else [0.0, 1.0])
        return vectors


class _ProvisionEmbedder:
    identity = "provision-test-embedder-v1"

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            is_query = lowered.startswith("task: search result | query:")
            is_shepherd_passage = "my shepherd" in lowered
            vectors.append([0.0, 1.0] if (is_query and "needs are met" in lowered) or is_shepherd_passage else [1.0, 0.0])
        return vectors


class TestHybridSuggestions(unittest.TestCase):
    def test_precomputes_and_reuses_verse_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            cache_dir = Path(d) / "cache"
            embedder = _TinyEmbedder()
            first = SemanticBibleIndex(BibleIndex(path), embedder, cache_dir=cache_dir, batch_size=2)
            calls_after_build = len(embedder.calls)
            again = SemanticBibleIndex(BibleIndex(path), embedder, cache_dir=cache_dir, batch_size=2)
            self.assertEqual(len(embedder.calls), calls_after_build)
            self.assertEqual(first.vectors.shape, again.vectors.shape)

    def test_hybrid_finds_semantic_and_keyword_matches_and_prioritizes_reference(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            bible = BibleIndex(path)
            embedder = _TinyEmbedder()
            semantic = SemanticBibleIndex(bible, embedder)
            results = hybrid_suggest(
                bible, "book of John chapter three verse sixteen", semantic_index=semantic, top_k=3
            )
            self.assertEqual(results[0].reference, "John 3:16")
            self.assertEqual(len(results), 3)

    def test_hybrid_can_ignore_old_explicit_reference_in_rolling_context(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            bible = BibleIndex(path)

            results = hybrid_suggest(
                bible,
                "John chapter three verse sixteen. The LORD is my shepherd.",
                include_references=False,
                top_k=3,
            )

            self.assertEqual(results[0].reference, "Psalms 23:1")

    def test_semantic_search_finds_a_paraphrase_without_shared_content_words(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nkjv.json"
            _write_bible(path)
            bible = BibleIndex(path)
            semantic = SemanticBibleIndex(bible, _ProvisionEmbedder())

            results = hybrid_suggest(
                bible,
                "my needs are met",
                semantic_index=semantic,
                top_k=3,
            )

            self.assertEqual(results[0].reference, "Psalms 23:1")


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
