"""Resolve spoken Bible references across rolling transcript updates."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .suggestions import BibleIndex, SuggestionResult


_BOOKS = (
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "1 Corinthians", "2 Corinthians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
    "Song of Solomon", "Song of Songs", "Ecclesiastes", "Lamentations", "Deuteronomy",
    "Leviticus", "Philippians", "Colossians", "Revelation", "Nehemiah", "Esther",
    "Proverbs", "Jeremiah", "Ezekiel", "Zechariah", "Malachi", "Matthew", "Mark",
    "Luke", "John", "Acts", "Romans", "Galatians", "Ephesians", "Hebrews", "James",
    "Jude", "Genesis", "Exodus", "Numbers", "Joshua", "Judges", "Ruth", "Ezra", "Job",
    "Psalms", "Psalm", "Isaiah", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah",
    "Micah", "Nahum", "Habakkuk", "Haggai", "Zephaniah", "Philemon", "Titus",
    "Song of Solomon",
)

_ORDINALS = {
    "first": "1", "1st": "1", "second": "2", "2nd": "2", "third": "3", "3rd": "3",
}
_NUMBER_WORDS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
_TOKENS = re.compile(r"\d+|[a-z]+", re.IGNORECASE)
_CUE_WORDS = {"chapter", "chap", "verse", "verses", "book", "of", "and", "or", "through", "to"}


def _number(words: list[str]) -> tuple[int | None, int]:
    """Read a small spoken number; return (value, consumed token count)."""
    if not words:
        return None, 0
    first = words[0].lower()
    if first.isdigit():
        return int(first), 1
    if first in _ORDINALS:
        return int(_ORDINALS[first]), 1
    if first not in _NUMBER_WORDS:
        return None, 0
    value = _NUMBER_WORDS[first]
    used = 1
    if value >= 20 and used < len(words) and words[used].lower() in _NUMBER_WORDS:
        unit = _NUMBER_WORDS[words[used].lower()]
        if 1 <= unit <= 9:
            value += unit
            used += 1
    if value == 1 and used < len(words) and words[used].lower() == "hundred":
        value = 100
        used += 1
        if used < len(words) and words[used].lower() == "and":
            used += 1
        tail, consumed = _number(words[used:])
        if tail is not None:
            value += tail
            used += consumed
    return value, used


def _book_patterns() -> list[tuple[list[str], str]]:
    result: list[tuple[list[str], str]] = []
    for book in dict.fromkeys(_BOOKS):
        tokens = book.lower().split()
        result.append((tokens, "Psalms" if book == "Psalm" else book))
    for number_word, digit in _ORDINALS.items():
        if number_word.endswith(("st", "nd", "rd")):
            continue
        for book in _BOOKS:
            if book.startswith(digit + " "):
                result.append(([number_word, *book.split(" ", 1)[1].lower().split()], book))
    result.sort(key=lambda item: len(item[0]), reverse=True)
    return result


_BOOK_PATTERNS = _book_patterns()


@dataclass(frozen=True)
class ReferenceMatch:
    reference: str
    confidence: float = 1.0


class SpokenReferenceTracker:
    """Keeps likely book/chapter context across pauses in live speech.

    Book-only and chapter-only mentions prime the next reference. The context
    expires after ``context_ttl`` so unrelated later numbers are not attached to
    an old book. Pass the rolling transcript on every update.
    """

    def __init__(self, context_ttl: float = 60.0) -> None:
        self.context_ttl = context_ttl
        self.book: str | None = None
        self.chapter: int | None = None
        self.updated_at = 0.0
        self.last_matches: list[ReferenceMatch] = []

    def reset(self) -> None:
        self.book = None
        self.chapter = None
        self.updated_at = 0.0
        self.last_matches = []

    def observe(self, transcript: str, now: float | None = None) -> list[ReferenceMatch]:
        now = time.monotonic() if now is None else now
        if self.book and now - self.updated_at > self.context_ttl:
            self.reset()

        tokens = [match.group(0).lower() for match in _TOKENS.finditer(transcript)]
        found: list[ReferenceMatch] = []
        index = 0
        while index < len(tokens):
            book_match = self._book_at(tokens, index)
            if book_match:
                book, consumed, explicit_book_phrase = book_match
                next_index = index + consumed
                # A bare common word such as "job" is not enough. Accept a book
                # when introduced as a book or followed by a reference cue/number.
                # A chapter and verse spoken without cue words can each take
                # two tokens (for example, "thirty eight thirty three").
                # Keep enough lookahead to parse both complete numbers.
                following = tokens[next_index:next_index + 8]
                has_reference_cue = any(word in {"chapter", "chap", "verse", "verses"} for word in following)
                number, _ = _number(following)
                if explicit_book_phrase or has_reference_cue or number is not None:
                    if book != self.book:
                        self.book, self.chapter = book, None
                        self.updated_at = now
                    index = next_index
                    # "John three sixteen" is a common reference form without
                    # the words chapter/verse.
                    if number is not None and not has_reference_cue:
                        chapter, used = _number(following)
                        verse, _ = _number(following[used:])
                        if chapter and verse:
                            self.chapter = chapter
                            self.updated_at = now
                            found.extend(self._verse_matches(verse, tokens, next_index + used, now))
                    continue

            word = tokens[index]
            if word in {"chapter", "chap"}:
                chapter, used = _number(tokens[index + 1:])
                if chapter is not None and self.book:
                    if chapter != self.chapter:
                        self.chapter = chapter
                        self.updated_at = now
                    index += 1 + used
                    continue
            if word in {"verse", "verses"} and self.book and self.chapter is not None:
                verse, used = _number(tokens[index + 1:])
                if verse is not None:
                    found.extend(self._verse_matches(verse, tokens, index + 1 + used, now))
                    index += 1 + used
                    continue
            index += 1

        # Explicit references should remain visible if later partial transcript
        # updates repeat the same words. Keep only unique references in order.
        unique: dict[str, ReferenceMatch] = {item.reference: item for item in found}
        if unique:
            self.last_matches = list(unique.values())
            return self.last_matches
        self.last_matches = []
        return []

    def _book_at(self, tokens: list[str], index: int) -> tuple[str, int, bool] | None:
        explicit = index + 2 < len(tokens) and tokens[index:index + 2] == ["book", "of"]
        start = index + 2 if explicit else index
        for pattern, canonical in _BOOK_PATTERNS:
            if tokens[start:start + len(pattern)] == pattern:
                return canonical, (start - index) + len(pattern), explicit
        return None

    def _verse_matches(self, verse: int, tokens: list[str], after_verse: int, now: float) -> list[ReferenceMatch]:
        if self.book is None or self.chapter is None:
            return []
        candidates = [verse]
        if after_verse < len(tokens) and tokens[after_verse] == "or":
            alternate, _ = _number(tokens[after_verse + 1:])
            if alternate is not None:
                candidates.append(alternate)
        results = []
        for candidate in candidates:
            reference = f"{self.book} {self.chapter}:{candidate}"
            results.append(ReferenceMatch(reference))
        self.updated_at = now
        return results


def reference_suggest(
    bible: BibleIndex,
    transcript: str,
    tracker: SpokenReferenceTracker | None = None,
    now: float | None = None,
) -> list[SuggestionResult]:
    """Resolve explicit or fragmented spoken references to exact Bible verses."""
    tracker = tracker or SpokenReferenceTracker()
    matches = tracker.observe(transcript, now=now)
    def key(reference: str) -> str:
        value = re.sub(r"[^a-z0-9:]+", "", reference.casefold())
        value = value.replace("psalms", "psalm").replace("songofsongs", "songofsolomon")
        return value

    by_reference = {key(verse.reference): verse for verse in bible.all_verses()}
    results = []
    for match in matches:
        verse = by_reference.get(key(match.reference))
        if verse:
            results.append(SuggestionResult(reference=verse.reference, verses=[verse]))
    return results
