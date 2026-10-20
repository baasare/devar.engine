"""Cached semantic Bible retrieval and lexical/semantic rank fusion.

The model runtime is supplied by the application. This keeps the shared engine
small and lets desktop applications choose an on-device EmbeddingGemma runner.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .references import SpokenReferenceTracker, reference_suggest
from .suggestions import BibleIndex, SuggestionResult, keyword_suggest


QUERY_PREFIX = "task: search result | query: "


class TextEmbedder(Protocol):
    """Minimal interface required from a local embedding model."""

    @property
    def identity(self) -> str: ...

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SemanticBibleIndex:
    """Precompute verse vectors once and reuse them for live transcript queries."""

    def __init__(
        self,
        bible: BibleIndex,
        embedder: TextEmbedder,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        progress: Callable[[float], None] | None = None,
    ) -> None:
        self.bible = bible
        self.embedder = embedder
        self.verses = bible.all_verses()
        self.identity = str(getattr(embedder, "identity", type(embedder).__qualname__))
        self.cache_key = self._fingerprint()
        self.vectors = self._load_or_build(cache_dir, batch_size, progress)

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(self.identity.encode("utf-8"))
        for verse in self.verses:
            digest.update(f"{verse.book}\0{verse.chapter}\0{verse.verse}\0{verse.text}\n".encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _document(verse) -> str:
        return f"title: {verse.reference} | text: {verse.text}"

    def _load_or_build(
        self,
        cache_dir: Path | None,
        batch_size: int,
        progress: Callable[[float], None] | None,
    ) -> np.ndarray:
        cache_path = cache_dir / f"verses-{self.cache_key}.npz" if cache_dir else None
        if cache_path and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as saved:
                    vectors = saved["vectors"]
                    if str(saved["key"].item()) == self.cache_key and vectors.shape[0] == len(self.verses):
                        return np.asarray(vectors, dtype=np.float16)
            except (OSError, ValueError, KeyError):
                pass

        chunks: list[np.ndarray] = []
        step = max(1, batch_size)
        for start in range(0, len(self.verses), step):
            docs = [self._document(verse) for verse in self.verses[start:start + step]]
            raw = np.asarray(self.embedder.embed_texts(docs), dtype=np.float32)
            if raw.ndim != 2 or raw.shape[0] != len(docs) or raw.shape[1] == 0:
                raise ValueError("Embedder returned an invalid document embedding matrix")
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            if np.any(norms == 0) or not np.isfinite(raw).all():
                raise ValueError("Embedder returned an empty or non-finite verse embedding")
            chunks.append((raw / norms).astype(np.float16))
            if progress:
                progress(min((start + len(docs)) / max(len(self.verses), 1), 1.0))
        vectors = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float16)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix="verses-", suffix=".npz", dir=cache_path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.savez(stream, key=np.asarray(self.cache_key), vectors=vectors)
                os.replace(temporary, cache_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return vectors

    def search(self, query: str, limit: int = 50) -> list[SuggestionResult]:
        if not query.strip() or not len(self.verses):
            return []
        query_text = QUERY_PREFIX + query.strip()
        raw = np.asarray(self.embedder.embed_texts([query_text]), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] != self.vectors.shape[1]:
            raise ValueError("Embedder returned an invalid query embedding")
        query_vector = raw[0]
        norm = float(np.linalg.norm(query_vector))
        if not norm or not np.isfinite(query_vector).all():
            return []
        scores = self.vectors.astype(np.float32) @ (query_vector / norm)
        count = min(max(limit, 0), len(scores))
        if count == 0:
            return []
        # Partial selection avoids sorting every verse when only a few suggestions are needed.
        candidate_ids = np.argpartition(scores, -count)[-count:]
        ranked_ids = candidate_ids[np.argsort(scores[candidate_ids])[::-1]]
        return [
            SuggestionResult(reference=self.verses[int(idx)].reference, verses=[self.verses[int(idx)]])
            for idx in ranked_ids
        ]


def hybrid_suggest(
    bible: BibleIndex,
    query: str,
    *,
    semantic_index: SemanticBibleIndex | None = None,
    reference_tracker: SpokenReferenceTracker | None = None,
    include_references: bool = True,
    top_k: int = 5,
    candidate_k: int = 50,
    rrf_k: int = 60,
    now: float | None = None,
) -> list[SuggestionResult]:
    """Combine direct-reference lookup, BM25 words, and semantic retrieval.

    Exact spoken references lead when resolved. Otherwise reciprocal-rank
    fusion combines BM25 and cosine rankings without assuming their raw scores
    share a scale. If semantic inference is unavailable, BM25 remains usable.
    """
    if top_k <= 0:
        return []
    direct = (
        reference_suggest(bible, query, tracker=reference_tracker, now=now)
        if include_references else []
    )
    lexical = keyword_suggest(bible, query, top_k=max(candidate_k, top_k))
    semantic: list[SuggestionResult] = []
    if semantic_index is not None:
        semantic = semantic_index.search(query, limit=max(candidate_k, top_k))

    # An exact reference is decisive. Keep a spoken correction such as "7 or 6"
    # in order and use the remaining slots for contextual matches.
    direct_keys = {result.reference.casefold() for result in direct}
    fused: dict[str, tuple[float, SuggestionResult]] = {}
    for ranked in (lexical, semantic):
        for rank, result in enumerate(ranked, start=1):
            key = result.reference.casefold()
            score, _ = fused.get(key, (0.0, result))
            fused[key] = (score + 1.0 / (rrf_k + rank), result)
    ordered = [result for result in direct]
    remaining = sorted(
        (value for key, value in fused.items() if key not in direct_keys),
        key=lambda item: item[0], reverse=True,
    )
    ordered.extend(result for _, result in remaining)
    return ordered[:top_k]
