"""Devar Engine: local transcription and Bible verse suggestion helpers."""

from .audio import AudioSettings, RecorderError, pcm_to_wav_bytes, wav_bytes_to_pcm
from .suggestions import BibleIndex, BibleVerse, SuggestionResult, keyword_suggest
from .references import SpokenReferenceTracker, reference_suggest
from .semantic import SemanticBibleIndex, TextEmbedder, hybrid_suggest
from .transcription import (
    AudioSegment,
    MoonshineTranscriber,
    TranscribedSegment,
    TranscriptionError,
    transcribe_pcm_bytes,
    transcribe_wav_bytes,
)

__all__ = [
    "AudioSegment",
    "AudioSettings",
    "BibleIndex",
    "BibleVerse",
    "MoonshineTranscriber",
    "RecorderError",
    "SuggestionResult",
    "SemanticBibleIndex",
    "SpokenReferenceTracker",
    "TextEmbedder",
    "TranscribedSegment",
    "TranscriptionError",
    "keyword_suggest",
    "hybrid_suggest",
    "pcm_to_wav_bytes",
    "transcribe_pcm_bytes",
    "transcribe_wav_bytes",
    "reference_suggest",
    "wav_bytes_to_pcm",
]
