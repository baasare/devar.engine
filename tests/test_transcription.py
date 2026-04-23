"""Tests for the local Moonshine transcription pipeline.

The tests avoid microphone hardware and avoid loading real Moonshine models by
using fake sources and fake transcriber backends.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from devar_engine.audio import AudioSettings
from devar_engine.transcription import (
    AudioSegment,
    MoonshineTranscriber,
    TranscribedSegment,
    _dedupe_overlapping_text,
    _pcm_to_moonshine_audio,
    append_transcript,
    format_timestamp,
    iter_audio_segments,
    live_transcribe,
)


@dataclass
class FakeSource:
    settings: AudioSettings
    payload: bytes
    name: str = "fake"

    def chunks(self, duration_seconds: float | None) -> Iterable[bytes]:
        size = self.settings.chunk_size_bytes
        for i in range(0, len(self.payload), size):
            yield self.payload[i: i + size]


class FakeMoonshineBackend:
    def __init__(self, response: str = "hello moonshine") -> None:
        self._response = response
        self.calls: list[dict] = []

    def transcribe(self, pcm: bytes, settings: AudioSettings) -> str:
        self.calls.append({"pcm": pcm, "settings": settings})
        return self._response


class FakeTranscriber:
    def transcribe(
        self,
        segment: AudioSegment,
        settings: AudioSettings,
        context: str | None = None,
    ) -> str:
        return f"segment {segment.index}"


class TestIterAudioSegments(unittest.TestCase):
    def test_splits_into_correct_segments(self) -> None:
        settings = AudioSettings(sample_rate=10, channels=1, sample_width=2, chunk_seconds=0.2)
        pcm = b"\x01\x00" * 25

        segs = list(iter_audio_segments([pcm], settings, buffer_seconds=1.0))

        self.assertEqual([s.index for s in segs], [1, 2, 3])
        self.assertEqual([len(s.pcm) for s in segs], [20, 20, 10])
        self.assertAlmostEqual(segs[0].start_seconds, 0.0)
        self.assertAlmostEqual(segs[2].end_seconds, 2.5)

    def test_rejects_zero_buffer(self) -> None:
        settings = AudioSettings(sample_rate=10, channels=1, sample_width=2)
        with self.assertRaises(ValueError):
            list(iter_audio_segments([], settings, buffer_seconds=0))

    def test_can_overlap_segments_without_duplicate_recording_audio(self) -> None:
        settings = AudioSettings(sample_rate=10, channels=1, sample_width=2, chunk_seconds=0.2)
        pcm = b"\x01\x00" * 25

        segs = list(iter_audio_segments([pcm], settings, buffer_seconds=1.0, overlap_seconds=0.2))

        self.assertEqual([s.index for s in segs], [1, 2, 3])
        self.assertEqual([len(s.pcm) for s in segs], [20, 20, 18])
        self.assertEqual([len(s.recording_pcm or b"") for s in segs], [20, 16, 14])
        self.assertAlmostEqual(segs[1].start_seconds, 0.8)
        self.assertAlmostEqual(segs[1].end_seconds, 1.8)
        self.assertEqual(sum(len(s.recording_pcm or b"") for s in segs), len(pcm))

    def test_rejects_overlap_at_least_as_large_as_buffer(self) -> None:
        settings = AudioSettings(sample_rate=10, channels=1, sample_width=2)
        with self.assertRaises(ValueError):
            list(iter_audio_segments([], settings, buffer_seconds=1.0, overlap_seconds=1.0))


class TestAppendTranscript(unittest.TestCase):
    def test_writes_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.txt"
            append_transcript(path, TranscribedSegment(1, 5.0, 65.0, "hello world"))
            self.assertEqual(path.read_text(), "[00:05 - 01:05] hello world\n")

    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            append_transcript(path, TranscribedSegment(1, 5.0, 65.0, "hello world"))
            obj = json.loads(path.read_text())
            self.assertEqual(obj, {"timestamp": "00:05 - 01:05", "text": "hello world"})


class TestFormatTimestamp(unittest.TestCase):
    def test_minutes_and_seconds(self) -> None:
        self.assertEqual(format_timestamp(65), "01:05")

    def test_with_hours(self) -> None:
        self.assertEqual(format_timestamp(3661), "01:01:01")


class TestTextDedupe(unittest.TestCase):
    def test_removes_repeated_prefix_from_overlap(self) -> None:
        current = _dedupe_overlapping_text(
            "Today we are talking about grace through faith",
            "grace through faith and not by works",
        )

        self.assertEqual(current, "and not by works")


class TestMoonshineTranscriber(unittest.TestCase):
    def test_sends_pcm_to_moonshine_backend(self) -> None:
        fake_backend = FakeMoonshineBackend("good morning")
        transcriber = MoonshineTranscriber(model="tiny", backend=fake_backend)
        settings = AudioSettings(sample_rate=16_000, channels=1, sample_width=2)
        segment = AudioSegment(index=1, start_seconds=0.0, end_seconds=1.0, pcm=b"\x00\x00" * 10)

        result = transcriber.transcribe(segment, settings)

        self.assertEqual(result, "good morning")
        self.assertEqual(transcriber.model, "moonshine/tiny")
        self.assertEqual(fake_backend.calls[0]["pcm"], segment.pcm)
        self.assertEqual(fake_backend.calls[0]["settings"], settings)

    def test_pcm_conversion_requires_16khz(self) -> None:
        settings = AudioSettings(sample_rate=8_000, channels=1, sample_width=2)
        with self.assertRaises(Exception):
            _pcm_to_moonshine_audio(b"\x00\x00" * 10, settings)

    def test_pcm_conversion_mixes_stereo_to_mono(self) -> None:
        settings = AudioSettings(sample_rate=16_000, channels=2, sample_width=2)
        pcm = (
            (32767).to_bytes(2, "little", signed=True)
            + (0).to_bytes(2, "little", signed=True)
            + (-32768).to_bytes(2, "little", signed=True)
            + (0).to_bytes(2, "little", signed=True)
        )

        audio = _pcm_to_moonshine_audio(pcm, settings)

        self.assertEqual(audio.shape, (2,))
        self.assertAlmostEqual(float(audio[0]), 0.5, places=3)
        self.assertAlmostEqual(float(audio[1]), -0.5, places=3)


class TestLiveTranscribe(unittest.TestCase):
    def test_transcription_preserves_order(self) -> None:
        settings = AudioSettings(sample_rate=10, channels=1, sample_width=2, chunk_seconds=0.2)
        source = FakeSource(settings=settings, payload=b"\x01\x00" * 20)

        with tempfile.TemporaryDirectory() as d:
            transcript_path = Path(d) / "t.txt"
            audio_path = Path(d) / "a.wav"

            segments = live_transcribe(
                source=source,
                duration_seconds=2.0,
                buffer_seconds=1.0,
                transcript_output=transcript_path,
                transcriber=FakeTranscriber(),
                audio_output=audio_path,
                overlap_seconds=0.0,
            )

            self.assertEqual([s.text for s in segments], ["segment 1", "segment 2"])
            self.assertIn("segment 1", transcript_path.read_text())
            with wave.open(str(audio_path), "rb") as wf:
                self.assertEqual(wf.getnframes(), 20)

    def test_context_argument_is_supplied_for_each_segment(self) -> None:
        received_contexts: list[str | None] = []

        class ContextCapturingTranscriber:
            def transcribe(
                self,
                seg: AudioSegment,
                settings: AudioSettings,
                context: str | None = None,
            ) -> str:
                received_contexts.append(context)
                return f"word{seg.index}"

        settings = AudioSettings(sample_rate=10, channels=1, sample_width=2, chunk_seconds=0.2)
        source = FakeSource(settings=settings, payload=b"\x01\x00" * 30)

        with tempfile.TemporaryDirectory() as d:
            live_transcribe(
                source=source,
                duration_seconds=3.0,
                buffer_seconds=1.0,
                transcript_output=Path(d) / "t.jsonl",
                transcriber=ContextCapturingTranscriber(),
                overlap_seconds=0.0,
            )

        self.assertEqual(len(received_contexts), 3)

    def test_overlap_does_not_duplicate_saved_wav_audio(self) -> None:
        settings = AudioSettings(sample_rate=10, channels=1, sample_width=2, chunk_seconds=0.2)
        source = FakeSource(settings=settings, payload=b"\x01\x00" * 30)

        with tempfile.TemporaryDirectory() as d:
            audio_path = Path(d) / "a.wav"
            live_transcribe(
                source=source,
                duration_seconds=3.0,
                buffer_seconds=1.0,
                overlap_seconds=0.2,
                transcript_output=Path(d) / "t.jsonl",
                transcriber=FakeTranscriber(),
                audio_output=audio_path,
            )

            with wave.open(str(audio_path), "rb") as wf:
                self.assertEqual(wf.getnframes(), 30)


if __name__ == "__main__":
    unittest.main()
