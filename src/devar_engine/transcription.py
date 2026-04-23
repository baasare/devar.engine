"""Near real-time audio transcription using local Moonshine ONNX models.

Audio is captured in short buffers, transcribed with Moonshine, and printed
to the terminal as each buffer returns. Capture and transcription run in
parallel so the next buffer records while the current one is being transcribed.

The rolling transcript is written to a JSONL file so suggest_verses.py can
watch it live in a second terminal.

Prerequisites
-------------
- useful-moonshine-onnx installed:  pip install "devar-engine[transcription]"
- numpy installed:                  pip install numpy

Usage
-----
    devar-engine-transcribe --duration 300
    devar-engine-transcribe --model moonshine/base --duration 300
    devar-engine-transcribe --source wav-file --input recordings/sermon.wav
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
import wave
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .audio import (
    DEFAULT_AUDIO_OUTPUT,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_CHANNELS,
    DEFAULT_TRANSCRIPT_OUTPUT,
    DEFAULT_MOONSHINE_MODEL,
    AudioSettings,
    MicrophoneSource,
    PcmSource,
    RecorderError,
    StdinRawPcmSource,
    ToneSource,
    WavFileSource,
    format_timestamp,
    list_microphone_devices,
    parse_device,
)


# ── Errors ────────────────────────────────────────────────────────────────────

class TranscriptionError(RuntimeError):
    """Raised when transcription cannot be completed."""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AudioSegment:
    index: int
    start_seconds: float
    end_seconds: float
    pcm: bytes
    recording_pcm: bytes | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class TranscribedSegment:
    index: int
    start_seconds: float
    end_seconds: float
    text: str


class SegmentTranscriber(Protocol):
    def transcribe(
        self,
        segment: AudioSegment,
        settings: AudioSettings,
        context: str | None = None,
    ) -> str:
        """Transcribe one audio segment and return plain text."""


# ── Moonshine transcriber ─────────────────────────────────────────────────────

MOONSHINE_MODELS = ("moonshine/tiny", "moonshine/base", "moonshine/medium", "tiny", "base", "medium")


class MoonshineBackend(Protocol):
    def transcribe(self, pcm: bytes, settings: AudioSettings) -> str:
        """Transcribe raw PCM bytes and return plain text."""


class _MoonshineOnnxBackend:
    """Lazy wrapper around the `useful-moonshine-onnx` package."""

    def __init__(self, model_name: str = DEFAULT_MOONSHINE_MODEL) -> None:
        self.model_name = self._normalize_model_name(model_name)
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._transcribe_lock = threading.Lock()

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if model_name in {"tiny", "base", "medium"}:
            return f"moonshine/{model_name}"
        if model_name in {"moonshine/tiny", "moonshine/base", "moonshine/medium"}:
            return model_name
        raise TranscriptionError(
            f"Unknown Moonshine model {model_name!r}. "
            "Use one of: moonshine/tiny, moonshine/base, moonshine/medium"
        )

    def _load_model(self) -> Any:
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from moonshine_onnx import MoonshineOnnxModel
            except ImportError as exc:
                raise TranscriptionError(
                    "The `useful-moonshine-onnx` package is not installed. "
                    'Install it with: pip install "devar-engine[transcription]"'
                ) from exc

            try:
                self._model = MoonshineOnnxModel(model_name=self.model_name)
            except Exception as exc:
                raise TranscriptionError(f"Could not load Moonshine model {self.model_name!r}: {exc}") from exc
            return self._model

    def transcribe(self, pcm: bytes, settings: AudioSettings) -> str:
        model = self._load_model()
        audio = _pcm_to_moonshine_audio(pcm, settings)
        try:
            from moonshine_onnx import transcribe as moonshine_transcribe

            with self._transcribe_lock:
                result = moonshine_transcribe(audio, model=model)
        except Exception as exc:
            raise TranscriptionError(f"Moonshine transcription failed: {exc}") from exc

        if isinstance(result, list):
            return " ".join(str(part) for part in result)
        return str(result)


class MoonshineTranscriber:
    """Transcribes PCM audio buffers with local Moonshine ONNX models.

    Supports `moonshine/tiny` and `moonshine/base`. Expects 16 kHz audio
    segments shorter than 64 seconds, which matches this pipeline's defaults.
    """

    def __init__(
        self,
        model: str = DEFAULT_MOONSHINE_MODEL,
        *,
        backend: MoonshineBackend | None = None,
    ) -> None:
        self.model = _MoonshineOnnxBackend._normalize_model_name(model)
        self.backend = backend if backend is not None else _MoonshineOnnxBackend(self.model)

    def transcribe(
        self,
        segment: AudioSegment,
        settings: AudioSettings,
        context: str | None = None,
    ) -> str:
        return self.backend.transcribe(segment.pcm, settings).strip()


def _pcm_to_moonshine_audio(pcm: bytes, settings: AudioSettings) -> Any:
    if settings.sample_rate != 16_000:
        raise TranscriptionError("Moonshine requires 16 kHz audio; use --sample-rate 16000")
    if settings.sample_width != 2:
        raise TranscriptionError("Moonshine currently requires 16-bit PCM; use --sample-width 2")
    if not pcm:
        raise TranscriptionError("Moonshine received an empty audio segment")

    try:
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError("Moonshine transcription requires numpy") from exc

    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if settings.channels > 1:
        audio = audio.reshape(-1, settings.channels).mean(axis=1)
    return audio


# ── Audio segmentation ────────────────────────────────────────────────────────

def iter_audio_segments(
    chunks: Iterable[bytes],
    settings: AudioSettings,
    buffer_seconds: float,
    overlap_seconds: float = 0.0,
) -> Iterable[AudioSegment]:
    if buffer_seconds <= 0:
        raise ValueError("buffer_seconds must be greater than zero")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be greater than or equal to zero")
    if overlap_seconds >= buffer_seconds:
        raise ValueError("overlap_seconds must be less than buffer_seconds")

    frame = settings.channels * settings.sample_width
    target = int(settings.bytes_per_second * buffer_seconds)
    target = max(frame, target - (target % frame))
    overlap = int(settings.bytes_per_second * overlap_seconds)
    overlap = overlap - (overlap % frame)
    step = target - overlap

    buf = bytearray()
    segment_start = 0
    recording_written_until = 0
    idx = 1

    for chunk in chunks:
        if not chunk:
            continue
        buf.extend(chunk)
        while len(buf) >= target:
            pcm = bytes(buf[:target])
            segment_end = segment_start + len(pcm)
            recording_start_offset = max(0, recording_written_until - segment_start)
            recording_pcm = pcm[recording_start_offset:]
            yield AudioSegment(
                index=idx,
                start_seconds=segment_start / settings.bytes_per_second,
                end_seconds=segment_end / settings.bytes_per_second,
                pcm=pcm,
                recording_pcm=recording_pcm,
            )
            recording_written_until = segment_end
            del buf[:step]
            segment_start += step
            idx += 1

    if buf and segment_start + len(buf) > recording_written_until:
        yield AudioSegment(
            index=idx,
            start_seconds=segment_start / settings.bytes_per_second,
            end_seconds=(segment_start + len(buf)) / settings.bytes_per_second,
            pcm=bytes(buf),
            recording_pcm=bytes(buf[max(0, recording_written_until - segment_start):]),
        )


def start_segment_capture(
    source: PcmSource,
    duration_seconds: float,
    buffer_seconds: float,
    overlap_seconds: float,
) -> tuple[queue.Queue[AudioSegment | Exception | None], threading.Thread]:
    seg_queue: queue.Queue[AudioSegment | Exception | None] = queue.Queue()

    def _run() -> None:
        try:
            for seg in iter_audio_segments(
                source.chunks(duration_seconds),
                source.settings,
                buffer_seconds,
                overlap_seconds=overlap_seconds,
            ):
                seg_queue.put(seg)
        except Exception as exc:
            seg_queue.put(exc)
        finally:
            seg_queue.put(None)

    t = threading.Thread(target=_run, name="moonshine-capture", daemon=True)
    t.start()
    return seg_queue, t


# ── Transcript writers ────────────────────────────────────────────────────────

def append_transcript(path: Path, segment: TranscribedSegment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".json", ".jsonl"}:
        _append_jsonl(path, segment)
    else:
        _append_text(path, segment)


def _append_text(path: Path, seg: TranscribedSegment) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"[{format_timestamp(seg.start_seconds)} - {format_timestamp(seg.end_seconds)}] "
            f"{seg.text}\n"
        )


def _append_jsonl(path: Path, seg: TranscribedSegment) -> None:
    record = {
        "timestamp": f"{format_timestamp(seg.start_seconds)} - {format_timestamp(seg.end_seconds)}",
        "text": seg.text,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dedupe_overlapping_text(previous: str, current: str, max_words: int = 18) -> str:
    """Remove repeated words caused by overlapping transcription windows."""
    previous_words = previous.split()
    current_words = current.split()
    if not previous_words or not current_words:
        return current

    def normalize(word: str) -> str:
        return re.sub(r"[^a-z0-9']", "", word.lower())

    previous_norm = [normalize(word) for word in previous_words]
    current_norm = [normalize(word) for word in current_words]
    max_overlap = min(max_words, len(previous_norm), len(current_norm))

    for size in range(max_overlap, 0, -1):
        if previous_norm[-size:] == current_norm[:size]:
            return " ".join(current_words[size:]).strip()
    return current


# ── Live transcription orchestrator ──────────────────────────────────────────

def live_transcribe(
    source: PcmSource,
    duration_seconds: float,
    buffer_seconds: float,
    transcript_output: Path,
    transcriber: SegmentTranscriber,
    audio_output: Path | None = None,
    max_parallel: int = 1,
    overlap_seconds: float = 1.0,
) -> list[TranscribedSegment]:
    """Capture audio and transcribe in near real-time using a parallel worker pool.

    While Moonshine processes buffer N, buffer N+1 is already recording.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if buffer_seconds <= 0:
        raise ValueError("buffer_seconds must be greater than zero")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be greater than or equal to zero")
    if overlap_seconds >= buffer_seconds:
        raise ValueError("overlap_seconds must be less than buffer_seconds")

    transcript_output = transcript_output.expanduser().resolve()
    transcript_output.parent.mkdir(parents=True, exist_ok=True)
    transcript_output.write_text("", encoding="utf-8")

    wav_writer: wave.Wave_write | None = None
    if audio_output is not None:
        audio_output = audio_output.expanduser().resolve()
        audio_output.parent.mkdir(parents=True, exist_ok=True)
        wav_writer = wave.open(str(audio_output), "wb")
        wav_writer.setnchannels(source.settings.channels)
        wav_writer.setsampwidth(source.settings.sample_width)
        wav_writer.setframerate(source.settings.sample_rate)

    seg_queue, capture_thread = start_segment_capture(source, duration_seconds, buffer_seconds, overlap_seconds)
    pending: deque[tuple[AudioSegment, Future[str]]] = deque()
    results: list[TranscribedSegment] = []
    last_prompt: str = ""

    def commit(seg: AudioSegment, text: str) -> None:
        nonlocal last_prompt
        text = _dedupe_overlapping_text(last_prompt, text)
        last_prompt = f"{last_prompt} {text}".strip()[-500:]
        ts = TranscribedSegment(
            index=seg.index,
            start_seconds=seg.start_seconds,
            end_seconds=seg.end_seconds,
            text=text,
        )
        results.append(ts)
        append_transcript(transcript_output, ts)
        print(f"[{format_timestamp(seg.start_seconds)}] {text}", flush=True)

    def flush_done() -> None:
        while pending and pending[0][1].done():
            seg, fut = pending.popleft()
            commit(seg, fut.result())

    try:
        with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="moonshine-transcribe") as pool:
            while True:
                item = seg_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                if wav_writer is not None:
                    wav_writer.writeframes(item.recording_pcm if item.recording_pcm is not None else item.pcm)
                future = pool.submit(transcriber.transcribe, item, source.settings, last_prompt or None)
                pending.append((item, future))
                flush_done()

            while pending:
                seg, fut = pending.popleft()
                commit(seg, fut.result())
    finally:
        capture_thread.join(timeout=2)
        if wav_writer is not None:
            wav_writer.close()

    return results


# ── Web/API entry points ─────────────────────────────────────────────────────

def transcribe_wav_bytes(
    wav_bytes: bytes,
    *,
    model: str = DEFAULT_MOONSHINE_MODEL,
    context: str | None = None,
    transcriber: MoonshineTranscriber | None = None,
) -> str:
    """Transcribe a WAV audio blob and return plain text.

    The WAV must be 16 kHz, 16-bit PCM. Mono is preferred; multi-channel input
    is mixed to mono during Moonshine conversion.
    """
    import io

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        settings = AudioSettings(
            sample_rate=wf.getframerate(),
            channels=wf.getnchannels(),
            sample_width=wf.getsampwidth(),
        )
        pcm = wf.readframes(wf.getnframes())

    return transcribe_pcm_bytes(
        pcm,
        settings,
        model=model,
        context=context,
        transcriber=transcriber,
    )


def transcribe_pcm_bytes(
    pcm: bytes,
    settings: AudioSettings | None = None,
    *,
    model: str = DEFAULT_MOONSHINE_MODEL,
    context: str | None = None,
    transcriber: MoonshineTranscriber | None = None,
) -> str:
    """Transcribe raw PCM bytes and return plain text."""
    if settings is None:
        settings = AudioSettings()

    if transcriber is None:
        transcriber = MoonshineTranscriber(model=model)

    duration = len(pcm) / settings.bytes_per_second
    segment = AudioSegment(
        index=1,
        start_seconds=0.0,
        end_seconds=duration,
        pcm=pcm,
    )
    return transcriber.transcribe(segment, settings, context)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transcribe audio locally with Moonshine.")
    p.add_argument("--source", choices=("microphone", "wav-file", "stdin", "tone"), default="microphone")
    p.add_argument("--duration", type=float, help="total capture duration in seconds")
    p.add_argument("--buffer-seconds", type=float, default=3.0,
                   help="seconds of audio per transcription request (default: 3 s)")
    p.add_argument("--overlap-seconds", type=float, default=1.0,
                   help="seconds of audio to overlap between requests (default: 1 s)")
    p.add_argument("--device", help="microphone device index or name")
    p.add_argument("--input", type=Path, help="WAV file path when --source wav-file")
    p.add_argument("--transcript-output", type=Path, default=DEFAULT_TRANSCRIPT_OUTPUT)
    p.add_argument("--audio-output", type=Path, default=DEFAULT_AUDIO_OUTPUT)
    p.add_argument("--no-audio-output", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    p.add_argument("--sample-width", type=int, default=DEFAULT_SAMPLE_WIDTH)
    p.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    p.add_argument("--max-parallel", type=int, default=1,
                   help="concurrent transcription jobs (default: 1)")
    p.add_argument("--model", default=DEFAULT_MOONSHINE_MODEL,
                   help=f"Moonshine model: moonshine/tiny, moonshine/base, or moonshine/medium (default: {DEFAULT_MOONSHINE_MODEL})")
    return p


def _source_from_args(args: argparse.Namespace) -> PcmSource:
    settings = AudioSettings(
        sample_rate=args.sample_rate,
        channels=args.channels,
        sample_width=args.sample_width,
        chunk_seconds=args.chunk_seconds,
    )
    if args.source == "microphone":
        return MicrophoneSource(settings=settings, device=parse_device(args.device))
    if args.source == "wav-file":
        if not args.input:
            raise RecorderError("--input is required with --source wav-file")
        return WavFileSource(path=args.input, settings=settings)
    if args.source == "stdin":
        return StdinRawPcmSource(settings=settings)
    if args.source == "tone":
        return ToneSource(settings=settings)
    raise RecorderError(f"unknown source: {args.source}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_devices:
        print(list_microphone_devices())
        return 0

    if args.duration is None:
        print("error: --duration is required", file=sys.stderr)
        return 2

    try:
        source = _source_from_args(args)
        transcriber = MoonshineTranscriber(model=args.model)
        started = time.monotonic()
        segments = live_transcribe(
            source=source,
            duration_seconds=args.duration,
            buffer_seconds=args.buffer_seconds,
            overlap_seconds=args.overlap_seconds,
            transcript_output=args.transcript_output,
            transcriber=transcriber,
            audio_output=None if args.no_audio_output else args.audio_output,
            max_parallel=args.max_parallel,
        )
    except (RecorderError, TranscriptionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    elapsed = time.monotonic() - started
    print(
        f"\nDone — {len(segments)} segments in {elapsed:.1f}s. "
        f"Transcript: {args.transcript_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
