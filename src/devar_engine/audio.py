"""Shared audio capture utilities and configuration for Devar's local engine."""

from __future__ import annotations

import math
import queue
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2          # 16-bit PCM
DEFAULT_CHUNK_SECONDS = 0.25

DEFAULT_DATA_DIR = Path.cwd() / "data"
DEFAULT_TRANSCRIPTS_DIR = Path.cwd() / "transcripts"
DEFAULT_RECORDINGS_DIR = Path.cwd() / "recordings"
DEFAULT_TRANSCRIPT_OUTPUT = DEFAULT_TRANSCRIPTS_DIR / "live.jsonl"
DEFAULT_AUDIO_OUTPUT = DEFAULT_RECORDINGS_DIR / "live.wav"
DEFAULT_BIBLE_PATH = DEFAULT_DATA_DIR / "nkjv.json"

DEFAULT_MOONSHINE_MODEL = "moonshine/base"


# ── Errors ────────────────────────────────────────────────────────────────────

class RecorderError(RuntimeError):
    """Raised when audio capture cannot be configured or read."""


# ── AudioSettings ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    sample_width: int = DEFAULT_SAMPLE_WIDTH
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.channels <= 0:
            raise ValueError("channels must be greater than zero")
        if self.sample_width not in (1, 2, 3, 4):
            raise ValueError("sample_width must be 1, 2, 3, or 4 bytes")
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than zero")

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width

    @property
    def chunk_size_bytes(self) -> int:
        size = int(self.bytes_per_second * self.chunk_seconds)
        frame = self.channels * self.sample_width
        return max(frame, size - (size % frame))


# ── PCM source protocol ───────────────────────────────────────────────────────

class PcmSource(Protocol):
    name: str
    settings: AudioSettings

    def chunks(self, duration_seconds: float | None) -> Iterable[bytes]:
        """Yield raw PCM chunks until the source or duration is exhausted."""


# ── ToneSource (deterministic test source) ────────────────────────────────────

@dataclass
class ToneSource:
    """Generates a sine-wave tone — no microphone hardware required."""

    settings: AudioSettings
    frequency_hz: float = 440.0
    amplitude: float = 0.25
    name: str = "tone"

    def chunks(self, duration_seconds: float | None) -> Iterable[bytes]:
        if duration_seconds is None:
            raise RecorderError("ToneSource requires an explicit duration")
        if self.settings.sample_width != 2:
            raise RecorderError("ToneSource only supports 16-bit PCM")

        total_frames = int(self.settings.sample_rate * duration_seconds)
        frames_per_chunk = max(
            1, self.settings.chunk_size_bytes // (self.settings.channels * self.settings.sample_width)
        )
        max_int16 = 32_767
        frame_index = 0

        while frame_index < total_frames:
            n = min(frames_per_chunk, total_frames - frame_index)
            chunk = bytearray()
            for i in range(n):
                t = (frame_index + i) / self.settings.sample_rate
                sample = int(max_int16 * self.amplitude * math.sin(2 * math.pi * self.frequency_hz * t))
                sample_bytes = sample.to_bytes(2, byteorder="little", signed=True)
                for _ in range(self.settings.channels):
                    chunk.extend(sample_bytes)
            frame_index += n
            yield bytes(chunk)


# ── WavFileSource ─────────────────────────────────────────────────────────────

@dataclass
class WavFileSource:
    path: Path
    settings: AudioSettings | None = None
    name: str = "wav-file"

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise RecorderError(f"WAV file not found: {self.path}")
        with wave.open(str(self.path), "rb") as wf:
            detected = AudioSettings(
                sample_rate=wf.getframerate(),
                channels=wf.getnchannels(),
                sample_width=wf.getsampwidth(),
            )
        if self.settings is None:
            self.settings = detected
        elif self.settings != detected:
            raise RecorderError(
                "WAV file settings do not match the requested settings; "
                "resampling is out of scope here"
            )

    def chunks(self, duration_seconds: float | None) -> Iterable[bytes]:
        assert self.settings is not None
        max_frames = None if duration_seconds is None else int(self.settings.sample_rate * duration_seconds)
        frames_per_chunk = max(
            1, self.settings.chunk_size_bytes // (self.settings.channels * self.settings.sample_width)
        )
        frames_read = 0
        with wave.open(str(self.path), "rb") as wf:
            while True:
                to_read = frames_per_chunk
                if max_frames is not None:
                    remaining = max_frames - frames_read
                    if remaining <= 0:
                        break
                    to_read = min(to_read, remaining)
                chunk = wf.readframes(to_read)
                if not chunk:
                    break
                frames_read += len(chunk) // (self.settings.channels * self.settings.sample_width)
                yield chunk


# ── MicrophoneSource ──────────────────────────────────────────────────────────

@dataclass
class MicrophoneSource:
    settings: AudioSettings
    device: int | str | None = None
    name: str = "microphone"

    def chunks(self, duration_seconds: float | None) -> Iterable[bytes]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RecorderError("MicrophoneSource requires the `sounddevice` package") from exc

        dtype = _sd_dtype(self.settings.sample_width)
        frames_per_chunk = max(
            1, self.settings.chunk_size_bytes // (self.settings.channels * self.settings.sample_width)
        )
        deadline = None if duration_seconds is None else time.monotonic() + duration_seconds
        audio_queue: queue.Queue[bytes | Exception] = queue.Queue()

        def _cb(indata: bytes, frames: int, t: object, status: object) -> None:
            if status:
                audio_queue.put(RecorderError(f"sounddevice status: {status}"))
            audio_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=self.settings.sample_rate,
            channels=self.settings.channels,
            dtype=dtype,
            blocksize=frames_per_chunk,
            device=self.device,
            callback=_cb,
        ):
            while deadline is None or time.monotonic() < deadline:
                timeout = None if deadline is None else max(0.0, min(0.5, deadline - time.monotonic()))
                try:
                    item = audio_queue.get(timeout=timeout)
                except queue.Empty:
                    continue
                if isinstance(item, Exception):
                    raise item
                yield item


# ── StdinRawPcmSource ─────────────────────────────────────────────────────────

@dataclass
class StdinRawPcmSource:
    settings: AudioSettings
    name: str = "stdin"

    def chunks(self, duration_seconds: float | None) -> Iterable[bytes]:
        remaining = (
            None
            if duration_seconds is None
            else int(self.settings.bytes_per_second * duration_seconds)
        )
        while True:
            to_read = self.settings.chunk_size_bytes
            if remaining is not None:
                if remaining <= 0:
                    break
                to_read = min(to_read, remaining)
            chunk = sys.stdin.buffer.read(to_read)
            if not chunk:
                break
            if remaining is not None:
                remaining -= len(chunk)
            yield chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sd_dtype(sample_width: int) -> str:
    mapping = {1: "int8", 2: "int16", 4: "int32"}
    if sample_width not in mapping:
        raise RecorderError("MicrophoneSource supports 8-, 16-, or 32-bit PCM only")
    return mapping[sample_width]


def pcm_to_wav_bytes(pcm: bytes, settings: AudioSettings) -> bytes:
    """Wrap raw PCM bytes in a WAV container and return the result as bytes."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(settings.channels)
        wf.setsampwidth(settings.sample_width)
        wf.setframerate(settings.sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, AudioSettings]:
    """Extract raw PCM and AudioSettings from WAV bytes."""
    import io
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        settings = AudioSettings(
            sample_rate=wf.getframerate(),
            channels=wf.getnchannels(),
            sample_width=wf.getsampwidth(),
        )
        pcm = wf.readframes(wf.getnframes())
    return pcm, settings


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def list_microphone_devices() -> str:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RecorderError("listing devices requires the `sounddevice` package") from exc
    return str(sd.query_devices())


def parse_device(raw: str | None) -> int | str | None:
    if raw is None or raw.strip() == "":
        return None
    return int(raw.strip()) if raw.strip().isdecimal() else raw.strip()
