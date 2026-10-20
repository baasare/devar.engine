# Devar Engine

Reusable sermon transcription and Bible verse suggestion engine.

`devar-engine` is the Python engine package for building sermon-assistant
workflows. It can transcribe sermon audio, maintain rolling transcript files,
and suggest Bible verses from a local JSON Bible file. It works as a
command-line tool for local experiments and as a library that can be imported by
a backend service.

## Installation

Install from PyPI:

```bash
python -m pip install devar-engine
```

Install with local microphone support:

```bash
python -m pip install "devar-engine[audio]"
```

Install with local Moonshine transcription support:

```bash
python -m pip install "devar-engine[transcription]"
```

Install both microphone capture and transcription support:

```bash
python -m pip install "devar-engine[all]"
```

Install the optional on-device EmbeddingGemma 2 runtime for Apple silicon:

```bash
python -m pip install "devar-engine[embedding]"
```

The first model preparation downloads about 1.27 GB. Verse vectors are cached
locally per Bible content and model version; semantic queries then embed only
the rolling transcript text.

Install from a checked-out repository while developing:

```bash
cd /path/to/devar.engine
python -m pip install -e ".[all]"
```

## Bible Data

Verse suggestions require a local Bible JSON file. The expected shape is a flat
array of verse objects:

```json
[
  {
    "book": "John",
    "chapter": 3,
    "verse": 16,
    "text": "For God so loved the world..."
  }
]
```

Required fields:

- `book`: Bible book name.
- `chapter`: Chapter number.
- `verse`: Verse number.
- `text`: Verse text.

An optional `translation` field is accepted, but it is not required. Bible files
are runtime data and should not be committed to source control unless you have
the right to distribute them.

## Command-Line Usage

`devar-engine` installs two console commands:

- `devar-engine-transcribe`: transcribes audio into a transcript file.
- `devar-engine-suggest`: suggests Bible verses from transcript text or an ad-hoc
  query.

### Suggest Verses From Text

```bash
devar-engine-suggest query \
  --bible data/nkjv.json \
  --text "Do not be anxious about anything" \
  --top-k 5
```

### Suggest Verses From A Transcript

Run one suggestion cycle:

```bash
devar-engine-suggest once \
  --bible data/nkjv.json \
  --transcript-file transcripts/live.jsonl
```

Watch a transcript and print suggestions repeatedly:

```bash
devar-engine-suggest loop \
  --bible data/nkjv.json \
  --transcript-file transcripts/live.jsonl \
  --interval-seconds 10 \
  --last-seconds 15
```

### Transcribe A WAV File

```bash
devar-engine-transcribe \
  --source wav-file \
  --input recordings/sermon.wav \
  --duration 3600 \
  --transcript-output transcripts/sermon.jsonl \
  --no-audio-output
```

For best results, use uncompressed WAV audio with:

- 16 kHz sample rate.
- 16-bit signed PCM samples.
- Mono audio.

### Transcribe From A Microphone

Install the audio and transcription extras first:

```bash
python -m pip install "devar-engine[all]"
```

List available input devices:

```bash
devar-engine-transcribe --list-devices
```

Record and transcribe:

```bash
devar-engine-transcribe \
  --source microphone \
  --duration 3600 \
  --transcript-output transcripts/live.jsonl \
  --audio-output recordings/live.wav
```

### Test Without Audio Hardware

```bash
devar-engine-transcribe \
  --source tone \
  --duration 5 \
  --transcript-output transcripts/tone.jsonl \
  --no-audio-output
```

## Python API

### Suggest Verses

```python
from pathlib import Path

from devar_engine.suggestions import BibleIndex, keyword_suggest

bible = BibleIndex(Path("data/nkjv.json"))
results = keyword_suggest(
    bible,
    "God loved the world and gave His Son",
    top_k=3,
)

for result in results:
    print(result.reference)
    print(result.display_text)
```

### Hybrid Bible Search

The hybrid ranker gives exact spoken references priority, then combines BM25
word matching with cosine ranks from an optional local embedding index. The
reference tracker can carry a book and chapter across transcript updates:

```python
from pathlib import Path

from devar_engine.embeddinggemma import EmbeddingGemma2, ensure_model
from devar_engine.references import SpokenReferenceTracker
from devar_engine.semantic import SemanticBibleIndex, hybrid_suggest
from devar_engine.suggestions import BibleIndex

bible = BibleIndex(Path("data/nkjv.json"))
embedder = EmbeddingGemma2(ensure_model())
semantic_index = SemanticBibleIndex(
    bible,
    embedder,
    cache_dir=Path.home() / ".cache/devar/verse-index",
)
references = SpokenReferenceTracker(context_ttl=60)

for text in ("the book of Job", "chapter five", "verse seven"):
    results = hybrid_suggest(
        bible,
        text,
        semantic_index=semantic_index,
        reference_tracker=references,
        top_k=3,
    )
    print([result.reference for result in results])
```

If the embedding runtime is unavailable, `hybrid_suggest` still returns exact
reference and BM25 results. Call `embedder.close()` when the local model is no
longer needed.

### Read Recent Transcript Context

```python
from pathlib import Path

from devar_engine.suggestions import read_recent_context

text, timestamp_range = read_recent_context(
    Path("transcripts/live.jsonl"),
    last_seconds=15,
)
```

### Transcribe WAV Bytes In A Backend

```python
from devar_engine.transcription import transcribe_wav_bytes

with open("recordings/sermon.wav", "rb") as f:
    wav_bytes = f.read()

text = transcribe_wav_bytes(wav_bytes)
```

### Convert PCM To WAV Bytes

```python
from devar_engine.audio import AudioSettings, pcm_to_wav_bytes

settings = AudioSettings(sample_rate=16000, channels=1, sample_width=2)
wav_bytes = pcm_to_wav_bytes(raw_pcm_bytes, settings)
```

## Transcript Format

JSONL transcript entries are written one object per line:

```json
{"timestamp": "00:00 - 00:05", "text": "Grace and peace to you"}
```

The suggestion command reads the most recent timestamped entries when possible.
For plain-text transcripts, it falls back to the last characters in the file.

## Runtime Files

Typical local directories:

```text
data/          Bible JSON files
recordings/    optional WAV recordings
transcripts/   generated transcript files
```

Treat these as local runtime directories. Generated transcripts, recordings, and
licensed Bible files should normally stay out of git.

## License

Devar Engine is licensed under the Apache License 2.0.

This package does not include Bible translations, recordings, transcripts,
speech model weights, or other runtime data. Those assets may have their own
licenses and usage restrictions.

## Development

Run the package tests from this directory:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Run a CLI smoke test with a local Bible file:

```bash
PYTHONPATH=src python -m devar_engine.cli.suggest query \
  --bible data/nkjv.json \
  --text "The Lord is my shepherd"
```

## Releasing

Build and verify distributions from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

Test a release on TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  devar-engine
```

Publish the real release to PyPI:

```bash
python -m twine upload dist/*
```

GitHub Actions workflows are also included for TestPyPI and PyPI publishing.
They are triggered manually with `workflow_dispatch` or automatically when you
push a tag like `devar-engine-v0.1.2`.

## Troubleshooting

`Bible file not found`

Pass `--bible /path/to/bible.json`, or place the file where your command expects
it.

`No suggestions`

Make sure the transcript has meaningful sermon text and the Bible file uses the
required flat verse-object shape.

`WAV file settings do not match`

Use 16 kHz, 16-bit, mono WAV input, or convert the file before passing it to
`devar-engine-transcribe`.
