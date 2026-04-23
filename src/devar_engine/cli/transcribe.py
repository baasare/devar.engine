"""Console entry point for sermon transcription."""

from devar_engine.transcription import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
