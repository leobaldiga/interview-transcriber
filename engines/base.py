"""The engine interface every backend implements."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .schema import TranscribeOptions, TranscriptResult


@runtime_checkable
class TranscriptionEngine(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier used in logs and in the TRANSCRIPTION_ENGINE env var."""
        ...

    def health(self) -> None:
        """Raise an `EngineUnavailableError` subclass if the engine cannot run right now."""
        ...

    def is_available(self) -> bool:
        """Non-raising form of `health()`."""
        ...

    def transcribe(self, audio_path: str | Path, options: TranscribeOptions) -> TranscriptResult:
        ...
