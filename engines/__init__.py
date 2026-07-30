"""Pluggable ASR backends.

Selection is by the ``TRANSCRIPTION_ENGINE`` env var (``faster_whisper`` |
``whisper_cpp``), defaulting to ``faster_whisper`` so an unconfigured install
behaves exactly as it did before engines existed.

TODO(phase 2): read the selection from the settings file the Settings panel will
write, falling back to this env var. `get_engine()` is the single seam — nothing
else needs to change.
"""

from __future__ import annotations

import os
from typing import Callable

from .base import TranscriptionEngine
from .errors import (
    EngineUnavailableError,
    InferenceHTTPError,
    MalformedResponseError,
    ModelNotLoadedError,
    ServerUnreachableError,
    TranscriptionError,
    TranscriptionTimeoutError,
)
from .faster_whisper_engine import FasterWhisperEngine
from .schema import EngineHooks, Segment, TranscribeOptions, TranscriptResult, Word
from .whisper_cpp_engine import WhisperCppHttpEngine

DEFAULT_ENGINE = "faster_whisper"

ENGINE_REGISTRY: dict[str, Callable[[], TranscriptionEngine]] = {
    "faster_whisper": FasterWhisperEngine,
    "whisper_cpp": WhisperCppHttpEngine,
}


def resolve_engine_name(name: str | None = None) -> str:
    selected = (name or os.environ.get("TRANSCRIPTION_ENGINE") or DEFAULT_ENGINE).strip().lower()
    if selected not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown transcription engine {selected!r}. Available: {', '.join(sorted(ENGINE_REGISTRY))}"
        )
    return selected


def get_engine(name: str | None = None) -> TranscriptionEngine:
    return ENGINE_REGISTRY[resolve_engine_name(name)]()


__all__ = [
    "DEFAULT_ENGINE",
    "ENGINE_REGISTRY",
    "EngineHooks",
    "EngineUnavailableError",
    "FasterWhisperEngine",
    "InferenceHTTPError",
    "MalformedResponseError",
    "ModelNotLoadedError",
    "Segment",
    "ServerUnreachableError",
    "TranscribeOptions",
    "TranscriptResult",
    "TranscriptionEngine",
    "TranscriptionError",
    "TranscriptionTimeoutError",
    "WhisperCppHttpEngine",
    "Word",
    "get_engine",
    "resolve_engine_name",
]
