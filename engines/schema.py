"""Canonical transcript schema shared by every transcription engine.

`to_payload()` is the contract with the rest of the app: it reproduces byte-for-byte
the `{"segments": [...], "info": {...}}` structure that `whisper_worker.py` has always
produced, so the transcript editor, the .srt/.txt writers and `*_words.json` keep
seeing the exact same shape. `to_dict()` is the richer, engine-aware view used by
diagnostics and the validation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_TIME_NDIGITS = 3
_PROB_NDIGITS = 4


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def word(self) -> str:
        """Alias — whisper.cpp calls this field `word`, faster-whisper calls it `word` too."""
        return self.text

    @classmethod
    def from_payload(cls, data: dict) -> "Word":
        text = data.get("text", data.get("word", ""))
        return cls(
            text=str(text).strip(),
            start=float(data["start"]),
            end=float(data["end"]),
            probability=float(data.get("probability", 1.0)),
        )

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, _TIME_NDIGITS),
            "end": round(self.end, _TIME_NDIGITS),
            "probability": round(self.probability, _PROB_NDIGITS),
        }

    def to_dict(self) -> dict:
        return self.to_payload()


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    id: Optional[int] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    temperature: Optional[float] = None

    @classmethod
    def from_payload(cls, data: dict) -> "Segment":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text", "")).strip(),
            words=[Word.from_payload(w) for w in data.get("words") or []],
            id=data.get("id"),
            avg_logprob=data.get("avg_logprob"),
            no_speech_prob=data.get("no_speech_prob"),
            temperature=data.get("temperature"),
        )

    def to_payload(self) -> dict:
        """Legacy shape consumed by app.py — exactly four keys, nothing more."""
        return {
            "start": round(self.start, _TIME_NDIGITS),
            "end": round(self.end, _TIME_NDIGITS),
            "text": self.text,
            "words": [w.to_payload() for w in self.words],
        }

    def to_dict(self) -> dict:
        out = self.to_payload()
        for key in ("id", "avg_logprob", "no_speech_prob", "temperature"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class TranscriptResult:
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    language_probability: float = 1.0
    duration: float = 0.0
    engine: str = ""
    has_word_timestamps: bool = True
    warnings: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict, engine: str = "") -> "TranscriptResult":
        info = payload.get("info") or {}
        segments = [Segment.from_payload(s) for s in payload.get("segments") or []]
        return cls(
            segments=segments,
            language=info.get("language"),
            language_probability=float(info.get("language_probability", 1.0)),
            duration=float(info.get("duration", 0.0)),
            engine=engine,
            has_word_timestamps=any(s.words for s in segments),
        )

    def info_payload(self) -> dict:
        return {
            "language": self.language or "en",
            "language_probability": round(self.language_probability, 3),
            "duration": round(self.duration, 2),
        }

    def to_payload(self) -> dict:
        return {"segments": [s.to_payload() for s in self.segments], "info": self.info_payload()}

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "info": self.info_payload(),
            "engine": self.engine,
            "has_word_timestamps": self.has_word_timestamps,
            "warnings": list(self.warnings),
            "wall_time_s": round(self.wall_time_s, 3),
        }

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)


@dataclass
class EngineHooks:
    """Optional side-channels an engine may call while running.

    `on_segment` receives legacy-shaped segment dicts so the live SSE preview in
    app.py keeps working without translation.
    """

    on_segment: Optional[Callable[[dict], None]] = None
    on_log: Optional[Callable[[str], None]] = None
    on_process_start: Optional[Callable[[int], None]] = None
    on_process_end: Optional[Callable[[int], None]] = None

    def log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)

    def segment(self, seg: dict) -> None:
        if self.on_segment:
            self.on_segment(seg)


@dataclass
class TranscribeOptions:
    language: Optional[str] = None
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    temperature: float = 0.0
    chunk_length: int = 30
    no_speech_threshold: float = 0.6
    hotwords: str = ""
    condition_on_prev: bool = False
    vad_preseg: bool = True
    hooks: EngineHooks = field(default_factory=EngineHooks)

    def worker_args(self, audio_path: str) -> dict[str, Any]:
        """The exact args-file payload `whisper_worker.py` has always received."""
        return {
            "audio_path": audio_path,
            "language": self.language,
            "model_size": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "chunk_length": self.chunk_length,
            "no_speech_threshold": self.no_speech_threshold,
            "hotwords": self.hotwords,
            "condition_on_prev": self.condition_on_prev,
        }
