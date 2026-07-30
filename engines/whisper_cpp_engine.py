"""whisper.cpp backend — talks HTTP to a local `whisper-server`.

Verified against ggml-org/whisper.cpp `examples/server/server.cpp` (master, 2026-07):

* the only inference endpoint is ``POST /inference``; there is **no**
  ``/v1/audio/transcriptions`` route despite the OpenAI-shaped payload;
* the audio field is ``file``; a missing one returns HTTP 400;
* ``response_format`` accepts ``json|text|srt|vtt|verbose_json``, and an
  *unknown* value silently degrades to ``{"text": ...}`` instead of erroring —
  which is why `_normalize` treats a segment-less body as a hard warning;
* ``verbose_json`` segment/word timestamps are floats in seconds (``t0 * 0.01``);
* form booleans are parsed by ``parse_str_to_bool``, which only accepts
  ``true|1|yes|y`` — capitalised ``True`` reads as false, so everything is
  lower-cased on the way out;
* ``token_timestamps`` defaults to ``!no_timestamps``, so the ``words`` array is
  populated by default. Entries are token-level, so a "word" may be a sub-word
  piece — the same granularity assumption downstream already makes for CJK.

Non-WAV input requires the server to have been started with ``--convert``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from .errors import (
    InferenceHTTPError,
    MalformedResponseError,
    ModelNotLoadedError,
    ServerUnreachableError,
    TranscriptionTimeoutError,
)
from .schema import Segment, TranscribeOptions, TranscriptResult, Word
from .timestamps import extract_span, parse_timestamp

NAME = "whisper_cpp"

DEFAULT_BASE_URL = "http://127.0.0.1:8089"
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 3600.0
DEFAULT_RETRIES = 3

# whisper.cpp reports `language` via whisper_lang_str_full, i.e. "english" not "en".
# app.py stores that value on the job and shows it in the UI, so map it back.
_LANGUAGE_NAME_TO_CODE = {
    "english": "en", "chinese": "zh", "german": "de", "spanish": "es", "russian": "ru",
    "korean": "ko", "french": "fr", "japanese": "ja", "portuguese": "pt", "turkish": "tr",
    "polish": "pl", "catalan": "ca", "dutch": "nl", "arabic": "ar", "swedish": "sv",
    "italian": "it", "indonesian": "id", "hindi": "hi", "finnish": "fi", "vietnamese": "vi",
    "hebrew": "he", "ukrainian": "uk", "greek": "el", "malay": "ms", "czech": "cs",
    "romanian": "ro", "danish": "da", "hungarian": "hu", "tamil": "ta", "norwegian": "no",
    "thai": "th", "urdu": "ur", "croatian": "hr", "bulgarian": "bg", "lithuanian": "lt",
    "latin": "la", "maori": "mi", "malayalam": "ml", "welsh": "cy", "slovak": "sk",
    "telugu": "te", "persian": "fa", "latvian": "lv", "bengali": "bn", "serbian": "sr",
    "azerbaijani": "az", "slovenian": "sl", "kannada": "kn", "estonian": "et",
    "macedonian": "mk", "breton": "br", "basque": "eu", "icelandic": "is", "armenian": "hy",
    "nepali": "ne", "mongolian": "mn", "bosnian": "bs", "kazakh": "kk", "albanian": "sq",
    "swahili": "sw", "galician": "gl", "marathi": "mr", "punjabi": "pa", "sinhala": "si",
    "khmer": "km", "shona": "sn", "yoruba": "yo", "somali": "so", "afrikaans": "af",
    "occitan": "oc", "georgian": "ka", "belarusian": "be", "tajik": "tg", "sindhi": "sd",
    "gujarati": "gu", "amharic": "am", "yiddish": "yi", "lao": "lo", "uzbek": "uz",
    "faroese": "fo", "haitian creole": "ht", "pashto": "ps", "turkmen": "tk", "nynorsk": "nn",
    "maltese": "mt", "sanskrit": "sa", "luxembourgish": "lb", "myanmar": "my",
    "tibetan": "bo", "tagalog": "tl", "malagasy": "mg", "assamese": "as", "tatar": "tt",
    "hawaiian": "haw", "lingala": "ln", "hausa": "ha", "bashkir": "ba", "javanese": "jw",
    "sundanese": "su", "cantonese": "yue",
}


def normalize_language(value) -> str | None:
    """Map whisper.cpp's full language name back to an ISO code, pass codes through."""
    if not value:
        return None
    text = str(value).strip().lower()
    if not text or text == "auto":
        return None
    if len(text) <= 3 and text not in _LANGUAGE_NAME_TO_CODE:
        return text
    return _LANGUAGE_NAME_TO_CODE.get(text, text)


def _is_special_token(text: str) -> bool:
    return text.startswith("[_") and text.endswith("]")


def _bool(value: bool) -> str:
    return "true" if value else "false"


class WhisperCppHttpEngine:
    def __init__(
        self,
        base_url: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        retries: int | None = None,
        session: requests.Session | None = None,
    ):
        self.base_url = (base_url or os.environ.get("WHISPER_CPP_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.connect_timeout = connect_timeout if connect_timeout is not None else _env_float(
            "WHISPER_CPP_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT
        )
        self.read_timeout = read_timeout if read_timeout is not None else _env_float(
            "WHISPER_CPP_TIMEOUT", DEFAULT_READ_TIMEOUT
        )
        self.retries = retries if retries is not None else int(_env_float("WHISPER_CPP_RETRIES", DEFAULT_RETRIES))
        self.session = session or requests.Session()

    @property
    def name(self) -> str:
        return NAME

    # ── availability ────────────────────────────────────────────────

    def health(self) -> None:
        url = f"{self.base_url}/health"
        try:
            response = self.session.get(url, timeout=(self.connect_timeout, self.connect_timeout))
        except requests.Timeout as exc:
            raise TranscriptionTimeoutError(
                f"whisper.cpp health check timed out after {self.connect_timeout}s at {url}"
            ) from exc
        except requests.ConnectionError as exc:
            raise ServerUnreachableError(
                f"Cannot reach whisper.cpp server at {self.base_url} — is the container running? ({exc})"
            ) from exc

        if response.status_code == 503:
            raise ModelNotLoadedError(
                f"whisper.cpp server at {self.base_url} is up but has no model loaded "
                f"(HTTP 503: {response.text.strip()[:200]})"
            )
        if response.status_code == 404:
            # Older builds have no /health route; a 404 still proves the server is listening.
            return
        if response.status_code >= 400:
            raise ModelNotLoadedError(
                f"whisper.cpp health check failed with HTTP {response.status_code}: {response.text.strip()[:200]}"
            )

        status = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                status = str(body.get("status", "")).lower()
        except ValueError:
            status = response.text.strip().lower()

        if "loading" in status:
            raise ModelNotLoadedError(
                f"whisper.cpp server at {self.base_url} is still loading its model (status={status!r})"
            )

    def is_available(self) -> bool:
        try:
            self.health()
        except (ServerUnreachableError, ModelNotLoadedError, TranscriptionTimeoutError):
            return False
        return True

    # ── inference ───────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path, options: TranscribeOptions) -> TranscriptResult:
        audio_path = Path(audio_path)
        hooks = options.hooks
        self.health()

        started = time.monotonic()
        payload = self._post_inference(audio_path, options)
        result = self._normalize(payload)
        result.wall_time_s = time.monotonic() - started

        if audio_path.suffix.lower() != ".wav":
            result.warnings.append(
                f"{audio_path.suffix} sent to whisper-server; the server must be started with --convert"
            )
        for warning in result.warnings:
            hooks.log(f"[whisper_cpp] WARNING: {warning}")
        for segment in result.segments:
            hooks.segment(segment.to_payload())
        return result

    def _form_fields(self, options: TranscribeOptions) -> dict[str, str]:
        fields = {
            "response_format": "verbose_json",
            "language": options.language or "auto",
            "temperature": str(options.temperature),
            "beam_size": str(options.beam_size),
            "no_speech_thold": str(options.no_speech_threshold),
            "no_timestamps": _bool(False),
            "token_timestamps": _bool(True),
            "translate": _bool(False),
        }
        if options.hotwords.strip():
            # whisper.cpp has no hotwords equivalent; the initial prompt is the
            # closest decoding bias it exposes.
            fields["prompt"] = options.hotwords.strip()
        return fields

    def _post_inference(self, audio_path: Path, options: TranscribeOptions) -> dict:
        url = f"{self.base_url}/inference"
        fields = self._form_fields(options)
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                with open(audio_path, "rb") as handle:
                    response = self.session.post(
                        url,
                        files={"file": (audio_path.name, handle, "application/octet-stream")},
                        data=fields,
                        timeout=(self.connect_timeout, self.read_timeout),
                    )
            except requests.Timeout as exc:
                raise TranscriptionTimeoutError(
                    f"whisper.cpp inference timed out after {self.read_timeout}s — "
                    f"raise WHISPER_CPP_TIMEOUT for long recordings"
                ) from exc
            except requests.ConnectionError as exc:
                # Transient only: retry the connection, never a rejected request.
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(delay)
                delay *= 2
                continue

            return self._decode(response)

        raise ServerUnreachableError(
            f"whisper.cpp server at {self.base_url} unreachable after {self.retries} attempts ({last_error})"
        )

    def _decode(self, response: requests.Response) -> dict:
        if response.status_code == 503:
            raise ModelNotLoadedError(
                f"whisper.cpp server has no model loaded (HTTP 503: {response.text.strip()[:200]})"
            )
        if response.status_code >= 400:
            raise InferenceHTTPError(
                f"whisper.cpp inference failed with HTTP {response.status_code}: {response.text.strip()[:300]}",
                status_code=response.status_code,
                body=response.text,
            )

        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise MalformedResponseError(
                f"whisper.cpp returned non-JSON body: {response.text.strip()[:200]!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise MalformedResponseError(f"whisper.cpp returned {type(payload).__name__}, expected a JSON object")
        if "error" in payload and "segments" not in payload:
            raise InferenceHTTPError(
                f"whisper.cpp reported an error: {payload['error']}", status_code=response.status_code
            )
        return payload

    # ── normalisation ───────────────────────────────────────────────

    def _normalize(self, payload: dict) -> TranscriptResult:
        warnings: list[str] = []
        raw_segments = payload.get("segments")

        if raw_segments is None:
            text = payload.get("text")
            if text is None:
                raise MalformedResponseError(
                    f"whisper.cpp response has neither 'segments' nor 'text' (keys: {sorted(payload)})"
                )
            warnings.append(
                "server returned plain text with no segments — this build ignored "
                "response_format=verbose_json; word and segment timestamps are unavailable"
            )
            raw_segments = []
            flat_text = str(text).strip()
        else:
            flat_text = ""

        if not isinstance(raw_segments, list):
            raise MalformedResponseError(f"'segments' is {type(raw_segments).__name__}, expected a list")

        segments: list[Segment] = []
        segments_without_words = 0
        for index, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                raise MalformedResponseError(f"segment {index} is {type(raw).__name__}, expected an object")
            try:
                start, end = extract_span(raw)
            except (KeyError, ValueError) as exc:
                raise MalformedResponseError(f"segment {index} has unusable timestamps: {exc}") from exc

            words = self._normalize_words(raw.get("words"), index)
            if not words:
                segments_without_words += 1

            segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=str(raw.get("text", "")).strip(),
                    words=words,
                    id=raw.get("id", index),
                    avg_logprob=raw.get("avg_logprob"),
                    no_speech_prob=raw.get("no_speech_prob"),
                    temperature=raw.get("temperature"),
                )
            )

        if flat_text and not segments:
            duration = _as_float(payload.get("duration"), 0.0)
            segments.append(Segment(start=0.0, end=duration, text=flat_text, words=[]))
            segments_without_words = 1

        if segments and segments_without_words:
            warnings.append(
                f"{segments_without_words}/{len(segments)} segments came back without word-level "
                "timestamps — speaker assignment will fall back to segment granularity"
            )

        duration = _as_float(payload.get("duration"), 0.0)
        if not duration and segments:
            duration = max(s.end for s in segments)

        language = normalize_language(payload.get("detected_language") or payload.get("language"))
        probability = _as_float(payload.get("detected_language_probability"), 1.0)

        return TranscriptResult(
            segments=segments,
            language=language,
            language_probability=probability,
            duration=duration,
            engine=NAME,
            has_word_timestamps=bool(segments) and segments_without_words == 0,
            warnings=warnings,
        )

    def _normalize_words(self, raw_words, segment_index: int) -> list[Word]:
        if not isinstance(raw_words, list):
            return []
        words: list[Word] = []
        for raw in raw_words:
            if not isinstance(raw, dict):
                raise MalformedResponseError(
                    f"segment {segment_index} word entry is {type(raw).__name__}, expected an object"
                )
            text = str(raw.get("word", raw.get("text", ""))).strip()
            if not text or _is_special_token(text):
                continue
            if "start" not in raw or "end" not in raw:
                # Partial word timing is worse than none: it would silently
                # produce a transcript with holes in the speaker alignment.
                return []
            try:
                start = parse_timestamp(raw["start"])
                end = parse_timestamp(raw["end"])
            except ValueError:
                return []
            words.append(Word(text=text, start=start, end=end, probability=_as_float(raw.get("probability"), 1.0)))
        return words


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    return _as_float(os.environ.get(key), default)
