"""Proof that routing faster-whisper through the engine layer changes nothing.

`whisper_worker.py` is untouched, so its payload is still the reference output.
These tests take a payload shaped exactly as the worker writes it and assert the
engine layer hands app.py back a byte-identical structure.
"""

import json

import pytest

from engines import ENGINE_REGISTRY, FasterWhisperEngine, WhisperCppHttpEngine, get_engine, resolve_engine_name
from engines.base import TranscriptionEngine
from engines.schema import TranscribeOptions, TranscriptResult

# Verbatim shape of whisper_worker.py's {"status": "ok", "payload": ...} body.
WORKER_PAYLOAD = {
    "segments": [
        {
            "start": 0.32,
            "end": 3.12,
            "text": "So tell me about the fieldwork.",
            "words": [
                {"text": "So", "start": 0.32, "end": 0.58, "probability": 0.9912},
                {"text": "tell", "start": 0.58, "end": 0.86, "probability": 0.9803},
                {"text": "fieldwork.", "start": 1.6, "end": 3.12, "probability": 0.8123},
            ],
        },
        {
            "start": 4.5,
            "end": 12.48,
            "text": "It went well, mostly.",
            "words": [{"text": "It", "start": 4.5, "end": 4.72, "probability": 0.9765}],
        },
    ],
    "info": {"language": "en", "language_probability": 0.987, "duration": 12.48},
}


def test_worker_payload_round_trips_byte_identically():
    result = TranscriptResult.from_payload(WORKER_PAYLOAD, engine="faster_whisper")
    assert result.to_payload() == WORKER_PAYLOAD
    assert json.dumps(result.to_payload(), indent=2) == json.dumps(WORKER_PAYLOAD, indent=2)


def test_round_trip_is_idempotent():
    once = TranscriptResult.from_payload(WORKER_PAYLOAD).to_payload()
    twice = TranscriptResult.from_payload(once).to_payload()
    assert once == twice


def test_segment_payload_keys_are_exactly_what_app_py_reads():
    payload = TranscriptResult.from_payload(WORKER_PAYLOAD).to_payload()
    for segment in payload["segments"]:
        assert list(segment) == ["start", "end", "text", "words"]
        for word in segment["words"]:
            assert list(word) == ["text", "start", "end", "probability"]


def test_segments_with_no_words_survive():
    payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "hm"}], "info": {}}
    result = TranscriptResult.from_payload(payload)
    assert result.to_payload()["segments"][0]["words"] == []
    assert result.has_word_timestamps is False


def test_info_defaults_match_the_worker():
    """The worker writes `language or "en"` and rounds the same way."""
    result = TranscriptResult(duration=12.4849, language_probability=0.98765)
    assert result.info_payload() == {"language": "en", "language_probability": 0.988, "duration": 12.48}


def test_to_dict_is_the_richer_view_and_does_not_leak_into_to_payload():
    result = TranscriptResult.from_payload(WORKER_PAYLOAD, engine="faster_whisper")
    result.warnings.append("something")
    rich = result.to_dict()

    assert rich["engine"] == "faster_whisper"
    assert rich["warnings"] == ["something"]
    assert "engine" not in result.to_payload()
    assert "warnings" not in result.to_payload()["info"]


def test_missing_words_key_and_word_alias_both_parse():
    payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": [
        {"word": " hi ", "start": 0.0, "end": 1.0}
    ]}], "info": {}}
    word = TranscriptResult.from_payload(payload).segments[0].words[0]
    assert word.text == "hi"
    assert word.word == "hi"
    assert word.probability == 1.0


def test_worker_args_match_the_legacy_args_file():
    options = TranscribeOptions(language="en", model="large-v3-turbo", device="cuda", compute_type="float16")
    args = options.worker_args("/tmp/a.wav")
    assert set(args) == {
        "audio_path", "language", "model_size", "device", "compute_type", "beam_size",
        "temperature", "chunk_length", "no_speech_threshold", "hotwords", "condition_on_prev",
    }


# ── registry ────────────────────────────────────────────────────────


def test_default_engine_is_faster_whisper(monkeypatch):
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    assert resolve_engine_name() == "faster_whisper"
    assert isinstance(get_engine(), FasterWhisperEngine)


def test_env_var_selects_whisper_cpp(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "  WHISPER_CPP ")
    assert isinstance(get_engine(), WhisperCppHttpEngine)


def test_unknown_engine_is_rejected(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "vosk")
    with pytest.raises(ValueError, match="Unknown transcription engine"):
        get_engine()


@pytest.mark.parametrize("factory", ENGINE_REGISTRY.values())
def test_every_registered_engine_satisfies_the_protocol(factory):
    engine = factory()
    assert isinstance(engine, TranscriptionEngine)
    assert engine.name
    assert callable(engine.health)
    assert callable(engine.is_available)
