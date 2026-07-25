import pytest

from engines.errors import MalformedResponseError
from engines.whisper_cpp_engine import WhisperCppHttpEngine, normalize_language


@pytest.fixture
def engine():
    return WhisperCppHttpEngine(base_url="http://127.0.0.1:8089")


def test_verbose_json_maps_to_canonical_schema(engine, load_fixture):
    result = engine._normalize(load_fixture("whisper_cpp_verbose_json.json"))

    assert result.engine == "whisper_cpp"
    assert result.language == "en"
    assert result.language_probability == pytest.approx(0.987)
    assert result.duration == pytest.approx(12.48)
    assert result.has_word_timestamps is True
    assert result.warnings == []
    assert len(result.segments) == 2

    first = result.segments[0]
    assert first.id == 0
    assert first.text == "So tell me about the fieldwork."
    assert (first.start, first.end) == (pytest.approx(0.32), pytest.approx(3.12))
    assert first.avg_logprob == pytest.approx(-0.2411)
    assert [w.text for w in first.words] == ["So", "tell", "me", "about", "the", "fieldwork."]
    assert first.words[0].start == pytest.approx(0.32)
    assert first.words[0].probability == pytest.approx(0.9912)


def test_special_tokens_are_dropped(engine, load_fixture):
    result = engine._normalize(load_fixture("whisper_cpp_verbose_json.json"))
    texts = [w.text for seg in result.segments for w in seg.words]
    assert "[_BEG_]" not in texts
    assert texts.count("mostly.") == 1


def test_payload_matches_legacy_worker_shape(engine, load_fixture):
    payload = engine._normalize(load_fixture("whisper_cpp_verbose_json.json")).to_payload()

    assert set(payload) == {"segments", "info"}
    assert set(payload["info"]) == {"language", "language_probability", "duration"}
    for segment in payload["segments"]:
        assert set(segment) == {"start", "end", "text", "words"}
        for word in segment["words"]:
            assert set(word) == {"text", "start", "end", "probability"}


def test_missing_word_timestamps_warn_loudly(engine, load_fixture):
    result = engine._normalize(load_fixture("whisper_cpp_no_words.json"))

    assert result.has_word_timestamps is False
    assert result.word_count == 0
    assert any("without word-level" in w for w in result.warnings)
    assert result.language == "de"


def test_partial_word_timing_drops_the_whole_segment(engine):
    payload = {
        "duration": 3.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "half timed",
                "words": [{"word": "half", "start": 0.0, "end": 1.0}, {"word": "timed"}],
            }
        ],
    }
    result = engine._normalize(payload)
    assert result.segments[0].words == []
    assert result.has_word_timestamps is False


def test_plain_json_fallback_degrades_with_a_warning(engine):
    result = engine._normalize({"text": "  just some text  "})

    assert len(result.segments) == 1
    assert result.segments[0].text == "just some text"
    assert result.has_word_timestamps is False
    assert any("response_format=verbose_json" in w for w in result.warnings)


def test_duration_falls_back_to_last_segment_end(engine):
    result = engine._normalize({"segments": [{"start": 0.0, "end": 41.5, "text": "x"}]})
    assert result.duration == pytest.approx(41.5)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"segments": "nope"},
        {"segments": [{"text": "no timestamps"}]},
        {"segments": ["not an object"]},
        {"segments": [{"start": 0.0, "end": 1.0, "words": ["not an object"]}]},
    ],
)
def test_malformed_payloads_raise(engine, payload):
    with pytest.raises(MalformedResponseError):
        engine._normalize(payload)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("english", "en"),
        ("English", "en"),
        ("cantonese", "yue"),
        ("haitian creole", "ht"),
        ("lao", "lo"),
        ("en", "en"),
        ("yue", "yue"),
        ("auto", None),
        ("", None),
        (None, None),
    ],
)
def test_language_names_map_to_iso_codes(value, expected):
    assert normalize_language(value) == expected


def test_form_fields_use_lowercase_booleans_and_verbose_json():
    from engines.schema import TranscribeOptions

    fields = WhisperCppHttpEngine()._form_fields(TranscribeOptions(language=None, hotwords=" pyannote "))

    assert fields["response_format"] == "verbose_json"
    assert fields["language"] == "auto"
    # whisper.cpp's parse_str_to_bool only accepts true|1|yes|y — "True" reads false.
    assert fields["token_timestamps"] == "true"
    assert fields["no_timestamps"] == "false"
    assert fields["prompt"] == "pyannote"
