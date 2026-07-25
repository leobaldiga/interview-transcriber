import json

import pytest
import requests

from engines.errors import (
    InferenceHTTPError,
    MalformedResponseError,
    ModelNotLoadedError,
    ServerUnreachableError,
    TranscriptionTimeoutError,
)
from engines.schema import TranscribeOptions
from engines.whisper_cpp_engine import WhisperCppHttpEngine


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Replays a scripted sequence of responses or exceptions."""

    def __init__(self, get=None, post=None):
        self._get = list(get or [])
        self._post = list(post or [])
        self.get_calls = []
        self.post_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._next(self._get)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self._next(self._post)

    @staticmethod
    def _next(queue):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_engine(session, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1:8081")
    kwargs.setdefault("retries", 3)
    return WhisperCppHttpEngine(session=session, **kwargs)


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "interview.wav"
    path.write_bytes(b"RIFF____WAVEfmt ")
    return path


# ── (a) server not running ──────────────────────────────────────────


def test_connection_refused_is_server_unreachable():
    engine = make_engine(FakeSession(get=[requests.ConnectionError("refused")]))
    with pytest.raises(ServerUnreachableError, match="is the container running"):
        engine.health()
    assert engine.is_available() is False


# ── (b) up but no model ─────────────────────────────────────────────


def test_503_health_is_model_not_loaded():
    session = FakeSession(get=[FakeResponse(503, {"status": "loading model"})])
    with pytest.raises(ModelNotLoadedError, match="no model loaded"):
        make_engine(session).health()


def test_200_loading_status_is_model_not_loaded():
    session = FakeSession(get=[FakeResponse(200, {"status": "loading model"})])
    with pytest.raises(ModelNotLoadedError, match="still loading"):
        make_engine(session).health()


def test_healthy_server_passes():
    session = FakeSession(get=[FakeResponse(200, {"status": "ok"})])
    assert make_engine(session).is_available() is True


def test_missing_health_route_is_treated_as_up():
    """Older builds have no /health; a 404 still proves something is listening."""
    session = FakeSession(get=[FakeResponse(404, text="Not Found")])
    make_engine(session).health()


# ── (c) HTTP error from inference ───────────────────────────────────


def test_inference_http_error_carries_status_and_body(audio):
    session = FakeSession(
        get=[FakeResponse(200, {"status": "ok"})],
        post=[FakeResponse(500, {"error": "failed to process audio"})],
    )
    with pytest.raises(InferenceHTTPError) as excinfo:
        make_engine(session).transcribe(audio, TranscribeOptions())
    assert excinfo.value.status_code == 500
    assert "failed to process audio" in excinfo.value.body


def test_400_is_not_retried(audio):
    session = FakeSession(get=[FakeResponse(200, {"status": "ok"})], post=[FakeResponse(400, {"error": "no file"})])
    with pytest.raises(InferenceHTTPError):
        make_engine(session).transcribe(audio, TranscribeOptions())
    assert len(session.post_calls) == 1


def test_503_during_inference_is_model_not_loaded(audio):
    session = FakeSession(get=[FakeResponse(200, {"status": "ok"})], post=[FakeResponse(503, {"status": "loading model"})])
    with pytest.raises(ModelNotLoadedError):
        make_engine(session).transcribe(audio, TranscribeOptions())


# ── (d) timeout ─────────────────────────────────────────────────────


def test_read_timeout_is_reported_as_timeout(audio):
    session = FakeSession(get=[FakeResponse(200, {"status": "ok"})], post=[requests.Timeout("read timed out")])
    with pytest.raises(TranscriptionTimeoutError, match="WHISPER_CPP_TIMEOUT"):
        make_engine(session).transcribe(audio, TranscribeOptions())
    assert len(session.post_calls) == 1


def test_health_timeout_is_reported_as_timeout():
    session = FakeSession(get=[requests.Timeout("connect timed out")])
    with pytest.raises(TranscriptionTimeoutError):
        make_engine(session).health()


# ── (e) malformed response ──────────────────────────────────────────


def test_non_json_body_is_malformed(audio):
    session = FakeSession(get=[FakeResponse(200, {"status": "ok"})], post=[FakeResponse(200, None, text="<html>502</html>")])
    with pytest.raises(MalformedResponseError, match="non-JSON"):
        make_engine(session).transcribe(audio, TranscribeOptions())


def test_json_array_body_is_malformed(audio):
    session = FakeSession(get=[FakeResponse(200, {"status": "ok"})], post=[FakeResponse(200, ["nope"])])
    with pytest.raises(MalformedResponseError, match="expected a JSON object"):
        make_engine(session).transcribe(audio, TranscribeOptions())


# ── retry policy ────────────────────────────────────────────────────


def test_transient_connection_errors_are_retried_then_give_up(audio, monkeypatch):
    monkeypatch.setattr("engines.whisper_cpp_engine.time.sleep", lambda _: None)
    session = FakeSession(get=[FakeResponse(200, {"status": "ok"})], post=[requests.ConnectionError("reset")])
    with pytest.raises(ServerUnreachableError, match="after 3 attempts"):
        make_engine(session).transcribe(audio, TranscribeOptions())
    assert len(session.post_calls) == 3


def test_retry_recovers_when_the_server_comes_back(audio, monkeypatch, load_fixture):
    monkeypatch.setattr("engines.whisper_cpp_engine.time.sleep", lambda _: None)
    session = FakeSession(
        get=[FakeResponse(200, {"status": "ok"})],
        post=[requests.ConnectionError("reset"), FakeResponse(200, load_fixture("whisper_cpp_verbose_json.json"))],
    )
    result = make_engine(session).transcribe(audio, TranscribeOptions())
    assert len(session.post_calls) == 2
    assert result.word_count == 10


# ── request shape ───────────────────────────────────────────────────


def test_inference_posts_multipart_file_field(audio, load_fixture):
    session = FakeSession(
        get=[FakeResponse(200, {"status": "ok"})],
        post=[FakeResponse(200, load_fixture("whisper_cpp_verbose_json.json"))],
    )
    make_engine(session).transcribe(audio, TranscribeOptions(language="en"))

    url, kwargs = session.post_calls[0]
    assert url == "http://127.0.0.1:8081/inference"
    assert "file" in kwargs["files"]
    assert kwargs["data"]["response_format"] == "verbose_json"
    assert kwargs["timeout"][1] == 3600.0


def test_non_wav_input_warns_about_convert(tmp_path, load_fixture):
    audio = tmp_path / "interview.m4a"
    audio.write_bytes(b"\x00\x00\x00 ftypM4A ")
    session = FakeSession(
        get=[FakeResponse(200, {"status": "ok"})],
        post=[FakeResponse(200, load_fixture("whisper_cpp_verbose_json.json"))],
    )
    result = make_engine(session).transcribe(audio, TranscribeOptions())
    assert any("--convert" in w for w in result.warnings)


def test_base_url_comes_from_env(monkeypatch):
    monkeypatch.setenv("WHISPER_CPP_URL", "http://storm1:9000/")
    assert WhisperCppHttpEngine().base_url == "http://storm1:9000"
