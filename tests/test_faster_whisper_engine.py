"""The faster-whisper engine must drive whisper_worker.py exactly as app.py did.

No model is loaded here — `spawn_worker` is replaced by a stub that writes the
files the real worker would write.
"""

import json

import pytest

from engines.errors import EngineUnavailableError, TranscriptionError
from engines.faster_whisper_engine import FasterWhisperEngine
from engines.schema import EngineHooks, TranscribeOptions
from tests.test_schema_roundtrip import WORKER_PAYLOAD


class FakeProc:
    def __init__(self, pid=4242, returncode=0):
        self.pid = pid
        self.returncode = returncode
        self.stderr = None

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


@pytest.fixture
def installed_engine(tmp_path):
    """A repo layout the engine considers healthy."""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    (tmp_path / "whisper_worker.py").touch()
    engine = FasterWhisperEngine(base_dir=tmp_path)
    engine.python_bin = venv_python
    return engine


def stub_worker(monkeypatch, *, result: dict, segment_lines: list[dict] | None = None, proc=None):
    captured = {}

    def _spawn(cmd, env):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["args"] = json.loads(open(cmd[2], encoding="utf-8").read())
        with open(cmd[3], "w", encoding="utf-8") as handle:
            json.dump(result, handle)
        with open(cmd[4], "w", encoding="utf-8") as handle:
            for line in segment_lines or []:
                handle.write(json.dumps(line) + "\n")
        return proc or FakeProc()

    monkeypatch.setattr("engines.faster_whisper_engine.spawn_worker", _spawn)
    return captured


def test_output_matches_the_worker_payload_exactly(installed_engine, monkeypatch):
    stub_worker(monkeypatch, result={"status": "ok", "payload": WORKER_PAYLOAD})

    result = installed_engine.transcribe("/tmp/interview.wav", TranscribeOptions())

    assert result.engine == "faster_whisper"
    assert result.to_payload() == WORKER_PAYLOAD
    assert result.has_word_timestamps is True
    assert result.wall_time_s >= 0


def test_args_file_carries_the_legacy_keys(installed_engine, monkeypatch):
    captured = stub_worker(monkeypatch, result={"status": "ok", "payload": WORKER_PAYLOAD})
    options = TranscribeOptions(language="th", model="large-v3-turbo", beam_size=5, hotwords="pyannote")

    installed_engine.transcribe("/tmp/a.wav", options)

    assert captured["args"] == {
        "audio_path": "/tmp/a.wav",
        "language": "th",
        "model_size": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "temperature": 0.0,
        "chunk_length": 30,
        "no_speech_threshold": 0.6,
        "hotwords": "pyannote",
        "condition_on_prev": False,
    }


def test_live_segments_are_streamed_through_the_hook(installed_engine, monkeypatch):
    lines = [{"start": 0.0, "end": 1.0, "text": "one", "words": []},
             {"start": 1.0, "end": 2.0, "text": "two", "words": []}]
    stub_worker(monkeypatch, result={"status": "ok", "payload": WORKER_PAYLOAD}, segment_lines=lines)

    seen = []
    installed_engine.transcribe(
        "/tmp/a.wav", TranscribeOptions(hooks=EngineHooks(on_segment=seen.append))
    )
    assert seen == lines


def test_pid_hooks_fire_around_the_subprocess(installed_engine, monkeypatch):
    stub_worker(monkeypatch, result={"status": "ok", "payload": WORKER_PAYLOAD}, proc=FakeProc(pid=99))

    started, ended = [], []
    installed_engine.transcribe(
        "/tmp/a.wav",
        TranscribeOptions(hooks=EngineHooks(on_process_start=started.append, on_process_end=ended.append)),
    )
    assert started == [99] and ended == [99]


def test_worker_error_status_propagates(installed_engine, monkeypatch):
    stub_worker(monkeypatch, result={"status": "error", "payload": "CUDA out of memory"})

    with pytest.raises(TranscriptionError, match="CUDA out of memory"):
        installed_engine.transcribe("/tmp/a.wav", TranscribeOptions())


def test_missing_venv_is_engine_unavailable(tmp_path):
    engine = FasterWhisperEngine(base_dir=tmp_path)
    assert engine.is_available() is False
    with pytest.raises(EngineUnavailableError, match="venv not found"):
        engine.health()
