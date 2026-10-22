"""Worker behavior tests using fake ML modules (no torch/ctranslate2 needed).

Covers: precision resolution policy, removal of the hidden INT8 override,
fatal chunk failures, diarization device requirements, and execution metadata.
"""

import json
import sys
import types
from pathlib import Path

import pytest

import diarize_worker
import whisper_worker


UUID_A = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c"


# --- Fake ML modules -----------------------------------------------------------

def _install_fake_ctranslate2(monkeypatch, supported):
    mod = types.ModuleType("ctranslate2")
    mod.get_supported_compute_types = lambda device, device_index=0: tuple(supported)
    monkeypatch.setitem(sys.modules, "ctranslate2", mod)


class _FakeWord:
    def __init__(self, text, start, end):
        self.word = text
        self.start = start
        self.end = end
        self.probability = 0.9


class _FakeSegment:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end
        self.words = [_FakeWord(text, start, end)]


class _FakeInfo:
    def __init__(self, language="en"):
        self.language = language
        self.language_probability = 0.99
        self.duration = 10.0


class _FakeModel:
    """WhisperModel stand-in. fail_on_chunk makes the 2nd chunk raise."""

    def __init__(self, model_size, device, compute_type):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.calls = 0
        self.fail_on_chunk = False

    def transcribe(self, path, **kwargs):
        self.calls += 1
        if self.fail_on_chunk and self.calls >= 2:
            raise RuntimeError("CUDA error: out of memory")
        return iter([_FakeSegment("hello", 0.0, 1.0)]), _FakeInfo()


def _install_fake_faster_whisper(monkeypatch, model_instance):
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = lambda model_size, device, compute_type: (
        model_instance or _FakeModel(model_size, device, compute_type)
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


# --- Precision policy -----------------------------------------------------------

def test_auto_resolves_pascal_style_support_to_int8_float32(monkeypatch):
    # Pascal (CC 6.1) runtimes support int8_float32 but not int8_float16.
    _install_fake_ctranslate2(monkeypatch, ["float32", "int8_float32"])
    assert whisper_worker._resolve_compute_type("auto", "cuda") == "int8_float32"


def test_auto_resolves_modern_gpu_to_int8_float16(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, ["float32", "float16", "int8", "int8_float32", "int8_float16"])
    assert whisper_worker._resolve_compute_type("auto", "cuda") == "int8_float16"


def test_auto_cpu_prefers_int8(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, ["float32", "int8"])
    assert whisper_worker._resolve_compute_type("auto", "cpu") == "int8"


def test_explicit_unsupported_compute_type_fails(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, ["float32", "int8_float32"])
    with pytest.raises(RuntimeError, match="UNSUPPORTED_COMPUTE_TYPE"):
        whisper_worker._resolve_compute_type("float16", "cuda")


def test_explicit_supported_compute_type_passes_through(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, ["float32", "int8_float32"])
    assert whisper_worker._resolve_compute_type("int8_float32", "cuda") == "int8_float32"


def test_no_hidden_int8_override(monkeypatch, tmp_path):
    # Historic bug: the worker unconditionally forced compute_type="int8" on
    # cuda. Now an explicit float16 request must reach WhisperModel unchanged.
    monkeypatch.setattr(whisper_worker, "_get_audio_duration", lambda p: 10.0)
    monkeypatch.setattr(whisper_worker, "_run_silero_vad", lambda *a, **k: [])
    monkeypatch.setattr(whisper_worker, "_tool", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(whisper_worker, "_extract_chunk", lambda *a: None)
    monkeypatch.setattr(whisper_worker, "_cuda_device_name", lambda d: None)
    _install_fake_ctranslate2(monkeypatch, ["float32", "float16"])

    captured = {}

    def _make(model_size, device, compute_type):
        m = _FakeModel(model_size, device, compute_type)
        captured["compute_type"] = compute_type
        captured["device"] = device
        return m

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = _make
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)

    args_file = tmp_path / "args.json"
    result_file = tmp_path / "result.json"
    segments_file = tmp_path / "segments.ndjson"
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    args_file.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "language": "en",
                "model_size": "large-v3-turbo",
                "device": "cuda",
                "compute_type": "float16",
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["whisper_worker.py", str(args_file), str(result_file), str(segments_file)]
    )
    whisper_worker.main()
    result = json.loads(result_file.read_text())
    assert result["status"] == "ok"
    assert captured["compute_type"] == "float16"  # explicit choice honored, no override
    assert result["execution"]["compute_type"] == "float16"
    assert result["execution"]["requested_compute_type"] == "float16"
    assert result["execution"]["device"] == "cuda"


def test_failed_chunk_is_fatal_not_skipped(monkeypatch, tmp_path):
    # A CUDA OOM mid-transcription must fail the whole job — never produce an
    # apparently-complete transcript with a hole in it.
    monkeypatch.setattr(whisper_worker, "_get_audio_duration", lambda p: 20.0)
    monkeypatch.setattr(
        whisper_worker, "_run_silero_vad", lambda *a, **k: [{"start": 0.0, "end": 15.0}, {"start": 15.0, "end": 30.0}]
    )
    monkeypatch.setattr(whisper_worker, "_tool", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(whisper_worker, "_extract_chunk", lambda *a: None)
    monkeypatch.setattr(whisper_worker, "_cuda_device_name", lambda d: None)
    _install_fake_ctranslate2(monkeypatch, ["float32", "int8_float32"])

    failing = _FakeModel("m", "cuda", "int8_float32")
    failing.fail_on_chunk = True
    _install_fake_faster_whisper(monkeypatch, failing)

    args_file = tmp_path / "args.json"
    result_file = tmp_path / "result.json"
    segments_file = tmp_path / "segments.ndjson"
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    args_file.write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "language": "en",
                "model_size": "large-v3-turbo",
                "device": "cuda",
                "compute_type": "auto",
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["whisper_worker.py", str(args_file), str(result_file), str(segments_file)]
    )
    with pytest.raises(SystemExit):
        whisper_worker.main()
    result = json.loads(result_file.read_text())
    assert result["status"] == "error"
    assert "out of memory" in result["payload"]


# --- Diarize worker device handling --------------------------------------------

def _run_diarize(monkeypatch, tmp_path, args_dict, fake_torch, extra_modules=None):
    args_file = tmp_path / "args.json"
    result_file = tmp_path / "result.json"
    args_file.write_text(json.dumps(args_dict))
    monkeypatch.setattr(sys, "argv", ["diarize_worker.py", str(args_file), str(result_file)])
    if fake_torch is not None:
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
    if extra_modules:
        for name, mod in extra_modules.items():
            monkeypatch.setitem(sys.modules, name, mod)
    diarize_worker.main()
    return json.loads(result_file.read_text())


def _fake_torch_module(cuda_available):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    serialization = types.ModuleType("torch.serialization")
    serialization.add_safe_globals = lambda x: None
    torch.serialization = serialization
    torch.torch_version = types.ModuleType("torch.torch_version")
    torch.torch_version.TorchVersion = object
    torch.load = lambda *a, **k: None
    torch.device = lambda d: d
    return torch


def test_diarize_unknown_device_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_diarize(
            monkeypatch,
            tmp_path,
            {"audio_path": "x.wav", "hf_token": "t", "device": "rocm"},
            None,
        )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "error"
    assert result["code"] == "RUNTIME_INCOMPATIBLE"


def test_diarize_cuda_required_no_silent_cpu_fallback(monkeypatch, tmp_path):
    # CUDA requested but unavailable -> loud failure, never CPU fallback.
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.set_audio_backend = lambda *a, **k: None
    torchaudio.get_audio_backend = lambda: "soundfile"
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = object
    pyannote_core = types.ModuleType("pyannote.core")
    pyannote_core.Annotation = object
    with pytest.raises(SystemExit):
        _run_diarize(
            monkeypatch,
            tmp_path,
            {"audio_path": "x.wav", "hf_token": "t", "device": "cuda"},
            _fake_torch_module(cuda_available=False),
            extra_modules={
                "torchaudio": torchaudio,
                "pyannote.audio": pyannote_audio,
                "pyannote.core": pyannote_core,
            },
        )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "error"
    assert result["code"] == "CUDA_UNAVAILABLE"


def test_diarize_cpu_honored_even_when_cuda_present(monkeypatch, tmp_path):
    # Explicit CPU must run on CPU even if CUDA is available.
    torch_mod = _fake_torch_module(cuda_available=True)

    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = object
    pyannote_core = types.ModuleType("pyannote.core")
    pyannote_core.Annotation = object
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.set_audio_backend = lambda *a, **k: None
    torchaudio.get_audio_backend = lambda: "soundfile"
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setitem(sys.modules, "pyannote.core", pyannote_core)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)

    # Pipeline.from_pretrained returns an object whose .to(device) records the device.
    used = {}

    class _Pipeline:
        @staticmethod
        def from_pretrained(*a, **k):
            return _Pipeline()

        def to(self, device):
            used["device"] = device
            return self

        def __call__(self, audio, **kwargs):
            turn = types.SimpleNamespace(start=0.0, end=2.0)

            class _Ann:
                def itertracks(self, yield_label=True):
                    return iter([(turn, "_", "SPEAKER_00")])

            return _Ann()

    pyannote_audio.Pipeline = _Pipeline

    result = _run_diarize(
        monkeypatch,
        tmp_path,
        {"audio_path": "x.wav", "hf_token": "t", "device": "cpu"},
        torch_mod,
    )
    assert result["status"] == "ok"
    assert used["device"] == "cpu"
    assert result["execution"]["device"] == "cpu"
    assert result["payload"] == [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
