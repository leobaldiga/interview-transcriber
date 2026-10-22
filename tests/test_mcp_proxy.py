"""Tests for the MCP proxy's GPU-selection surface (mcp_proxy.py).

The proxy is an integration surface (no unit tests existed for it before).
These tests mock the httpx client so the tool functions run against canned
app responses — no app process, no GPU.
"""

import asyncio
import types
from pathlib import Path

import pytest

import mcp_proxy

GPU_A = "GPU-60ce4d58-ba50-b3fa-82da-3e51b259e857"
GPU_B = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c"


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise mcp_proxy.httpx.HTTPStatusError("err", request=None, response=None)


class FakeClient:
    """Records requests; serves canned responses by URL substring."""

    requests: list[dict] = []
    responses: dict[str, object] = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        FakeClient.requests.append({"method": "GET", "url": url})
        for key, resp in FakeClient.responses.items():
            if key in url:
                return FakeResponse(resp)
        return FakeResponse({}, status_code=404)

    async def post(self, url, files=None, data=None, json=None):
        FakeClient.requests.append(
            {"method": "POST", "url": url, "data": data, "files": files}
        )
        for key, resp in FakeClient.responses.items():
            if key in url:
                return FakeResponse(resp)
        return FakeResponse({}, status_code=404)


@pytest.fixture()
def fake_http(monkeypatch):
    FakeClient.requests = []
    FakeClient.responses = {}
    fake_module = types.SimpleNamespace(
        AsyncClient=FakeClient,
        Timeout=lambda *a, **k: None,
        HTTPStatusError=Exception,
    )
    monkeypatch.setattr(mcp_proxy, "httpx", fake_module)
    return FakeClient


def run(coro):
    return asyncio.run(coro)


def _settings_payload(needs_selection=False, backend="cuda"):
    resolved = {"backend": backend, "id": GPU_A if backend == "cuda" else "cpu"}
    return {
        "settings": {
            "transcription": {"device": {"backend": backend, "id": GPU_A if backend == "cuda" else None}},
            "diarization": {"device": {"backend": "same_as_transcription", "id": None}},
        },
        "resolved": resolved,
        "resolved_device_name": "NVIDIA GeForce GTX 1050 Ti" if backend == "cuda" else None,
        "resolution_error": None,
        "needs_selection": needs_selection,
    }


# --- list_whisper_models messaging -------------------------------------------


def test_models_note_unconfigured(fake_http):
    fake_http.responses["/api/compute/settings"] = _settings_payload(needs_selection=True)
    out = run(mcp_proxy.list_whisper_models())
    assert "No transcription device is configured" in out["coresident_note"]
    assert out["selected_device"] is None


def test_models_note_explicit_cpu(fake_http):
    fake_http.responses["/api/compute/settings"] = _settings_payload(backend="cpu")
    out = run(mcp_proxy.list_whisper_models())
    assert "explicitly set to CPU" in out["coresident_note"]
    assert out["selected_device"]["backend"] == "cpu"


def test_models_note_selected_gpu(fake_http):
    fake_http.responses["/api/compute/settings"] = _settings_payload()
    out = run(mcp_proxy.list_whisper_models())
    assert "1050 Ti" in out["coresident_note"]
    assert "cannot coordinate with other processes" in out["coresident_note"]


# --- get_compute_selection ----------------------------------------------------


def test_get_compute_selection_roles(fake_http):
    fake_http.responses["/api/compute/settings"] = _settings_payload()
    fake_http.responses["/api/vram"] = {"devices": [{"id": GPU_A, "memory_free_mib": 3900}]}
    out = run(mcp_proxy.get_compute_selection())
    assert out["selected"]["id"] == GPU_A
    assert out["needs_selection"] is False
    assert out["diarization_setting"] == {"backend": "same_as_transcription", "id": None}
    assert out["devices"][0]["id"] == GPU_A


# --- transcribe_audio: override forwarding + VRAM warnings --------------------


def _audio_file(tmp_path):
    p = tmp_path / "interview.mp3"
    p.write_bytes(b"\x00" * 64)
    return p


def test_transcribe_forwards_overrides(fake_http, tmp_path):
    fake_http.responses["/api/vram"] = {"devices": [{"id": GPU_A, "memory_free_mib": 3900}]}
    fake_http.responses["/api/transcribe"] = {"job_id": "job-1"}
    out = run(
        mcp_proxy.transcribe_audio(
            str(_audio_file(tmp_path)),
            "large-v3-turbo",
            transcription_device=GPU_A,
            diarization_device="same",
            whisper_compute_type="int8_float32",
        )
    )
    assert out["job_id"] == "job-1"
    post = [r for r in fake_http.requests if r["method"] == "POST"][0]
    assert post["data"]["transcription_device"] == GPU_A
    assert post["data"]["diarization_device"] == "same"
    assert post["data"]["whisper_compute_type"] == "int8_float32"


def test_unknown_override_device_gets_no_pooled_vram_warning(fake_http, tmp_path):
    # The saved default (GPU_A) is nearly full; the override (GPU_B) is absent
    # from the inventory. The warning must NOT quote GPU_A's pooled numbers.
    fake_http.responses["/api/vram"] = {
        "devices": [{"id": GPU_A, "memory_free_mib": 200}],
        "vram_total": "4.0 GB",
        "vram_used": "3.8 GB",
    }
    fake_http.responses["/api/transcribe"] = {"job_id": "job-2"}
    out = run(
        mcp_proxy.transcribe_audio(
            str(_audio_file(tmp_path)),
            "large-v3-turbo",
            transcription_device=GPU_B,
        )
    )
    assert out["job_id"] == "job-2"
    assert out["warning"] is None


def test_override_low_vram_warns_from_that_device(fake_http, tmp_path):
    fake_http.responses["/api/vram"] = {
        "devices": [{"id": GPU_A, "memory_free_mib": 3900}, {"id": GPU_B, "memory_free_mib": 150}]
    }
    fake_http.responses["/api/transcribe"] = {"job_id": "job-3"}
    out = run(
        mcp_proxy.transcribe_audio(
            str(_audio_file(tmp_path)),
            "large-v3-turbo",
            transcription_device=GPU_B,
        )
    )
    assert out["warning"] is not None
    assert "150 MiB" in out["warning"]


def test_cpu_override_warns_slow(fake_http, tmp_path):
    fake_http.responses["/api/vram"] = {"devices": []}
    fake_http.responses["/api/transcribe"] = {"job_id": "job-4"}
    out = run(
        mcp_proxy.transcribe_audio(
            str(_audio_file(tmp_path)),
            "large-v3-turbo",
            transcription_device="cpu",
        )
    )
    assert "CPU" in out["warning"]


# --- get_job: execution metadata preservation ---------------------------------


def test_get_job_preserves_execution_metadata(fake_http):
    fake_http.responses["/api/jobs/"] = {
        "status": "complete",
        "filename": "interview.mp3",
        "language": "en",
        "files": {"txt": "outputs/MCP/j1/x.txt"},
        "execution": {
            "transcription": {
                "device": "cuda",
                "device_id": GPU_A,
                "device_name": "NVIDIA GeForce GTX 1050 Ti",
                "compute_type": "int8_float32",
                "elapsed_seconds": 42.1,
            },
            "diarization": {
                "device": "cuda",
                "device_id": GPU_A,
                "device_name": "NVIDIA GeForce GTX 1050 Ti",
                "elapsed_seconds": 10.5,
            },
        },
    }
    fake_http.responses["/api/preview/"] = {"content": "hello"}
    out = run(mcp_proxy.get_job("j1"))
    assert out["status"] == "done"
    assert out["execution"]["transcription"]["device_id"] == GPU_A
    assert out["execution"]["transcription"]["compute_type"] == "int8_float32"
    assert out["execution"]["diarization"]["elapsed_seconds"] == 10.5
