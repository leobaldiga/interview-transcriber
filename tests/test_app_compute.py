"""End-to-end app tests (mocked workers/discovery): snapshot isolation, device
routing, submission validation, and the compute API surface."""

import csv
import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app
import compute_devices
import compute_settings as cs


UUID_A = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c"  # RTX 3060
UUID_B = "GPU-60ce4d58-ba50-b3fa-82da-3e51b259e857"  # GTX 1050 Ti


@pytest.fixture
def two_gpus(monkeypatch):
    def _fake_list(timeout=5.0):
        return [
            compute_devices.DeviceInfo("cuda", UUID_A, "NVIDIA GeForce RTX 3060", 0, "00000000:01:00.0", "580.178.04", 12288, 11943, 345),
            compute_devices.DeviceInfo("cuda", UUID_B, "NVIDIA GeForce GTX 1050 Ti", 1, "00000000:02:00.0", "580.178.04", 4096, 144, 3952),
        ]

    monkeypatch.setattr(compute_devices, "list_cuda_devices", _fake_list)


@pytest.fixture
def client():
    with TestClient(app.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_state():
    """Tests share the app module's in-memory job dict and settings store —
    reset both so test order never leaks state."""
    app._compute_store.path.unlink(missing_ok=True)
    app.jobs.clear()
    app._queue_order.clear()
    app._active_pids.clear()
    app._live_segments.clear()
    app._compute_tests.clear()
    app._compute_tests_order.clear()
    if app.JOBS_INDEX_PATH.exists():
        app.JOBS_INDEX_PATH.unlink()
    yield


def _wait_for_status(client, job_id, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    return job


def _upload(client, **extra):
    files = {"file": ("interview.mp3", b"ID3 fake audio", "audio/mpeg")}
    return client.post("/api/transcribe", files=files, data=extra)


# --- Submission + snapshots ----------------------------------------------------

def test_legacy_submission_unconfigured_multiple_gpus_rejected(two_gpus, client):
    # No saved selection + multiple GPUs -> the job must not be accepted on a guess.
    resp = _upload(client, language="en")
    assert resp.status_code == 400
    assert "DEVICE_NOT_CONFIGURED" in resp.text or "Multiple GPUs" in resp.text


def test_submission_after_selecting_device(two_gpus, client):
    resp = client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    assert resp.status_code == 200
    resp = _upload(client, language="en")
    assert resp.status_code == 200
    job = client.get(f"/api/jobs/{resp.json()['job_id']}").json()
    assert job["compute"]["transcription"]["device_id"] == UUID_B
    assert job["compute"]["diarization"]["device_id"] == UUID_B  # same-as resolved
    assert job["compute"]["transcription"]["requested_compute_type"] == "auto"


def test_per_job_overrides(two_gpus, client):
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    resp = _upload(
        client,
        language="en",
        transcription_device=UUID_A,
        diarization_device="cpu",
        whisper_compute_type="int8_float32",
    )
    assert resp.status_code == 200
    job = client.get(f"/api/jobs/{resp.json()['job_id']}").json()
    assert job["compute"]["transcription"]["device_id"] == UUID_A
    assert job["compute"]["diarization"]["backend"] == "cpu"
    assert job["compute"]["transcription"]["requested_compute_type"] == "int8_float32"


def test_invalid_override_rejected(two_gpus, client):
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    resp = _upload(client, transcription_device="gpu1")
    assert resp.status_code == 400
    resp = _upload(client, transcription_device=UUID_A, whisper_compute_type="superfast")
    assert resp.status_code == 400


def test_snapshot_isolated_from_default_changes(two_gpus, client):
    """Changing saved defaults must NOT affect already-accepted queued jobs."""
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    # Block the queue so the job stays queued while we change defaults.
    release = app._gpu_work_lock.acquire()
    try:
        resp = _upload(client, language="en")
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        _wait_for_status(client, job_id, {"queued"})  # ensure enqueued

        # Change the default device.
        client.put(
            "/api/compute/settings",
            json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_A}}},
        )
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["compute"]["transcription"]["device_id"] == UUID_B  # snapshot unchanged
    finally:
        app._gpu_work_lock.release()
        _wait_for_status(client, job_id, {"complete", "error", "transcribing", "diarizing", "merging", "building_outputs"})


# --- Fake job through the queue -------------------------------------------------

def test_fake_job_routes_and_records_execution(two_gpus, client, monkeypatch):
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )

    captured = {}

    def fake_transcribe(audio_path, language="en", job_id=None, model_size=None, adv=None, compute=None):
        captured["transcribe_compute"] = compute
        segments = [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}]
        info = {"duration": 1.0, "language": "en"}
        execution = {"device": "cuda", "device_id": UUID_B, "compute_type": "int8_float32"}
        return segments, info, execution

    def fake_diarize(audio_path, num_speakers=None, job=None, adv=None, compute=None):
        captured["diarize_compute"] = compute
        turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        return turns, {"device": "cuda", "device_id": UUID_B}

    monkeypatch.setattr(app, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(app, "diarize_audio", fake_diarize)

    resp = _upload(client, language="en")
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    job = _wait_for_status(client, job_id, {"complete", "error"}, timeout=15)
    assert job["status"] == "complete", job.get("error")
    assert job["execution"]["transcription"]["device_id"] == UUID_B
    assert job["execution"]["diarization"]["device_id"] == UUID_B
    # Workers received the job's snapshot, not the current defaults.
    assert captured["transcribe_compute"]["transcription"]["device_id"] == UUID_B
    assert captured["diarize_compute"]["diarization"]["device_id"] == UUID_B
    assert job["files"]["txt"]


def test_snapshot_revalidation_missing_gpu(two_gpus, client, monkeypatch):
    """A job accepted for a GPU that vanishes before execution fails — no fallback."""
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    # Block the queue so the job stays queued while we remove GPU B.
    app._gpu_work_lock.acquire()
    try:
        resp = _upload(client, language="en")
        job_id = resp.json()["job_id"]
        assert resp.status_code == 200

        def _only_a(timeout=5.0):
            return [compute_devices.DeviceInfo("cuda", UUID_A, "RTX 3060", 0, None, "580.178.04", 12288, 11943, 345)]

        monkeypatch.setattr(compute_devices, "list_cuda_devices", _only_a)
    finally:
        app._gpu_work_lock.release()

    job = _wait_for_status(client, job_id, {"error", "complete"}, timeout=15)
    assert job["status"] == "error"
    assert "DEVICE_NOT_AVAILABLE" in (job.get("error") or "")
    assert "No fallback" in (job.get("error") or "")


# --- Status/VRAM/compute endpoints ------------------------------------------------

def test_status_reports_devices_and_selection(two_gpus, client):
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    status = client.get("/api/status").json()
    assert status["gpu"] is True
    assert len(status["devices"]) == 2
    assert status["selected_device"] == UUID_B
    assert status["gpu_detail"]["name"] == "NVIDIA GeForce RTX 3060"  # legacy = first GPU


def test_vram_reports_selected_device_not_pooled(two_gpus, client):
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    vram = client.get("/api/vram").json()
    assert vram["selected_device"] == UUID_B
    assert vram["vram_total"] == "4.0 GB"  # 1050 Ti, not the pooled/first GPU
    assert len(vram["devices"]) == 2


def test_vram_unconfigured_multi_gpu_does_not_guess_first_gpu(two_gpus, client):
    """Multi-GPU + no selection: inventory returned, but no device misattributed."""
    vram = client.get("/api/vram").json()
    assert vram["selected_device"] is None
    assert vram["vram_total"] is None
    assert vram["vram_used"] is None
    assert vram["vram_pct"] == 0
    assert len(vram["devices"]) == 2  # inventory preserved


def test_vram_cpu_selection_reports_cpu_backend(two_gpus, client):
    """A valid CPU selection is not 'no valid device': backend is reported."""
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cpu", "id": None}}},
    )
    vram = client.get("/api/vram").json()
    assert vram["selected_backend"] == "cpu"
    assert vram["selected_device"] is None  # no GPU to point at
    assert vram["vram_total"] is None  # no GPU VRAM to report
    assert len(vram["devices"]) == 2  # GPU inventory still shown


def test_vram_configured_gpu_missing_does_not_guess_first_gpu(two_gpus, client, monkeypatch):
    """A configured GPU that disappears: no first-GPU fallback in telemetry."""
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_A}}},
    )
    # Simulate UUID_A vanishing from the inventory (only UUID_B remains).
    monkeypatch.setattr(
        app.compute_devices,
        "list_cuda_devices",
        lambda timeout=5.0: [
            compute_devices.DeviceInfo(
                "cuda", UUID_B, "NVIDIA GeForce GTX 1050 Ti", 1, "00000000:02:00.0", "580.178.04", 4096, 144, 3952
            )
        ],
    )
    vram = client.get("/api/vram").json()
    assert vram["selected_device"] is None
    assert vram["vram_total"] is None
    assert vram["vram_used"] is None
    assert len(vram["devices"]) == 1


def test_compute_settings_endpoints(two_gpus, client):
    get = client.get("/api/compute/settings").json()
    assert get["settings"]["transcription"]["device"] is None
    assert get["needs_selection"] is True

    resp = client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_A}}},
    )
    assert resp.status_code == 200

    get = client.get("/api/compute/settings").json()
    assert get["resolved"]["id"] == UUID_A
    assert get["resolved_device_name"] == "NVIDIA GeForce RTX 3060"
    assert get["needs_selection"] is False

    # Invalid document rejected with 400.
    resp = client.put("/api/compute/settings", json={"schema_version": 99})
    assert resp.status_code == 400


def test_compute_devices_endpoint(two_gpus, client):
    inv = client.get("/api/compute/devices").json()
    assert inv["cpu"]["backend"] == "cpu"
    assert len(inv["cuda_devices"]) == 2
    by_id = {d["id"]: d for d in inv["cuda_devices"]}
    assert by_id[UUID_B]["memory_free_mib"] == 3952


def test_malformed_settings_file_surfaced(two_gpus, client, monkeypatch):
    # Corrupt the settings file the store points at; the API must explain it.
    path = app._compute_store.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    try:
        resp = client.get("/api/compute/settings")
        assert resp.status_code == 409
        body = resp.json()
        assert body["recoverable_by_save"] is True
    finally:
        path.unlink()


def test_worker_env_routes_to_snapshot_device(two_gpus, client):
    """The subprocess launch environment pins the job snapshot's GPU UUID."""
    client.put(
        "/api/compute/settings",
        json={"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}},
    )
    resp = _upload(client, language="en", transcription_device=UUID_A)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    compute = app.jobs[job_id]["compute"]
    stage_device = {
        "backend": compute["transcription"]["backend"],
        "id": compute["transcription"].get("device_id"),
    }
    py_exe = app._worker_python_or_none(app.worker_runtime.ROLE_WHISPER)
    if py_exe is None:
        pytest.skip("no worker interpreter in sandbox")
    env = app.worker_runtime.build_worker_env(py_exe, stage_device)
    assert env["CUDA_VISIBLE_DEVICES"] == UUID_A  # the override, not the saved default
