"""Tests for compute_settings: validation, persistence, snapshots, overrides."""

import json
import os
from pathlib import Path

import pytest

import compute_settings as cs


UUID_A = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c"
UUID_B = "GPU-60ce4d58-ba50-b3fa-82da-3e51b259e857"


# --- Device ref validation ---------------------------------------------------

def test_cpu_ref_roundtrip():
    ref = cs.validate_device_ref({"backend": "cpu", "id": None})
    assert ref == {"backend": "cpu", "id": None}


def test_cuda_ref_requires_full_uuid():
    with pytest.raises(cs.ComputeSettingsError):
        cs.validate_device_ref({"backend": "cuda", "id": "0"})
    with pytest.raises(cs.ComputeSettingsError):
        cs.validate_device_ref({"backend": "cuda", "id": None})
    ref = cs.validate_device_ref({"backend": "cuda", "id": UUID_A})
    assert ref == {"backend": "cuda", "id": UUID_A}


def test_unsupported_backend_rejected():
    with pytest.raises(cs.ComputeSettingsError):
        cs.validate_device_ref({"backend": "metal", "id": None})


def test_same_only_for_diarization():
    with pytest.raises(cs.ComputeSettingsError):
        cs.validate_device_ref({"backend": cs.SAME_AS_TRANSCRIPTION, "id": None})
    ok = cs.validate_device_ref({"backend": cs.SAME_AS_TRANSCRIPTION}, allow_same=True)
    assert ok == cs.SAME_DEVICE


def test_resolve_same_copies_transcription_device():
    resolved = cs.resolve_same(cs.SAME_DEVICE, {"backend": "cuda", "id": UUID_A})
    assert resolved == {"backend": "cuda", "id": UUID_A}
    # and it is a copy, not a shared reference
    resolved["id"] = "mutated"
    assert UUID_A


# --- Settings validation -----------------------------------------------------

def test_default_settings_shape():
    s = cs.default_settings()
    assert s["schema_version"] == 1
    assert s["transcription"]["engine"] == "faster_whisper"
    assert s["transcription"]["device"] is None
    assert s["transcription"]["compute_type"] == "auto"
    assert s["diarization"]["device"] == cs.SAME_DEVICE
    assert s["allow_device_fallback"] is False


def test_validate_settings_rejects_unknown_version():
    with pytest.raises(cs.ComputeSettingsError, match="schema_version"):
        cs.validate_settings({"schema_version": 99})
    with pytest.raises(cs.ComputeSettingsError):
        cs.validate_settings({"nope": 1})


def test_validate_settings_rejects_bad_compute_type():
    with pytest.raises(cs.ComputeSettingsError, match="compute_type"):
        cs.validate_settings({"schema_version": 1, "transcription": {"compute_type": "superfast"}})


def test_validate_settings_rejects_engine():
    with pytest.raises(cs.ComputeSettingsError, match="engine"):
        cs.validate_settings({"schema_version": 1, "transcription": {"engine": "whisper_cpp"}})


def test_validate_settings_accepts_valid_document():
    doc = {
        "schema_version": 1,
        "transcription": {"device": {"backend": "cuda", "id": UUID_A}, "compute_type": "int8_float32"},
        "diarization": {"device": {"backend": "cpu", "id": None}},
        "allow_device_fallback": False,
    }
    out = cs.validate_settings(doc)
    assert out["transcription"]["device"] == {"backend": "cuda", "id": UUID_A}
    assert out["transcription"]["compute_type"] == "int8_float32"
    assert out["diarization"]["device"] == {"backend": "cpu", "id": None}


# --- Device resolution (first-run policy) ------------------------------------

def test_unconfigured_single_gpu_resolves_that_gpu():
    s = cs.default_settings()
    device = cs.resolve_settings_device(s, {UUID_A})
    assert device == {"backend": "cuda", "id": UUID_A}


def test_unconfigured_multiple_gpus_requires_choice():
    s = cs.default_settings()
    with pytest.raises(cs.DeviceSelectionError, match="Multiple GPUs"):
        cs.resolve_settings_device(s, {UUID_A, UUID_B})


def test_unconfigured_no_gpu_resolves_cpu():
    s = cs.default_settings()
    device = cs.resolve_settings_device(s, set())
    assert device == cs.CPU_DEVICE


def test_configured_missing_gpu_is_error_never_fallback():
    s = cs.default_settings()
    s["transcription"]["device"] = {"backend": "cuda", "id": UUID_A}
    with pytest.raises(cs.DeviceNotAvailableError):
        cs.resolve_settings_device(s, {UUID_B})


def test_configured_cpu_stays_cpu_even_with_gpus():
    s = cs.default_settings()
    s["transcription"]["device"] = cs.CPU_DEVICE
    device = cs.resolve_settings_device(s, {UUID_A, UUID_B})
    assert device == cs.CPU_DEVICE


# --- Store persistence ---------------------------------------------------------

def test_store_roundtrip(tmp_path):
    store = cs.ComputeSettingsStore(tmp_path / "compute.json")
    saved = store.save(
        {"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_B}}}
    )
    assert saved["transcription"]["device"]["id"] == UUID_B
    loaded = store.load()
    assert loaded == saved


def test_store_missing_file_returns_defaults(tmp_path):
    store = cs.ComputeSettingsStore(tmp_path / "nope.json")
    assert store.load() == cs.default_settings()
    assert not store.path.exists()  # read never creates the file


def test_store_malformed_file_raises_without_overwriting(tmp_path):
    path = tmp_path / "compute.json"
    path.write_text("{not json", encoding="utf-8")
    store = cs.ComputeSettingsStore(path)
    with pytest.raises(cs.SettingsFileError):
        store.load()
    assert path.read_text(encoding="utf-8") == "{not json"  # untouched


def test_store_future_version_raises(tmp_path):
    path = tmp_path / "compute.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    store = cs.ComputeSettingsStore(path)
    with pytest.raises(cs.SettingsFileError, match="schema_version"):
        store.load()


def test_store_save_is_atomic_no_tmp_left_behind(tmp_path):
    store = cs.ComputeSettingsStore(tmp_path / "compute.json")
    store.save({"schema_version": 1})
    leftovers = [p for p in tmp_path.iterdir() if p.name != "compute.json"]
    assert leftovers == []


def test_store_update_merges(tmp_path):
    store = cs.ComputeSettingsStore(tmp_path / "compute.json")
    store.save({"schema_version": 1, "transcription": {"device": {"backend": "cuda", "id": UUID_A}}})
    out = store.update({"transcription": {"compute_type": "int8_float32"}})
    assert out["transcription"]["device"]["id"] == UUID_A  # merge preserved device
    assert out["transcription"]["compute_type"] == "int8_float32"


# --- Overrides + job snapshots ---------------------------------------------------

def test_parse_device_override_forms():
    assert cs.parse_device_override("") is None
    assert cs.parse_device_override(None) is None
    assert cs.parse_device_override("cpu") == cs.CPU_DEVICE
    assert cs.parse_device_override(UUID_A) == {"backend": "cuda", "id": UUID_A}
    with pytest.raises(cs.InvalidOverrideError):
        cs.parse_device_override("gpu1")
    with pytest.raises(cs.InvalidOverrideError):
        cs.parse_device_override("same")  # not allowed for transcription
    assert cs.parse_device_override("same", allow_same=True) == cs.SAME_DEVICE


def test_snapshot_defaults_same_diarization():
    settings = cs.default_settings()
    settings["transcription"]["device"] = {"backend": "cuda", "id": UUID_A}
    snap = cs.build_job_snapshot(settings, {UUID_A, UUID_B})
    assert snap["transcription"]["device_id"] == UUID_A
    assert snap["diarization"]["device_id"] == UUID_A  # same-as resolved
    assert snap["diarization"]["backend"] == "cuda"
    assert snap["transcription"]["requested_compute_type"] == "auto"
    assert snap["allow_device_fallback"] is False


def test_snapshot_overrides_beat_saved_defaults():
    settings = cs.default_settings()
    settings["transcription"]["device"] = {"backend": "cuda", "id": UUID_A}
    snap = cs.build_job_snapshot(
        settings,
        {UUID_A, UUID_B},
        transcription_override={"backend": "cpu", "id": None},
        diarization_override=cs.SAME_DEVICE,
        compute_type_override="int8_float32",
    )
    assert snap["transcription"]["backend"] == "cpu"
    assert snap["diarization"]["backend"] == "cpu"  # same -> follows override
    assert snap["transcription"]["requested_compute_type"] == "int8_float32"


def test_snapshot_rejects_missing_gpu_override():
    settings = cs.default_settings()
    settings["transcription"]["device"] = {"backend": "cuda", "id": UUID_A}
    with pytest.raises(cs.DeviceNotAvailableError):
        cs.build_job_snapshot(
            settings, {UUID_B}, transcription_override={"backend": "cuda", "id": UUID_A}
        )


def test_snapshot_rejects_unconfigured_multiple_gpus():
    with pytest.raises(cs.DeviceSelectionError):
        cs.build_job_snapshot(cs.default_settings(), {UUID_A, UUID_B})


def test_snapshot_is_deep_copied():
    settings = cs.default_settings()
    settings["transcription"]["device"] = {"backend": "cuda", "id": UUID_A}
    snap = cs.build_job_snapshot(settings, {UUID_A})
    snap["transcription"]["device_id"] = "mutated"
    assert settings["transcription"]["device"]["id"] == UUID_A
