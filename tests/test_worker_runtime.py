"""Tests for worker_runtime: interpreter resolution + environment construction."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import worker_runtime as wr


UUID_A = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c"


@pytest.fixture
def fake_venvs(tmp_path, monkeypatch):
    """Create two fake venv python executables and point BASE_DIR at tmp_path."""
    for role, dirname in (("whisper", ".venv"), ("pyannote", ".venv-pyannote")):
        py = tmp_path / dirname / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("#!/bin/sh\nexit 0\n")
        py.chmod(0o755)
    monkeypatch.setattr(wr, "BASE_DIR", tmp_path)
    return tmp_path


def test_legacy_defaults_resolve(fake_venvs):
    assert wr.resolve_worker_python(wr.ROLE_WHISPER) == wr._venv_python(fake_venvs / ".venv")
    assert wr.resolve_worker_python(wr.ROLE_PYANNOTE) == wr._venv_python(fake_venvs / ".venv-pyannote")


def test_env_var_override_absolute(fake_venvs, tmp_path, monkeypatch):
    custom = tmp_path / "custom-venv/bin/python"
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_text("#!/bin/sh\nexit 0\n")
    custom.chmod(0o755)
    monkeypatch.setenv("WHISPER_WORKER_PYTHON", str(custom))
    assert wr.resolve_worker_python(wr.ROLE_WHISPER) == custom


def test_env_var_override_relative_resolved_against_base(fake_venvs, monkeypatch):
    monkeypatch.setenv("PYANNOTE_WORKER_PYTHON", ".venv-whisper-pascal/bin/python")
    py = fake_venvs / ".venv-whisper-pascal/bin/python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    assert wr.resolve_worker_python(wr.ROLE_PYANNOTE) == py


def test_missing_interpreter_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "BASE_DIR", tmp_path)
    with pytest.raises(wr.WorkerRuntimeError, match="not found"):
        wr.resolve_worker_python(wr.ROLE_WHISPER)


def test_unknown_role_rejected():
    with pytest.raises(wr.WorkerRuntimeError, match="unknown worker role"):
        wr.resolve_worker_python("tensorrt")


def _fake_site_packages(monkeypatch, tmp_path, dirs):
    """Make _site_packages return tmp_path/site with the given subdirectories."""
    site = tmp_path / "site"
    site.mkdir()
    for d in dirs:
        (site / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(wr, "_site_packages", lambda py: site)
    return site


def test_build_env_cuda_sets_uuid_visibility(fake_venvs, monkeypatch):
    _fake_site_packages(monkeypatch, fake_venvs, [])
    env = wr.build_worker_env(fake_venvs / ".venv/bin/python", {"backend": "cuda", "id": UUID_A})
    assert env["CUDA_VISIBLE_DEVICES"] == UUID_A  # full UUID, never host index
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PYTHONPATH"] == ""
    assert env["TOKENIZERS_PARALLELISM"] == "false"


def test_build_env_cpu_hides_all_gpus(fake_venvs, monkeypatch):
    _fake_site_packages(monkeypatch, fake_venvs, [])
    py = fake_venvs / ".venv/bin/python"
    env = wr.build_worker_env(py, {"backend": "cpu", "id": None})
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_build_env_does_not_mutate_parent(fake_venvs, monkeypatch):
    _fake_site_packages(monkeypatch, fake_venvs, [])
    py = fake_venvs / ".venv/bin/python"
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    wr.build_worker_env(py, {"backend": "cuda", "id": UUID_A})
    wr.build_worker_env(py, {"backend": "cpu", "id": None})
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


def test_build_env_constructs_library_path_from_env_packages(fake_venvs, monkeypatch):
    site = _fake_site_packages(
        monkeypatch,
        fake_venvs,
        ["nvidia/cudnn/lib", "nvidia/cublas/lib", "torch/lib"],
    )
    env = wr.build_worker_env(fake_venvs / ".venv/bin/python", {"backend": "cuda", "id": UUID_A})
    ld = env["LD_LIBRARY_PATH"]
    assert ld is not None
    parts = ld.split(os.pathsep)
    assert str(site / "nvidia/cudnn/lib") in parts
    assert str(site / "torch/lib") in parts
    # no arbitrary inherited path survives
    monkeypatch.setenv("LD_LIBRARY_PATH", "/some/incompatible/system/cuda/lib64")
    env2 = wr.build_worker_env(fake_venvs / ".venv/bin/python", {"backend": "cuda", "id": UUID_A})
    assert "/some/incompatible/system/cuda/lib64" not in (env2["LD_LIBRARY_PATH"] or "")


def test_build_env_no_nvidia_packages_pops_inherited(fake_venvs, monkeypatch):
    _fake_site_packages(monkeypatch, fake_venvs, [])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/some/random/path")
    env = wr.build_worker_env(fake_venvs / ".venv/bin/python", {"backend": "cpu", "id": None})
    assert "LD_LIBRARY_PATH" not in env


def test_discover_worker_lib_dirs(fake_venvs, monkeypatch):
    site = _fake_site_packages(
        monkeypatch,
        fake_venvs,
        ["nvidia/cudnn/lib", "nvidia/cublas/lib", "nvidia/cudnn/no-lib-here", "torch/lib"],
    )
    py = fake_venvs / ".venv/bin/python"
    dirs = wr.discover_worker_lib_dirs(py)
    assert str(site / "nvidia/cudnn/lib") in dirs
    assert str(site / "nvidia/cublas/lib") in dirs
    assert str(site / "torch/lib") in dirs
    assert not any("no-lib-here" in d for d in dirs)
