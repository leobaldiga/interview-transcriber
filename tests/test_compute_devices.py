"""Tests for compute_devices: nvidia-smi parsing (mocked), error handling."""

import subprocess
from pathlib import Path

import pytest

import compute_devices as cd


UUID_A = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c"
UUID_B = "GPU-60ce4d58-ba50-b3fa-82da-3e51b259e857"


def _fake_run(stdout="", returncode=0, stderr="", raises=None):
    def _run(cmd, **kwargs):
        if raises:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return _run


def test_no_nvidia_smi_means_no_devices(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(raises=FileNotFoundError()))
    assert cd.list_cuda_devices() == []
    assert cd.available_cuda_ids() == set()


def test_parses_two_gpu_csv(monkeypatch):
    csv_out = (
        "0, GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c, NVIDIA GeForce RTX 3060, 00000000:01:00.0, 580.178.04, 12288, 11943, 345\n"
        "1, GPU-60ce4d58-ba50-b3fa-82da-3e51b259e857, NVIDIA GeForce GTX 1050 Ti, 00000000:02:00.0, 580.178.04, 4096, 144, 3952\n"
    )
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout=csv_out))
    devices = cd.list_cuda_devices()
    assert len(devices) == 2
    # sorted by UUID, not host index
    assert devices[0].id == UUID_A
    assert devices[0].name == "NVIDIA GeForce RTX 3060"
    assert devices[0].memory_total_mib == 12288
    assert devices[0].memory_free_mib == 345
    assert devices[1].id == UUID_B
    assert devices[1].memory_used_mib == 144


def test_skips_blank_and_mig_rows(monkeypatch):
    csv_out = (
        "\n"
        "0, GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c, NVIDIA GeForce RTX 3060, , 580.178.04, 12288, 11943, 345\n"
        "0:0, MIG-abcdef, MIG Device, , 580.178.04, 1000, 0, 1000\n"
    )
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout=csv_out))
    devices = cd.list_cuda_devices()
    assert len(devices) == 1
    assert devices[0].id == UUID_A


def test_handles_unavailable_memory_fields(monkeypatch):
    csv_out = "0, GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c, Some GPU, , , [N/A], [N/A], [N/A]\n"
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout=csv_out))
    devices = cd.list_cuda_devices()
    assert devices[0].memory_total_mib is None
    assert devices[0].memory_free_mib is None
    assert devices[0].driver_version is None


def test_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(returncode=9, stderr="driver not loaded"))
    with pytest.raises(cd.DeviceDiscoveryError, match="exited with code 9"):
        cd.list_cuda_devices()


def test_timeout_raises(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(raises=subprocess.TimeoutExpired("nvidia-smi", 5)))
    with pytest.raises(cd.DeviceDiscoveryError, match="timed out"):
        cd.list_cuda_devices()


def test_device_inventory_shape(monkeypatch):
    csv_out = f"1, {UUID_B}, GTX 1050 Ti, , 580.178.04, 4096, 144, 3952\n"
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout=csv_out))
    inv = cd.device_inventory()
    assert inv["cpu"]["backend"] == "cpu"
    assert len(inv["cuda_devices"]) == 1
    assert inv["cuda_devices"][0]["id"] == UUID_B
    assert inv["cuda_devices"][0]["memory_free_mib"] == 3952


def test_find_device(monkeypatch):
    csv_out = f"0, {UUID_A}, RTX 3060, , 580.178.04, 12288, 11943, 345\n"
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout=csv_out))
    devices = cd.list_cuda_devices()
    assert cd.find_device(devices, UUID_A).name == "RTX 3060"
    assert cd.find_device(devices, "GPU-does-not-exist") is None
