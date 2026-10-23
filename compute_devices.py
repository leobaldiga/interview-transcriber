"""GPU/CPU device discovery for the Interview Transcriber.

Discovery is allocation-free: it shells out to `nvidia-smi` (when present)
and parses its CSV output. It never imports torch/ctranslate2, never creates
a CUDA context, and never allocates GPU memory — safe to call from the
FastAPI parent process for status panels and settings forms.

Devices are identified by their full NVIDIA UUID (stable across reboots and
bus re-enumeration). Host indices are display-only and may change.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

_NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
]


@dataclass
class DeviceInfo:
    backend: str  # "cuda"
    id: str  # full GPU UUID
    name: str
    host_index: Optional[int] = None  # display only — NOT a stable identifier
    pci_bus_id: Optional[str] = None
    driver_version: Optional[str] = None
    memory_total_mib: Optional[int] = None
    memory_used_mib: Optional[int] = None
    memory_free_mib: Optional[int] = None

    def to_api(self) -> dict:
        return {
            "backend": self.backend,
            "id": self.id,
            "name": self.name,
            "host_index": self.host_index,
            "pci_bus_id": self.pci_bus_id,
            "driver_version": self.driver_version,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
        }


class DeviceDiscoveryError(Exception):
    """nvidia-smi exists but failed in a way worth surfacing."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail[:500]


def _parse_memory(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value or not value.isdigit():
        return None
    return int(value)


def _parse_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def list_cuda_devices(timeout: float = 5.0) -> list[DeviceInfo]:
    """Enumerate NVIDIA GPUs via nvidia-smi. Empty list when no NVIDIA stack.

    Raises DeviceDiscoveryError when nvidia-smi exists but fails/times out —
    that condition is worth reporting rather than hiding as "no GPUs".
    """
    nvidia_smi = os.environ.get("NVIDIA_SMI_PATH", "nvidia-smi")
    try:
        result = subprocess.run(
            [nvidia_smi] + _NVIDIA_SMI_QUERY[1:],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return []  # no NVIDIA userspace on this machine — fine
    except subprocess.TimeoutExpired:
        raise DeviceDiscoveryError("nvidia-smi timed out", f"query exceeded {timeout}s")
    except OSError as e:
        raise DeviceDiscoveryError(f"could not run nvidia-smi: {e}")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise DeviceDiscoveryError(
            f"nvidia-smi exited with code {result.returncode}", stderr or stdout
        )

    devices: list[DeviceInfo] = []
    reader = csv.reader(io.StringIO(result.stdout))
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        # Pad short rows so a missing optional field doesn't crash parsing.
        cells = [cell.strip() for cell in row]
        while len(cells) < 8:
            cells.append("")
        uuid = cells[1]
        if not uuid.startswith("GPU-"):
            continue  # e.g. MIG instance rows — not selectable as a whole GPU here
        devices.append(
            DeviceInfo(
                backend="cuda",
                id=uuid,
                name=cells[2] or "Unknown NVIDIA GPU",
                host_index=_parse_int(cells[0]),
                pci_bus_id=cells[3] or None,
                driver_version=cells[4] or None,
                memory_total_mib=_parse_memory(cells[5]),
                memory_used_mib=_parse_memory(cells[6]),
                memory_free_mib=_parse_memory(cells[7]),
            )
        )
    # Stable sort by UUID — never trust host index ordering.
    devices.sort(key=lambda d: d.id)
    return devices


def device_inventory(timeout: float = 5.0) -> dict:
    """Full inventory for API responses: CPU + every CUDA device."""
    cuda_devices = list_cuda_devices(timeout=timeout)
    return {
        "cpu": {
            "backend": "cpu",
            "id": "cpu",
            "name": "CPU",
        },
        "cuda_devices": [d.to_api() for d in cuda_devices],
    }


def available_cuda_ids(timeout: float = 5.0) -> set[str]:
    """Convenience: just the set of selectable GPU UUIDs."""
    return {d.id for d in list_cuda_devices(timeout=timeout)}


def find_device(devices: list[DeviceInfo], device_id: str) -> Optional[DeviceInfo]:
    for d in devices:
        if d.id == device_id:
            return d
    return None
