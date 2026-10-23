"""Persistent compute configuration for the Interview Transcriber.

This module owns the saved processing-device settings (which GPU runs
transcription and diarization), their validation, and their atomic
persistence to disk. It deliberately imports NO machine-learning packages:
the FastAPI parent process stays GPU-allocation-free, and these settings
contain no executable paths — they only reference devices by stable IDs
(CPU, or full NVIDIA GPU UUIDs).

Configuration shape (schema_version 1):

    {
      "schema_version": 1,
      "transcription": {
        "engine": "faster_whisper",
        "device": {"backend": "cuda", "id": "GPU-..."} | {"backend": "cpu", "id": null},
        "compute_type": "auto"
      },
      "diarization": {
        "device": {"backend": "same_as_transcription", "id": null} | <device ref>
      },
      "allow_device_fallback": false
    }

Design rules (see docs/gpu-selection.md):
  * Selections are by UUID, never by host index (indices reorder across boots).
  * A missing/invalid saved GPU is a configuration ERROR — never silently
    fall back to another GPU or to CPU.
  * First run with multiple GPUs requires an explicit choice before compute
    jobs are accepted; a single-GPU machine may resolve that one GPU; a
    no-GPU machine resolves CPU.
  * "same_as_transcription" is resolved at job acceptance, never later.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

VALID_BACKENDS = {"cpu", "cuda"}
SAME_AS_TRANSCRIPTION = "same_as_transcription"

# Compute types accepted in settings/forms. "auto" is resolved by the worker
# against the runtime's supported list (see whisper_worker.py); the others
# are passed through to CTranslate2 unchanged and must be supported there.
VALID_COMPUTE_TYPES = {
    "auto",
    "float32",
    "float16",
    "bfloat16",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
}

# Full NVIDIA GPU UUID, e.g. GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c
_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class ComputeSettingsError(Exception):
    """Base error for compute configuration problems."""

    code = "COMPUTE_SETTINGS_INVALID"

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class SettingsFileError(ComputeSettingsError):
    code = "SETTINGS_FILE_ERROR"


class DeviceSelectionError(ComputeSettingsError):
    code = "DEVICE_NOT_CONFIGURED"


class DeviceNotAvailableError(ComputeSettingsError):
    code = "DEVICE_NOT_AVAILABLE"


class InvalidOverrideError(ComputeSettingsError):
    code = "INVALID_DEVICE_OVERRIDE"


# --- Device references -------------------------------------------------------

CPU_DEVICE: dict = {"backend": "cpu", "id": None}
SAME_DEVICE: dict = {"backend": SAME_AS_TRANSCRIPTION, "id": None}


def is_valid_gpu_uuid(value: Any) -> bool:
    return isinstance(value, str) and bool(_UUID_RE.match(value))


def validate_device_ref(ref: Any, *, allow_same: bool = False, field: str = "device") -> dict:
    """Validate and normalize a device reference. Returns a deep copy."""
    if not isinstance(ref, dict):
        raise ComputeSettingsError(f"{field} must be an object, got {type(ref).__name__}")
    backend = ref.get("backend")
    if backend == SAME_AS_TRANSCRIPTION:
        if not allow_same:
            raise ComputeSettingsError(f"{field}: 'same_as_transcription' is only valid for diarization")
        return {"backend": SAME_AS_TRANSCRIPTION, "id": None}
    if backend not in VALID_BACKENDS:
        raise ComputeSettingsError(
            f"{field}: unsupported backend {backend!r} (expected 'cpu', 'cuda'"
            + (", or 'same_as_transcription'" if allow_same else "")
            + ")"
        )
    if backend == "cpu":
        return {"backend": "cpu", "id": None}
    device_id = ref.get("id")
    if not is_valid_gpu_uuid(device_id):
        raise ComputeSettingsError(f"{field}: CUDA selection requires a full GPU UUID, got {device_id!r}")
    return {"backend": "cuda", "id": device_id}


def resolve_same(ref: dict, transcription_ref: dict) -> dict:
    """Resolve a diarization ref against the job's transcription ref."""
    if ref.get("backend") == SAME_AS_TRANSCRIPTION:
        return copy.deepcopy(transcription_ref)
    return copy.deepcopy(ref)


# --- Settings document --------------------------------------------------------

def default_settings() -> dict:
    """Settings used when nothing has been configured yet.

    device=None means "unconfigured": resolution against available hardware
    happens at job acceptance (see resolve_settings_device) and fails with
    NeedsDeviceSelection when multiple GPUs are present.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "transcription": {
            "engine": "faster_whisper",
            "device": None,
            "compute_type": "auto",
        },
        "diarization": {
            "device": copy.deepcopy(SAME_DEVICE),
        },
        "allow_device_fallback": False,
    }


def validate_settings(raw: Any) -> dict:
    """Validate a settings document; returns a normalized deep copy or raises."""
    if not isinstance(raw, dict):
        raise ComputeSettingsError("settings must be a JSON object")
    out = default_settings()
    if "schema_version" not in raw:
        raise ComputeSettingsError("settings missing schema_version")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ComputeSettingsError(
            f"unsupported settings schema_version {raw['schema_version']!r} (expected {SCHEMA_VERSION})"
        )

    transcription = raw.get("transcription", {})
    if not isinstance(transcription, dict):
        raise ComputeSettingsError("transcription must be an object")
    engine = transcription.get("engine", "faster_whisper")
    if engine != "faster_whisper":
        raise ComputeSettingsError(f"unsupported transcription engine {engine!r}")
    device = transcription.get("device")
    if device is not None:
        out["transcription"]["device"] = validate_device_ref(device, field="transcription.device")
    compute_type = transcription.get("compute_type", "auto")
    if compute_type not in VALID_COMPUTE_TYPES:
        raise ComputeSettingsError(
            f"unsupported compute_type {compute_type!r} (expected one of {sorted(VALID_COMPUTE_TYPES)})"
        )
    out["transcription"]["compute_type"] = compute_type

    diarization = raw.get("diarization", {})
    if not isinstance(diarization, dict):
        raise ComputeSettingsError("diarization must be an object")
    dref = diarization.get("device")
    if dref is not None:
        out["diarization"]["device"] = validate_device_ref(dref, allow_same=True, field="diarization.device")

    out["allow_device_fallback"] = bool(raw.get("allow_device_fallback", False))
    return out


def resolve_settings_device(
    settings: dict,
    available_cuda_ids: set[str],
    *,
    has_cpu: bool = True,
) -> dict:
    """Resolve the transcription device from saved settings + available hardware.

    Raises DeviceSelectionError / DeviceNotAvailableError; never guesses a GPU.
    """
    device = settings.get("transcription", {}).get("device")
    if device is None:
        # Unconfigured first run.
        if not available_cuda_ids:
            if has_cpu:
                return copy.deepcopy(CPU_DEVICE)
            raise DeviceSelectionError("No CUDA devices available and CPU selection is not possible")
        if len(available_cuda_ids) == 1:
            (only,) = available_cuda_ids
            return {"backend": "cuda", "id": only}
        raise DeviceSelectionError(
            "Multiple GPUs are available — select a processing device in Compute settings before "
            "submitting transcription jobs"
        )
    if device["backend"] == "cpu":
        return copy.deepcopy(CPU_DEVICE)
    if device["id"] not in available_cuda_ids:
        raise DeviceNotAvailableError(
            f"Selected GPU {device['id']} is not available on this machine. "
            "Update Compute settings — no fallback device will be chosen automatically."
        )
    return copy.deepcopy(device)


# --- Persistence (thread-safe, atomic) ----------------------------------------

def settings_path() -> Path:
    override = os.environ.get("COMPUTE_SETTINGS_PATH", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "config" / "compute.json"


class ComputeSettingsStore:
    """Thread-safe load/validate/save of the compute settings document.

    Saving is atomic (temp file + os.replace) and serialized with a lock.
    A malformed or future-version file raises on load but is NEVER overwritten
    by a read; only an explicit successful save rewrites it.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else settings_path()
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict:
        """Return validated settings, or defaults when no file exists.

        Raises SettingsFileError when the file exists but cannot be parsed or
        validated (the file is left untouched in that case).
        """
        with self._lock:
            if not self._path.exists():
                return default_settings()
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise SettingsFileError(f"Could not read compute settings {self._path}: {e}") from e
            try:
                return validate_settings(raw)
            except ComputeSettingsError as e:
                raise SettingsFileError(f"Invalid compute settings in {self._path}: {e.message}") from e

    def save(self, settings: dict) -> dict:
        validated = validate_settings(settings)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".compute.json.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(validated, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, self._path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        return validated

    def update(self, changes: dict) -> dict:
        """Merge changes into current settings and save atomically."""
        with self._lock:
            current = self.load()
            merged = copy.deepcopy(current)
            _deep_merge(merged, changes)
            return self.save(merged)


def _deep_merge(target: dict, changes: dict) -> dict:
    for key, value in changes.items():
        if key == "schema_version":
            continue
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


# --- Job snapshots ---------------------------------------------------------------

def parse_device_override(value: Any, *, allow_same: bool = False, field: str = "device") -> Optional[dict]:
    """Parse a device override string from a form/API field.

    '' / None  -> no override (use saved default)
    'cpu'      -> CPU
    'same'     -> same as transcription (diarization only)
    'GPU-...'  -> that CUDA device (full UUID)
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidOverrideError(f"{field}: expected a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        return None
    if text == "cpu":
        return copy.deepcopy(CPU_DEVICE)
    if text == "same":
        if not allow_same:
            raise InvalidOverrideError(f"{field}: 'same' is only valid for diarization")
        return copy.deepcopy(SAME_DEVICE)
    if is_valid_gpu_uuid(text):
        return {"backend": "cuda", "id": text}
    raise InvalidOverrideError(
        f"{field}: invalid device selection {text!r} — use 'cpu', "
        + ("'same', " if allow_same else "")
        + "or a full GPU UUID (GPU-xxxxxxxx-...)"
    )


def build_job_snapshot(
    settings: dict,
    available_cuda_ids: set[str],
    *,
    transcription_override: Optional[dict] = None,
    diarization_override: Optional[dict] = None,
    compute_type_override: Optional[str] = None,
) -> dict:
    """Build the immutable compute snapshot stored on a job at acceptance time.

    The snapshot is fully resolved: 'same_as_transcription' is replaced by
    the job's own transcription device, and compute_type is the concrete
    requested value ('auto' allowed — the worker resolves it against the
    runtime's supported list and records what it actually used).
    """
    transcription_device = resolve_settings_device(settings, available_cuda_ids)
    if transcription_override is not None:
        transcription_device = copy.deepcopy(transcription_override)

    diarization_device = settings.get("diarization", {}).get("device") or copy.deepcopy(SAME_DEVICE)
    if diarization_override is not None:
        diarization_device = copy.deepcopy(diarization_override)
    diarization_device = resolve_same(diarization_device, transcription_device)

    # Validate CUDA selections against hardware at acceptance (fail fast).
    for ref, field in ((transcription_device, "transcription"), (diarization_device, "diarization")):
        if ref["backend"] == "cuda" and ref["id"] not in available_cuda_ids:
            raise DeviceNotAvailableError(
                f"{field}: selected GPU {ref['id']} is not available on this machine"
            )

    compute_type = compute_type_override or settings.get("transcription", {}).get("compute_type", "auto")
    if compute_type not in VALID_COMPUTE_TYPES:
        raise ComputeSettingsError(f"unsupported compute_type {compute_type!r}")

    return {
        "schema_version": SCHEMA_VERSION,
        "engine": "faster_whisper",
        "transcription": {
            "backend": transcription_device["backend"],
            "device_id": transcription_device["id"],
            "requested_compute_type": compute_type,
        },
        "diarization": {
            "backend": diarization_device["backend"],
            "device_id": diarization_device["id"],
        },
        "allow_device_fallback": False,
    }
