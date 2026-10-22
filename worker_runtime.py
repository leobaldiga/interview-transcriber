"""Shared worker-runtime layer for the Interview Transcriber.

One code path builds the environment and picks the interpreter for BOTH
production worker launches and compatibility probes. It never imports any
machine-learning packages into the calling process.

Responsibilities:
  * Resolve worker Python executables (legacy defaults, env-var overrides).
  * Build each child's environment: CUDA visibility pinned to the selected
    GPU UUID (or fully hidden for CPU), and a controlled LD_LIBRARY_PATH
    derived from the chosen environment's own NVIDIA packages.
  * Keep the parent process environment untouched (a fresh dict per child).

Library-path policy (Linux): the inherited LD_LIBRARY_PATH is discarded and,
if the worker environment provides NVIDIA/CUDA shared libraries, replaced
with those directories before the worker starts. This prevents a newer
system cuDNN (e.g. Pascal-incompatible 9.11+) from being loaded instead of
the environment's pinned copy. Workers must NOT pop LD_LIBRARY_PATH
themselves — the parent's decision is authoritative.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

IS_WINDOWS = os.name == "nt"

BASE_DIR = Path(__file__).resolve().parent

ROLE_WHISPER = "whisper"
ROLE_PYANNOTE = "pyannote"

_ROLE_ENV_VAR = {
    ROLE_WHISPER: "WHISPER_WORKER_PYTHON",
    ROLE_PYANNOTE: "PYANNOTE_WORKER_PYTHON",
}
_ROLE_LEGACY_VENV = {
    ROLE_WHISPER: ".venv",
    ROLE_PYANNOTE: ".venv-pyannote",
}


class WorkerRuntimeError(Exception):
    def __init__(self, message: str, code: str = "WORKER_RUNTIME_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if IS_WINDOWS else venv_dir / "bin" / "python"


def resolve_worker_python(role: str) -> Path:
    """Resolve the Python executable for a worker role.

    Resolution order: dedicated env var (e.g. WHISPER_WORKER_PYTHON), then the
    legacy default venv. Relative paths are resolved against the repo root.
    Raises WorkerRuntimeError with a clear message when missing — never falls
    back to a different role's interpreter.
    """
    if role not in _ROLE_ENV_VAR:
        raise WorkerRuntimeError(f"unknown worker role {role!r}")
    override = os.environ.get(_ROLE_ENV_VAR[role], "").strip()
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if not candidate.exists():
            raise WorkerRuntimeError(
                f"{_ROLE_ENV_VAR[role]} points to {candidate}, which does not exist", code="WORKER_INTERPRETER_MISSING"
            )
        if not os.access(candidate, os.X_OK):
            raise WorkerRuntimeError(f"{candidate} is not executable", code="WORKER_INTERPRETER_MISSING")
        return candidate
    default = _venv_python(BASE_DIR / _ROLE_LEGACY_VENV[role])
    if not default.exists():
        raise WorkerRuntimeError(
            f"Worker interpreter for role {role!r} not found at {default}. "
            f"Run the installer or set {_ROLE_ENV_VAR[role]}.",
            code="WORKER_INTERPRETER_MISSING",
        )
    return default


def _site_packages(py_exe: Path) -> Optional[Path]:
    """Ask the worker's interpreter where its purelib directory is."""
    try:
        result = subprocess.run(
            [str(py_exe), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return Path(out) if out else None


def discover_worker_lib_dirs(py_exe: Path) -> list[str]:
    """Library directories from the worker environment's own NVIDIA packages.

    Covers the pip-installed nvidia-* wheels (site-packages/nvidia/*/lib) and
    torch's bundled libs (site-packages/torch/lib). Returns [] when the
    environment has none (e.g. a CPU-only build).
    """
    if IS_WINDOWS:
        return []
    site = _site_packages(py_exe)
    if site is None or not site.is_dir():
        return []
    dirs: list[str] = []
    nvidia_root = site / "nvidia"
    if nvidia_root.is_dir():
        for entry in sorted(nvidia_root.iterdir()):
            lib = entry / "lib"
            if lib.is_dir():
                dirs.append(str(lib))
    torch_lib = site / "torch" / "lib"
    if torch_lib.is_dir():
        dirs.append(str(torch_lib))
    return dirs


def build_worker_env(py_exe: Path, device: dict) -> dict:
    """Build a fresh environment dict for one worker launch.

    device: {"backend": "cuda", "id": "GPU-..."} or {"backend": "cpu"}.
    The parent's os.environ is copied, never mutated.
    """
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "HOME": str(Path.home()),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    path_parts = [str(py_exe.parent)]
    if IS_WINDOWS:
        path_parts.extend([str(Path(sys.executable).parent), env.get("PATH", "")])
    else:
        path_parts.extend(["/usr/local/bin", "/usr/bin", "/bin", env.get("PATH", "")])
    env["PATH"] = os.pathsep.join([p for p in path_parts if p])

    backend = (device or {}).get("backend", "cpu")
    if backend == "cuda":
        # Full-UUID visibility: the selected GPU becomes the child's only
        # visible device (logical cuda:0 inside the worker).
        env["CUDA_VISIBLE_DEVICES"] = device["id"]
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        # CPU selection hides every GPU so nothing can opportunistically allocate.
        env["CUDA_VISIBLE_DEVICES"] = ""

    if not IS_WINDOWS:
        lib_dirs = discover_worker_lib_dirs(py_exe)
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env
