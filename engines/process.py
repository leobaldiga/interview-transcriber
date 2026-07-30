"""Subprocess helpers shared by the whisper and pyannote workers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"


def venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if IS_WINDOWS else venv_dir / "bin" / "python"


def build_worker_env(py_exe: Path) -> dict:
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

    if not IS_WINDOWS:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def popen_kwargs() -> dict:
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def spawn_worker(cmd: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env, **popen_kwargs())
