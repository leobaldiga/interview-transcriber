#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
IS_WINDOWS = platform.system().lower().startswith("win")


def echo(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1) -> None:
    echo(f"[ERROR] {msg}")
    raise SystemExit(code)


def run(cmd: list[str], quiet: bool = False, env: dict[str, str] | None = None) -> None:
    if quiet:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout)
            fail(f"Command failed: {' '.join(cmd)}", result.returncode)
    else:
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env)
        if result.returncode != 0:
            fail(f"Command failed: {' '.join(cmd)}", result.returncode)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def find_python() -> str:
    candidates = []
    if os.environ.get("PYTHON"):
        candidates.append(os.environ["PYTHON"])
    if IS_WINDOWS:
        candidates.extend(["python", "py"])
    else:
        candidates.extend(["python3", "python"])

    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return candidate
    fail("Python not found on PATH.")
    return ""


def get_python_version(py: str) -> tuple[int, int]:
    code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    out = subprocess.check_output([py, "-c", code], text=True).strip()
    major, minor = out.split(".")
    return int(major), int(minor)


def venv_python(venv_dir: Path) -> Path:
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def remove_dir(path: Path) -> None:
    if path.exists():
      shutil.rmtree(path)


def pip_install(py_exe: Path, packages: list[str], index_url: str | None = None, quiet: bool = False) -> None:
    cmd = [str(py_exe), "-m", "pip", "install"]
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(packages)
    if quiet:
        cmd.append("--quiet")
    run(cmd, quiet=False)


def gpu_torch_index_url() -> str | None:
    if command_exists("nvidia-smi"):
        return "https://download.pytorch.org/whl/cu128"
    return None


def prompt_hf_token(token_path: Path) -> None:
    echo("")
    echo("HuggingFace token (required for pyannote diarization pipeline).")
    echo("Get yours at: https://huggingface.co/settings/tokens")
    echo("You must also accept the model terms at:")
    echo("  https://huggingface.co/pyannote/speaker-diarization-3.1")
    echo("")

    if token_path.exists() and token_path.stat().st_size > 0:
        echo("[SKIP] .hftoken already exists — remove it to re-enter.")
        return

    try:
        token = input("Paste your HuggingFace token (hf_...): ").strip()
    except EOFError:
        token = ""

    if token:
        token_path.write_text(token + "\n", encoding="utf-8")
        try:
            os.chmod(token_path, 0o600)
        except OSError:
            pass
        echo("[OK] Token saved to .hftoken")
    else:
        echo("[WARN] No token entered — diarization will fail at runtime.")
        token_path.touch(exist_ok=True)


def write_env(env_path: Path) -> None:
    if env_path.exists():
        echo("[SKIP] .env already exists")
        return

    env_path.write_text(
        "HOST=0.0.0.0\n"
        "PORT=8765\n",
        encoding="utf-8",
    )
    echo("[OK] .env created")


def create_venv(py: str, path: Path) -> Path:
    remove_dir(path)
    run([py, "-m", "venv", str(path)])
    py_exe = venv_python(path)
    if not py_exe.exists():
        fail(f"Virtualenv python not found at {py_exe}")
    run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    return py_exe


def main() -> None:
    echo("")
    echo("==========================================")
    echo("  Interview Transcriber v5 — Setup")
    echo("==========================================")
    echo("")

    py = find_python()
    major, minor = get_python_version(py)
    if major < 3 or (major == 3 and minor < 10):
        fail(f"Python 3.10+ required (found {major}.{minor})")
    echo(f"[OK] Python {major}.{minor}")

    torch_index = gpu_torch_index_url()
    if torch_index:
        echo("[OK] NVIDIA GPU detected — using CUDA PyTorch wheels")
    else:
        echo("[WARN] nvidia-smi not found — using default CPU PyTorch wheels")

    echo("")
    echo("Creating .venv (FastAPI + faster-whisper + VAD deps)...")
    app_venv = create_venv(py, BASE_DIR / ".venv")
    app_torch = ["torch", "torchaudio"]
    pip_install(app_venv, app_torch, index_url=torch_index)
    pip_install(
        app_venv,
        [
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
            "jinja2",
            "faster-whisper",
            "soundfile",
            "ffmpeg-python",
        ],
    )
    echo("[OK] .venv created")

    echo("")
    echo("Creating .venv-pyannote (PyTorch + pyannote.audio)...")
    echo("  This downloads large PyTorch wheels — may take a few minutes.")
    pyannote_venv = create_venv(py, BASE_DIR / ".venv-pyannote")
    pip_install(
        pyannote_venv,
        ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0"],
        index_url=torch_index,
    )
    pip_install(
        pyannote_venv,
        [
            "pyannote.audio==3.3.2",
            "huggingface_hub==0.25.2",
            "matplotlib",
        ],
    )
    echo("[OK] .venv-pyannote created")

    prompt_hf_token(BASE_DIR / ".hftoken")

    (BASE_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    echo("[OK] outputs/ and uploads/ directories created")

    write_env(BASE_DIR / ".env")

    echo("")
    echo("==========================================")
    echo("  Setup complete!")
    echo("==========================================")
    echo("")
    echo("Start the server:")
    echo("")

    if IS_WINDOWS:
        echo(r"  .\.venv\Scripts\python.exe app.py")
    else:
        echo("  ./.venv/bin/python app.py")

    echo("")
    echo("If needed, export settings from .env before launch.")
    echo("Then open http://localhost:8765")
    echo("")


if __name__ == "__main__":
    main()
