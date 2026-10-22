#!/usr/bin/env python3
"""Build the Pascal-compatible worker environments (non-destructive).

Creates .venv-whisper-pascal and .venv-pyannote-pascal alongside the existing
.venv / .venv-pyannote — the legacy environments are NEVER touched or removed.
After validation, point the app at the new interpreters via:

    export WHISPER_WORKER_PYTHON=.venv-whisper-pascal/bin/python
    export PYANNOTE_WORKER_PYTHON=.venv-pyannote-pascal/bin/python

(Relative paths resolve against the repo root; see worker_runtime.py.)

Pins and rationale: see requirements/whisper-pascal.txt and
requirements/pyannote-pascal.txt, and docs/pascal-worker-profile.md.
Dependency resolution was verified against the PyTorch cu126 index + PyPI
(torch 2.8.0+cu126 pins nvidia-cudnn-cu12==9.10.2.21 natively — the last cuDNN
line with Pascal support).

Usage:
    python3 scripts/setup_worker_envs.py [--python /usr/bin/python3.12]
                                         [--force] [--whisper-only] [--pyannote-only]
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_DIR = BASE_DIR / "requirements"
TORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"

IS_WINDOWS = platform.system().lower().startswith("win")

ENVS = {
    "whisper": {
        "dir": BASE_DIR / ".venv-whisper-pascal",
        "manifest": REQUIREMENTS_DIR / "whisper-pascal.txt",
        "torch_stack": ["torch==2.8.0+cu126", "torchaudio==2.8.0+cu126", "nvidia-cudnn-cu12==9.10.2.21"],
    },
    "pyannote": {
        "dir": BASE_DIR / ".venv-pyannote-pascal",
        "manifest": REQUIREMENTS_DIR / "pyannote-pascal.txt",
        "torch_stack": ["torch==2.8.0+cu126", "torchvision==0.23.0+cu126", "torchaudio==2.8.0+cu126",
                        "nvidia-cudnn-cu12==9.10.2.21"],
    },
}


def echo(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1) -> None:
    echo(f"[ERROR] {msg}")
    raise SystemExit(code)


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def check_python_version(py: str) -> None:
    out = subprocess.check_output(
        [py, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"], text=True
    ).strip()
    major, minor = (int(x) for x in out.split())
    version = f"{major}.{minor}"
    if major != 3 or minor < 9 or minor > 13:
        fail(
            f"Python {version} is not usable for the Pascal profile: torch 2.8.0+cu126 "
            "wheels exist only for Python 3.9-3.13. Point --python at a 3.9-3.13 "
            "interpreter (the existing .venv-pyannote interpreter is known-good)."
        )
    echo(f"[OK] Python {version}")


def create_venv(py: str, target: Path, force: bool) -> Path:
    if target.exists():
        if not force:
            fail(
                f"{target} already exists — refusing to rebuild it (use --force to recreate, "
                "which DELETES that environment only; the legacy .venv/.venv-pyannote are never touched)"
            )
        shutil.rmtree(target)
    echo(f"Creating {target.name} ...")
    subprocess.run([py, "-m", "venv", str(target)], check=True)
    exe = venv_python(target)
    if not exe.exists():
        fail(f"venv python not found at {exe}")
    subprocess.run([str(exe), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    return exe


def pip_install(exe: Path, packages: list[str], index_url: str | None = None) -> None:
    cmd = [str(exe), "-m", "pip", "install"]
    if index_url:
        cmd += ["--index-url", index_url, "--extra-index-url", "https://pypi.org/simple"]
    cmd += packages
    echo(f"  installing: {' '.join(packages)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default="", help="interpreter to build the venvs from (default: python3)")
    parser.add_argument("--force", action="store_true", help="recreate existing target environments")
    parser.add_argument("--whisper-only", action="store_true")
    parser.add_argument("--pyannote-only", action="store_true")
    args = parser.parse_args()

    py = args.python or (shutil.which("python3") or shutil.which("python"))
    if not py:
        fail("No Python interpreter found; pass --python /path/to/python3.12")
    check_python_version(py)

    selected = list(ENVS)
    if args.whisper_only:
        selected = ["whisper"]
    if args.pyannote_only:
        selected = ["pyannote"]

    for name in selected:
        spec = ENVS[name]
        if not spec["manifest"].exists():
            fail(f"missing manifest {spec['manifest']}")
        exe = create_venv(py, spec["dir"], args.force)
        pip_install(exe, spec["torch_stack"], index_url=TORCH_CU126_INDEX)
        pip_install(exe, ["-r", str(spec["manifest"])])
        check = subprocess.run([str(exe), "-m", "pip", "check"], capture_output=True, text=True)
        if check.returncode != 0:
            echo(f"[WARN] pip check reported issues in {spec['dir'].name}:")
            echo((check.stdout or check.stderr).strip()[:1000])
        else:
            echo(f"[OK] pip check clean for {spec['dir'].name}")
        freeze_path = REQUIREMENTS_DIR / f"{name}-pascal-freeze.txt"
        freeze = subprocess.run([str(exe), "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        freeze_path.write_text(freeze.stdout, encoding="utf-8")
        echo(f"[OK] freeze written to {freeze_path.relative_to(BASE_DIR)}")

    echo()
    echo("Environments built. Validate on the target machine before switching the app:")
    echo("  python3 scripts/run_probe.py --device GPU-<uuid> --smoke")
    echo()
    echo("Then point the app at the new interpreters (e.g. in your .env / shell):")
    echo("  export WHISPER_WORKER_PYTHON=.venv-whisper-pascal/bin/python")
    echo("  export PYANNOTE_WORKER_PYTHON=.venv-pyannote-pascal/bin/python")
    echo()
    echo("Legacy .venv / .venv-pyannote were left untouched — rollback is just")
    echo("unsetting those two variables and restarting the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
