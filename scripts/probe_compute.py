#!/usr/bin/env python3
"""Compute compatibility probe for the Interview Transcriber.

Runs INSIDE a worker environment (launched by app.py's test API or manually
via scripts/probe_cli.py) against one isolated device. Implements the real
hardware checks — this script imports torch / faster-whisper / pyannote and
therefore only runs on the target machine, never in the web-app process.

Usage (internal — same args/result-file convention as the workers):
    python probe_compute.py <args_file> <result_file>

args_file:  { "role": "whisper"|"pyannote", "device": {"backend","id"},
              "mode": "inventory"|"torch"|"whisper_smoke"|"pyannote_smoke",
              "model": optional model for whisper_smoke,
              "audio_path": optional fixture for smoke tests }
result_file: { "status": "ok"|"error", "payload": {...}, "execution": {...} }
"""

from __future__ import annotations

import json
import os
import sys
import time


def _write(result_file, status, payload, execution=None):
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"status": status, "payload": payload, "execution": execution or {}}, f)


def _fail(result_file, code, message, detail=""):
    _write(result_file, "error", {"code": code, "message": message, "detail": detail[:2000]})


def probe_inventory(role: str, device: dict) -> dict:
    """Package versions + device identity + supported compute types. No model load."""
    import platform

    info = {
        "role": role,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            props = torch.cuda.get_device_properties(0)
            info["device_total_memory_mib"] = round(props.total_memory / 1024 / 1024)
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"
    if role == "whisper":
        try:
            import ctranslate2

            info["ctranslate2"] = ctranslate2.__version__
            dev = "cuda" if device.get("backend") == "cuda" else "cpu"
            info["supported_compute_types"] = sorted(
                set(ctranslate2.get_supported_compute_types(dev, device_index=0))
            )
        except Exception as e:
            info["ctranslate2_error"] = f"{type(e).__name__}: {e}"
        try:
            import faster_whisper

            info["faster_whisper"] = faster_whisper.__version__
        except Exception as e:
            info["faster_whisper_error"] = f"{type(e).__name__}: {e}"
    else:
        try:
            import pyannote.audio

            info["pyannote_audio"] = pyannote.audio.__version__
        except Exception as e:
            info["pyannote_error"] = f"{type(e).__name__}: {e}"
    return info


def probe_torch(device: dict) -> dict:
    """Execute real CUDA kernels: allocation, matmul, conv, sync, finite check."""
    import torch

    backend = device.get("backend", "cpu")
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE: CUDA not available on the isolated device")

    dev = torch.device("cuda:0" if backend == "cuda" else "cpu")
    t0 = time.time()
    a = torch.randn(512, 512, device=dev)
    b = torch.randn(512, 512, device=dev)
    c = a @ b
    x = torch.randn(1, 1, 1000, device=dev)
    conv = torch.nn.Conv1d(1, 8, kernel_size=5).to(dev)
    y = conv(x)
    if backend == "cuda":
        torch.cuda.synchronize()
    for tensor in (c, y):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("RUNTIME_INCOMPATIBLE: non-finite values from GPU kernels")
    return {
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(0) if backend == "cuda" else None,
        "matmul_shape": list(c.shape),
        "conv_shape": list(y.shape),
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def _tone_wav(path: str, seconds: float = 3.0, rate: int = 16000):
    """Generate a synthetic tone fixture (no speech — proves the decode path)."""
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, wave.astype(np.float32), rate)


def probe_whisper_smoke(device: dict, model: str, audio_path: str | None) -> dict:
    """Load the production model and run a complete decode (generator consumed)."""
    import tempfile

    from faster_whisper import WhisperModel

    backend = device.get("backend", "cpu")
    if audio_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        _tone_wav(audio_path)

    t0 = time.time()
    model_obj = WhisperModel(model, device=backend, compute_type="auto")
    segments_gen, info = model_obj.transcribe(
        audio_path,
        beam_size=1,
        word_timestamps=True,
        vad_filter=False,
    )
    segments = list(segments_gen)  # consuming the generator runs the decode
    return {
        "model": model,
        "device": backend,
        "segments": len(segments),
        "language": getattr(info, "language", None),
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": "Synthetic-tone fixture proves model load + decode execution; "
        "transcript quality is validated separately with a real interview.",
    }


def probe_pyannote_smoke(device: dict, hf_token: str, audio_path: str | None) -> dict:
    """Run the real diarization pipeline on a short fixture."""
    import tempfile

    if device.get("backend") == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE: CUDA not available on the isolated device")

    if audio_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        _tone_wav(audio_path, seconds=5.0)

    import torch

    from pyannote.audio import Pipeline

    t0 = time.time()
    from torch.torch_version import TorchVersion

    from pyannote.audio.core.task import Problem, Resolution, Specifications

    with torch.serialization.safe_globals([TorchVersion, Specifications, Problem, Resolution]):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
    pipeline.to(torch.device("cuda:0" if device.get("backend") == "cuda" else "cpu"))
    diarization = pipeline(audio_path)
    turns = list(diarization.itertracks(yield_label=True))
    return {
        "device": device.get("backend"),
        "turns": len(turns),
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": "Tone fixture proves pipeline load + inference; diarization "
        "quality is validated separately with real two-speaker audio.",
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: probe_compute.py <args_file> <result_file>", file=sys.stderr)
        sys.exit(1)
    args_file, result_file = sys.argv[1], sys.argv[2]

    with open(args_file, "r", encoding="utf-8") as f:
        args = json.load(f)

    role = args.get("role", "whisper")
    device = args.get("device", {"backend": "cpu"})
    mode = args.get("mode", "inventory")

    try:
        if mode == "inventory":
            _write(result_file, "ok", probe_inventory(role, device))
        elif mode == "torch":
            _write(result_file, "ok", probe_torch(device))
        elif mode == "whisper_smoke":
            _write(
                result_file,
                "ok",
                probe_whisper_smoke(device, args.get("model", "large-v3-turbo"), args.get("audio_path")),
            )
        elif mode == "pyannote_smoke":
            _write(
                result_file,
                "ok",
                probe_pyannote_smoke(device, args.get("hf_token", ""), args.get("audio_path")),
            )
        else:
            _fail(result_file, "RUNTIME_INCOMPATIBLE", f"unknown probe mode {mode!r}")
    except Exception as e:
        import traceback

        _fail(result_file, "PROBE_FAILED", f"{type(e).__name__}: {e}", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
