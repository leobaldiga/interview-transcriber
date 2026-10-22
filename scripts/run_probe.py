#!/usr/bin/env python3
"""Manual compute-validation CLI (run on the deployment host).

Uses the exact same interpreter resolution, environment construction, and
probe subprocess as the web app's /api/compute/tests endpoint — a pass here
is a pass for the app. Requires no running web app.

Examples (from the repo root):
    python3 scripts/run_probe.py --device GPU-60ce4d58-... --smoke
    python3 scripts/run_probe.py --device cpu
    python3 scripts/run_probe.py            # probes the currently selected device

Run with the repo root on sys.path (it inserts itself). The probe subprocess
runs under each worker role's own interpreter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compute_devices
import compute_settings
import worker_runtime

PROBE_SCRIPT = REPO_ROOT / "scripts" / "probe_compute.py"
PROBE_TIMEOUT_S = 900


def _hf_token() -> str:
    from app import _load_hf_token  # same token resolution as the app

    return _load_hf_token()


def _launch_probe(py_exe: Path, env: dict, probe_args: dict) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        args_file = f.name
        json.dump(probe_args, f)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        result_file = f.name
    try:
        proc = subprocess.Popen(
            [str(py_exe), str(PROBE_SCRIPT), args_file, result_file],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=(os.name != "nt"),
        )
        try:
            proc.wait(timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
            return {"status": "error", "code": "PROBE_TIMEOUT", "message": f"exceeded {PROBE_TIMEOUT_S}s"}
        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            stderr = proc.stderr.read().decode(errors="replace")[-800:] if proc.stderr else ""
            return {
                "status": "error",
                "code": "WORKER_CRASHED",
                "message": f"probe exited with code {proc.returncode}",
                "stderr_tail": stderr,
            }
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        for p in (args_file, result_file):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def _print_step(name: str, result: dict):
    ok = result.get("status") == "ok"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if ok:
        payload = result.get("payload", {})
        for key in ("device_name", "device_capability", "torch", "ctranslate2", "faster_whisper",
                    "pyannote_audio", "supported_compute_types", "elapsed_seconds", "model", "turns"):
            if key in payload:
                print(f"         {key}: {payload[key]}")
    else:
        payload = result.get("payload", {})
        print(f"         code: {payload.get('code')}")
        print(f"         {payload.get('message', '')[:300]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate compute devices for the Interview Transcriber")
    parser.add_argument("--device", default="", help="GPU UUID, 'cpu', or omit for the selected device")
    parser.add_argument("--smoke", action="store_true", help="include model smoke tests (downloads/loads models)")
    parser.add_argument("--json", action="store_true", help="emit the raw JSON report")
    args = parser.parse_args()

    device_ref = None
    if args.device:
        try:
            device_ref = compute_settings.parse_device_override(args.device, field="device")
        except compute_settings.ComputeSettingsError as e:
            print(f"[ERROR] {e.code}: {e.message}")
            return 2
        if device_ref is None:
            device_ref = compute_settings.CPU_DEVICE
    else:
        store = compute_settings.ComputeSettingsStore()
        try:
            settings = store.load()
            device_ref = compute_settings.resolve_settings_device(
                settings, compute_devices.available_cuda_ids()
            )
        except (compute_settings.ComputeSettingsError, compute_devices.DeviceDiscoveryError) as e:
            print(f"[ERROR] {getattr(e, 'code', 'ERROR')}: {getattr(e, 'message', e)}")
            return 2

    hf_token = _hf_token()
    device_label = device_ref.get("id") or "cpu"
    print(f"Probing device: {device_label} (smoke tests: {'yes' if args.smoke else 'no'})")

    overall_ok = True
    report: dict = {"device": device_ref}
    for role in (worker_runtime.ROLE_WHISPER, worker_runtime.ROLE_PYANNOTE):
        print(f"[{role}]")
        try:
            py_exe = worker_runtime.resolve_worker_python(role)
            env = worker_runtime.build_worker_env(py_exe, device_ref)
        except worker_runtime.WorkerRuntimeError as e:
            print(f"  [FAIL] interpreter: {e.message}")
            overall_ok = False
            continue
        steps = {
            "inventory": {"role": role, "device": device_ref, "mode": "inventory"},
            "torch": {"role": role, "device": device_ref, "mode": "torch"},
        }
        if args.smoke:
            smoke = {"role": role, "device": device_ref, "mode": "whisper_smoke" if role == "whisper" else "pyannote_smoke"}
            if role == "pyannote":
                smoke["hf_token"] = hf_token
            steps["smoke"] = smoke
        report[role] = {}
        for name, probe_args in steps.items():
            result = _launch_probe(py_exe, env, probe_args)
            report[role][name] = result
            _print_step(name, result)
            if result.get("status") != "ok":
                overall_ok = False

    if args.json:
        print(json.dumps(report, indent=2))
    print(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
