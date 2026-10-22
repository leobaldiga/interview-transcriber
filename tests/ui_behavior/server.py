"""Test launcher: the real app with a faked two-GPU inventory.

Serves the ACTUAL app.py handlers on 127.0.0.1:8791 — only the device
inventory is faked (no GPU in this sandbox). The UI behavior test in
ui_behavior_test.mjs drives the real app.js against these handlers.
"""
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))  # repo root

import compute_devices

GPU_3060 = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c"
GPU_1050 = "GPU-60ce4d58-ba50-b3fa-82da-3e51b259e857"


def fake_devices(timeout=5.0):
    return [
        compute_devices.DeviceInfo(
            "cuda", GPU_3060, "NVIDIA GeForce RTX 3060", 0, None, "580.178.04",
            12288, 11943, 345,
        ),
        compute_devices.DeviceInfo(
            "cuda", GPU_1050, "NVIDIA GeForce GTX 1050 Ti", 1, None, "580.178.04",
            4096, 144, 3952,
        ),
    ]


compute_devices.list_cuda_devices = fake_devices
import app  # noqa: E402  (imports compute_devices too — same patched module)

app.compute_devices.list_cuda_devices = fake_devices

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app.app, host="127.0.0.1", port=8791, log_level="warning")
