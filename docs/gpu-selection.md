# GPU Selection

How the Interview Transcriber picks which GPU (or CPU) runs Whisper transcription
and pyannote diarization — and how to change it.

This feature exists so a small secondary GPU (e.g. a GTX 1050 Ti) can own the
transcription workload while a larger card (e.g. an RTX 3060) stays reserved
for an LLM server, without the two ever colliding on VRAM.

## Concepts

### Saved defaults (per-role, per-machine)

The app stores one saved selection per role in `config/compute.json`
(gitignored; override location with `COMPUTE_SETTINGS_PATH`):

```json
{
  "schema_version": 1,
  "transcription": {
    "device": { "backend": "cuda", "id": "GPU-<uuid>" },
    "compute_type": "auto"
  },
  "diarization": { "device": { "backend": "cuda", "id": "GPU-<uuid>" } }
}
```

- `transcription` — the device Whisper runs on, and its precision
  (`transcription.compute_type`): `"auto"` (default) or an explicit mode
  (`int8_float16`, `int8_float32`, `int8`, `float16`, `float32`).
- `diarization` — the device pyannote runs on (may differ).

Devices are addressed by **GPU UUID** (e.g. `GPU-60ce4d58-...`), not by CUDA
index — indexes can reorder across reboots or driver updates; UUIDs cannot.

### Per-job overrides

Every submission (web UI Advanced settings, `/api/transcribe`, or the MCP
proxy's `transcribe_audio`) accepts optional overrides:

| Form field | Meaning |
|---|---|
| `transcription_device` | GPU UUID or `"cpu"`. Empty = saved default. |
| `diarization_device` | GPU UUID, `"cpu"`, or `"same"` (follow this job's transcription device). Empty = saved default. |
| `whisper_compute_type` | Explicit precision for this job only. |

Overrides are validated at submission time and **snapshotted into the job**.
Changing saved defaults later never affects already-queued or running jobs.

### Compute types and the `auto` policy

With `compute_type: "auto"`:

- On CUDA, the worker probes the device in order
  `int8_float16 → int8_float32 → int8 → float16 → float32` and uses the first
  one the device reports as supported. Pascal cards (GTX 10-series) are
  expected to land on `int8_float32` (no usable FP16 throughput) — the
  on-device probe confirms it.
- On CPU, the order is `int8 → int8_float16 → ...` (int8 first).

Explicit values are passed straight to faster-whisper and fail loudly
(`UNSUPPORTED_COMPUTE_TYPE`) if the device can't run them.

### No silent fallbacks

- First run with **multiple GPUs** and nothing configured: submissions that
don't override the device are rejected with `DEVICE_NOT_CONFIGURED` — the
user must pick one. (Single-GPU and CPU-only machines auto-resolve their
only option on first run — a deterministic resolution, not a fallback.)
- If a selected/overridden GPU is missing at run time, the job fails with
  `DEVICE_NOT_AVAILABLE`. The app **never** falls back to another GPU or to
  CPU on its own.
- A worker process that dies mid-job marks the job failed
  (`WORKER_CRASHED`); chunk errors are fatal — there are no silently
  incomplete transcripts.

## Where settings live

| Thing | Location |
|---|---|
| Saved compute defaults | `config/compute.json` (env: `COMPUTE_SETTINGS_PATH`) |
| Worker venv overrides | `WHISPER_WORKER_PYTHON`, `PYANNOTE_WORKER_PYTHON` (see below) |
| Legacy venvs (untouched) | `.venv`, `.venv-pyannote` |

`WHISPER_WORKER_PYTHON` / `PYANNOTE_WORKER_PYTHON` point each worker at its
own interpreter (absolute path, or a path relative to the repo root). Unset =
workers use their previous interpreters (the legacy `.venv` /
`.venv-pyannote` defaults) — this is the rollback path. See
[pascal-worker-profile.md](pascal-worker-profile.md) for the ready-made
profile for Pascal GPUs.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/compute/devices` | Device inventory: CPU entry + every CUDA device (UUID, name, VRAM, free, driver) |
| `GET /api/compute/settings` | Saved defaults + what they resolve to right now |
| `PUT /api/compute/settings` | Save defaults (validated; bad device UUID rejected) |
| `POST /api/compute/tests` | Run the compatibility test suite, optionally with model smoke tests (Whisper + pyannote). Returns a test id; `GET /api/compute/tests/{id}` polls. |
| `GET /api/vram` | Per-device VRAM for the **selected** device (never pooled across GPUs) |
| `GET /api/status` | Job counts + per-device VRAM |

Error codes used across these APIs and jobs: `DEVICE_NOT_CONFIGURED`,
`DEVICE_NOT_AVAILABLE`, `UNSUPPORTED_COMPUTE_TYPE`, `CUDA_UNAVAILABLE`,
`WORKER_CRASHED`, `RUNTIME_INCOMPATIBLE`, `DEVICE_DISCOVERY_FAILED`,
`PROBE_TIMEOUT`.

## Web UI

The **Compute Devices** collapsible panel (below the VRAM monitor) shows each
CUDA device with name and free VRAM, lets you save default devices per role,
choose the Whisper precision, and run the compatibility Test. Per-job overrides
live under **Advanced settings**.

## MCP tools

- `get_compute_selection()` — current saved selection + device inventory.
- `transcribe_audio(..., transcription_device=..., diarization_device=...)` —
  per-job overrides; the proxy never picks a device itself.
- `get_job()` now reports the execution device (name, UUID, compute type,
  elapsed time) for completed jobs.

## Execution model

One GPU job at a time, strictly serialized: Whisper finishes and exits before
pyannote starts. Both workers report the device they actually ran on in the
result JSON (`execution` block), so transcripts record their provenance.

## Rollback

```bash
unset WHISPER_WORKER_PYTHON PYANNOTE_WORKER_PYTHON
# restart the app
```

Saved defaults can be cleared from the UI (or delete `config/compute.json`).
Legacy venvs and the host NVIDIA driver are never touched by any of this.
