# Interview Transcriber

Local web application for transcribing, diarizing, reviewing, and organizing long-form interview audio. Transcription and diarization run on the host rather than a hosted transcription API. Package installation and model downloads require network access; browser clients and optional MCP clients can retrieve recordings or transcripts through the application.

Stack: FastAPI, faster-whisper, Silero VAD, pyannote.audio, and ffmpeg. Default branch: `main`.

## Features

- Audio upload, queued jobs, live transcription, and persistent job history.
- Whisper model selection, including `large-v3-turbo`, `large-v3`, and Thai model options; primary and secondary language hints.
- Word timestamps, speaker diarization, turn cleanup, and configurable speaker counts/presets.
- Job groups, correction dictionaries, and a transcript editor with synchronized audio and speaker rename/merge.
- TXT, SRT, and word-level JSON exports.
- Persistent UUID-based GPU selection for transcription and diarization, with per-job overrides and compatibility tests.
- Optional MCP tools for transcription, group discovery, transcript search, and chunked retrieval.

## Installation status and requirements

The deployment validated on October 22, 2026 used Ubuntu 24.04, Python 3.12.3, a GTX 1050 Ti for both workers, and a separate RTX 3060 hosting an LLM. CLI and web model smoke tests passed, followed by full `large-v3-turbo` and `large-v3` jobs. This does not establish compatibility with every GPU, operating system, or dependency combination.

Use Python 3.12 for the documented deployment path. You also need Git, `ffmpeg` and `ffprobe` on PATH, and a Hugging Face token with access to the diarization models. Windows installation code is provided; today's hardware validation was on Linux. CPU execution is available but was not the focus of that validation. Native macOS/Metal acceleration is not established by this release.

**Warning:** `setup.sh` and `install.py` delete and recreate `.venv` and `.venv-pyannote`. They are fresh-install/rebuild tools, not routine upgrade commands. Back up an existing deployment before rebuilding.

### Current installer limitations

- The base installers choose CUDA 12.8 PyTorch wheels whenever `nvidia-smi` exists. They do not select a GPU-generation-specific profile automatically.
- Several base application dependencies are unpinned. A fresh default installation is not an exact reconstruction of the validated worker stack.
- The tested Pascal workers are installed separately using the manifests in `requirements/`. In particular, `av==17.0.1` is pinned there for faster-whisper 1.2.1 compatibility; the base installers do not consume that manifest.
- Do not change a working host NVIDIA driver merely to install the separate worker environments.

## Hugging Face access

Accept the terms for [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), then create an appropriate token at [Hugging Face settings](https://huggingface.co/settings/tokens).

The app checks `HF_TOKEN`/`HFTOKEN`, then `.hf_token`, then `.hftoken` in the repository root. The installers write `.hftoken`. Keep tokens out of Git; restrict token-file permissions on Linux with `chmod 600 .hftoken`. Missing token/model access prevents diarization.

## Fresh Linux installation

Run this only in a new checkout or when intentionally rebuilding the base environments. On Ubuntu 24.04:

```bash
sudo apt update
sudo apt install git ffmpeg python3.12 python3.12-venv python3-pip

git clone https://github.com/leobaldiga/interview-transcriber.git
cd interview-transcriber

# Select Python explicitly, rather than an unrelated python3 on PATH.
PYTHON=/usr/bin/python3.12 /usr/bin/python3.12 install.py
```

The cross-platform installer honors the `PYTHON` environment variable. The alternative Linux installer remains `bash setup.sh`, but it uses the first `python3` on PATH and recreates the same base environments.

For a Pascal GPU, continue with the separate worker profile below before submitting jobs. For other hardware, validate the default worker runtimes rather than assuming the base install is compatible.

### Start in a terminal

```bash
cd /path/to/interview-transcriber
export HOST=127.0.0.1
export PORT=8765
.venv/bin/python app.py
```

Open `http://127.0.0.1:8765`. For access from another machine, choose an appropriate reachable bind address and restrict access using your trusted network or Tailscale. The default HOST is `0.0.0.0`; the example above deliberately limits access to localhost.

`app.py` does not automatically load `.env`. Export needed variables in the launching shell, or set them in the service configuration. Merely editing `.env` does not configure a running service.

## Pascal / GTX 10-series worker profile

The validated profile uses PyTorch `2.8.0+cu126`, cuDNN `9.10.2.21`, faster-whisper `1.2.1`, CTranslate2 `4.7.1`, PyAV `17.0.1`, and pyannote.audio `3.3.2`.

From an existing base installation, create the replacement workers with a known Python 3.12 interpreter:

```bash
.venv-pyannote/bin/python scripts/setup_worker_envs.py \
    --python "$PWD/.venv-pyannote/bin/python"

.venv-whisper-pascal/bin/python -m pip check
.venv-pyannote-pascal/bin/python -m pip check

export WHISPER_WORKER_PYTHON="$PWD/.venv-whisper-pascal/bin/python"
export PYANNOTE_WORKER_PYTHON="$PWD/.venv-pyannote-pascal/bin/python"

nvidia-smi --query-gpu=uuid,name --format=csv,noheader

# Replace GPU-REPLACE-WITH-UUID with the intended GPU UUID.
.venv/bin/python scripts/run_probe.py \
    --device GPU-REPLACE-WITH-UUID --smoke --json
```

Set both worker overrides before probing; otherwise the probe uses the legacy interpreters. Require `OVERALL: PASS`. Start the app from the same shell, or put the overrides into its service configuration.

The worker installer creates `.venv-whisper-pascal` and `.venv-pyannote-pascal`, leaving the two base environments untouched. It refuses existing target directories unless `--force` is supplied; that flag deletes and rebuilds the selected replacement environments. Its Python guard accepts 3.9–3.13, but Python 3.12 is the version validated here. A failed installer `pip check` currently produces a warning, so run the explicit checks above as deployment gates.

Resolved package snapshots are written to `requirements/whisper-pascal-freeze.txt` and `requirements/pyannote-pascal-freeze.txt`. The installer uses the manifests, not these snapshots; they are audit records, not an automatically enforced lockfile.

See [GPU selection](docs/gpu-selection.md) and [Pascal profile details](docs/pascal-worker-profile.md). Some profile notes predate the completed hardware run; the dated validation statement above records the deployment outcome.

## First-run device selection

1. Open Compute Devices.
2. Choose a transcription GPU or CPU and a diarization device. Same as transcription follows the transcription selection.
3. Leave Whisper precision on Automatic unless intentionally testing a supported explicit mode.
4. Click Save defaults, refresh, and confirm persistence.
5. Enable Include model smoke tests and run Test selected configuration.
6. Submit a short real two-speaker recording; verify text, speaker assignment, and GPU allocation before processing long interviews.

On an unconfigured multi-GPU machine, choose a default or explicit job override before submission. Devices are saved by UUID, not an index that can reorder. Per-job overrides are under Advanced Settings; queued/running jobs retain their submitted settings.

Whisper finishes and exits before pyannote starts. Missing selected devices fail rather than silently moving work onto another GPU or CPU. On the tested 1050 Ti, supported CTranslate2 modes were `float32`, `int8`, and `int8_float32`; the application's automatic policy prefers `int8_float32` among those. Check completed-job execution metadata for the actual production selection.

## Windows

For a fresh install with Python 3.12 available through the launcher:

```powershell
git clone https://github.com/leobaldiga/interview-transcriber.git
cd interview-transcriber
$env:PYTHON = (py -3.12 -c "import sys; print(sys.executable)")
py -3.12 install.py
.\.venv\Scripts\python.exe app.py
```

Install ffmpeg separately and ensure `ffmpeg` and `ffprobe` are on PATH. Running the interpreter directly avoids PowerShell activation-policy changes. Windows environment variables use `$env:NAME = 'value'`. The base CUDA-selection and dependency limitations above also apply; Linux Pascal validation is not a Windows compatibility guarantee.

## Configuration

| Variable | Purpose |
|---|---|
| `HOST`, `PORT` | Listening address and port; defaults `0.0.0.0`, `8765` |
| `WHISPER_MODEL` | Default model; `large-v3-turbo` |
| `HF_TOKEN` / `HFTOKEN` | Alternative to the token files |
| `WHISPER_WORKER_PYTHON` | Whisper subprocess interpreter; unset uses legacy default |
| `PYANNOTE_WORKER_PYTHON` | Diarization subprocess interpreter; unset uses legacy default |
| `COMPUTE_SETTINGS_PATH` | Saved compute settings; default `config/compute.json` |

Worker interpreter paths can be absolute or relative to the repository root. Do not globally restrict the app to one GPU when relying on its full inventory and per-role device routing.

### Existing systemd service

For an existing `interview-transcriber.service`, inspect it before editing:

```bash
sudo systemctl cat interview-transcriber.service
sudo systemctl edit interview-transcriber.service
```

Add a drop-in, substituting your actual absolute repository path:

```ini
[Service]
Environment="WHISPER_WORKER_PYTHON=/absolute/path/interview-transcriber/.venv-whisper-pascal/bin/python"
Environment="PYANNOTE_WORKER_PYTHON=/absolute/path/interview-transcriber/.venv-pyannote-pascal/bin/python"
```

After active jobs finish:

```bash
sudo systemctl daemon-reload
sudo systemctl restart interview-transcriber.service
sudo systemctl status interview-transcriber.service --no-pager
sudo journalctl -u interview-transcriber.service -n 60 --no-pager
```

The installers do not create this backend service automatically. Its WorkingDirectory must point to the repository, and ExecStart should run the app using `.venv/bin/python`. Interpreter overrides configure the worker environments; saved Compute Devices settings configure their hardware. Shell exports do not update a systemd service.

## Upgrading an existing installation

Do not rerun the base installers for routine updates. Back up recordings, outputs/history, correction dictionaries, tokens, compute settings, and service configuration. Review local changes first, and finish active jobs before updating the deployed checkout.

```bash
cd /path/to/interview-transcriber
git status --short
git switch main
git pull --ff-only origin main
```

Stop if local changes or divergent history require reconciliation. Rebuild environments only when dependency changes require it, using a planned maintenance window. Restart the actual launcher/service after code changes, rerun compatibility tests, and validate representative audio. Updating Git alone does not install changed requirements.

### Rollback

Keep the old code reference and base environments until validation finishes. For a shell launch, unsetting the worker overrides returns interpreter selection to the legacy defaults after restart:

```bash
unset WHISPER_WORKER_PYTHON PYANNOTE_WORKER_PYTHON
```

For systemd, remove or edit only the worker-override drop-in, reload systemd, and restart after jobs finish. Saved GPU selections are separate and may also need adjustment; legacy environments are not necessarily compatible with the currently selected GPU. Restore the prior code version separately if rolling back application changes.

## Workflow and outputs

Upload MP3, WAV, M4A, OGG, or FLAC through the browser. Select language hints, model, group, and speaker count when known. Advanced settings expose transcription controls, diarization presets/cleanup, and per-job compute overrides.

Use the job list to monitor, rename, move, cancel, or delete jobs. Review/export completed results or open the synchronized transcript editor to rename/merge speakers.

Outputs live under `outputs/<group>/<job_id>/`: TXT, SRT, word-level JSON, and job metadata. `outputs/jobs.json` indexes job history. Dictionary files are `outputs/dictionary.json` globally and `outputs/<group>/dictionary.json` per group. Back up these directories; do not commit research recordings or transcripts. Interrupted jobs may need resubmission after a restart.

## Architecture and repository map

The FastAPI app queues jobs, decodes audio with ffmpeg when needed, launches isolated worker processes, and combines word timestamps with speaker turns. Whisper performs VAD-chunked transcription; pyannote performs diarization. Worker processes exit after use so model allocations are not deliberately kept resident between jobs.

| Path | Purpose |
|---|---|
| `app.py`, `templates/`, `static/` | Backend and browser interface |
| `whisper_worker.py`, `diarize_worker.py` | Model subprocesses |
| `compute_devices.py`, `compute_settings.py`, `worker_runtime.py` | Discovery, saved selection, isolation, interpreter routing |
| `setup.sh`, `install.py` | Base environment rebuild/install tools |
| `scripts/setup_worker_envs.py` | Separate Pascal worker installer |
| `scripts/run_probe.py`, `scripts/probe_compute.py` | Runtime and model compatibility probes |
| `requirements/` | Worker manifests and environment snapshots |
| `tests/` | Automated and browser-behavior test code |
| `mcp_proxy.py`, `mcp_README.md` | Optional MCP integration and its setup guide |
| `uploads/`, `outputs/`, `config/` | Local recordings, results, and machine settings |

## API and MCP

Common APIs include `/api/transcribe`, `/api/jobs`, `/api/jobs/{id}`, `/api/status`, `/api/vram`, `/api/groups`, transcript preview/download/editor routes, and correction dictionaries.

Compute endpoints:

- `GET /api/compute/devices`
- `GET /api/compute/settings`
- `PUT /api/compute/settings`
- `POST /api/compute/tests`
- `GET /api/compute/tests/{id}`

The optional MCP proxy normally listens on port 8002 and drives the running backend rather than executing models itself. Tools include `list_audio_files`, `list_whisper_models`, `transcribe_audio`, `get_job`, `list_jobs`, `get_compute_selection`, `list_groups`, `search_transcripts`, and `read_transcript_chunk`.

Search can match interview identifiers, filenames/titles, or transcript text. Chunk retrieval returns character offsets and a SHA-256 fingerprint; clients should follow `next_start_char` until `has_more` is false and stop if the fingerprint changes. See [MCP setup](mcp_README.md) for installation and client configuration.

## Troubleshooting

- Unsupported Python: use an explicit Python 3.12 interpreter for environment creation rather than whichever `python3` happens to resolve first.
- CUDA OOM: verify the saved/job UUID and selected GPU's free VRAM. Use a separate GPU, a smaller model, or an explicit CPU choice; do not assume automatic fallback.
- `metadata_errors` rejected by `av.open`: verify the worker's PyAV version. The validated Whisper profile pins 17.0.1; changing the app environment alone does not change an overridden worker environment.
- Pyannote checkpoint loading error: compare the probe and production worker paths and capture the full error. The probe has a scoped allowlist for known checkpoint metadata; do not automatically allowlist arbitrary globals or load untrusted checkpoints unrestricted.
- Diarization access failure: verify token readability and acceptance of both model terms.
- Missing worker interpreter: inspect the override paths and executable permissions before rebuilding anything.
- Empty device panel or missing UI behavior: refresh the page and inspect browser errors, then check the compute APIs and service logs.
- Poor speaker separation: set the expected speaker count when known and review cleanup/preset settings. Smoke tests do not establish interview quality.
- Long-recording drift: inspect VAD/chunking logs and validate on representative long audio rather than only synthetic fixtures.

## Security and privacy

The application has no authentication. Restrict ports 8765 and 8002 to localhost or a trusted network/Tailscale deployment; do not expose them to the public internet. Treat audio, transcripts, tokens, and backups as sensitive. Local inference does not prevent a remote browser or MCP client from accessing data through an exposed application.

## License

Original application code is licensed under the [MIT License](LICENSE). Third-party libraries, model weights, and binaries remain subject to their own licenses and access terms.
