# Interview Transcriber

Local web app for transcribing, diarizing, reviewing, and organizing long-form interview audio. Everything runs on your own machine — audio never leaves it.

**Stack:** FastAPI · faster-whisper · Silero VAD · pyannote.audio · ffmpeg  
**Primary target:** Ubuntu 24.04 with an NVIDIA GPU  
**Also supports:** Windows via `install.py`; CPU-only (slow)  
**Optional:** MCP tool server so an LLM in Open WebUI can drive transcriptions — see [`mcp_README.md`](mcp_README.md)

## Overview

Built for qualitative research interviews, focus groups, oral histories, panels, and other long recordings where speaker separation matters. Upload a file in the browser, get back a speaker-labeled transcript, then clean it up in the built-in editor.

Main features:

- Drag-and-drop upload for `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.mp4`, `.webm`
- VAD-chunked Whisper transcription with word-level timestamps and live streaming of segments as they arrive
- Speaker diarization with pyannote, with presets for interview / focus group / panel / monologue
- Speaker-turn cleanup: short-turn merging, backchannel absorption ("mm-hmm", "yeah"), optional cap on speaker count
- Thai/CJK-aware speaker assignment (segment-level rather than per-character)
- Selectable Whisper model per job, including Thai fine-tunes (Pathumma, biodatlab)
- Primary + secondary language hint for code-switched recordings
- Correction dictionaries (global and per-group) applied to transcribed words
- Job groups, queueing with position display, cancel / rename / move / delete, persistent history across restarts
- Transcript editor with synced audio playback and speaker rename/merge that regenerates all outputs
- Downloadable `.txt`, `.srt`, and word-level `.json`

## Architecture

Two separate Python environments avoid torch / CUDA dependency conflicts between Whisper and pyannote:

- `.venv` — FastAPI app, `faster-whisper`, torch (for Silero VAD), and the optional MCP proxy
- `.venv-pyannote` — pinned torch 2.8.0 + `pyannote.audio` 3.3.2

Job flow:

1. `app.py` receives the upload, creates a job, and enqueues it (one GPU job at a time).
2. If needed, ffmpeg converts the file to 16 kHz mono WAV.
3. `whisper_worker.py` runs in a subprocess from `.venv`: Silero VAD finds speech regions, merges them into ~20–30 s chunks, and faster-whisper transcribes each with word timestamps. Segments are streamed back over NDJSON so the UI updates live.
4. Dictionary corrections are applied.
5. `diarize_worker.py` runs in a subprocess from `.venv-pyannote` and returns speaker turns.
6. `app.py` assigns each word to the speaker with the greatest temporal overlap, merges short turns, absorbs backchannels, relabels speakers `Speaker 0..N`, and writes outputs.

Workers exit after each job, so VRAM is released between jobs and a CUDA crash in one worker cannot take down the server.

## Repository layout

```text
.
├── app.py                  # FastAPI app, job queue, speaker-merge logic, output builders
├── whisper_worker.py       # VAD + faster-whisper subprocess
├── diarize_worker.py       # pyannote subprocess
├── setup.sh                # Linux installer (creates both venvs)
├── install.py              # Cross-platform installer (Windows-friendly)
├── .env.example
├── mcp_proxy.py            # Optional MCP tool server (see mcp_README.md)
├── mcp_README.md
├── requirements-mcp.txt
├── whisper-mcp.service     # systemd user unit for the MCP proxy
├── static/
│   ├── css/
│   ├── js/
│   └── favicon.svg
├── templates/
│   ├── index.html          # Upload, settings, job list, previews
│   └── editor.html         # Transcript editor
├── uploads/                # Incoming audio (created at setup)
└── outputs/                # <group>/<job_id>/{txt,srt,json}, jobs.json, dictionary.json
```

## Requirements

- Python 3.10+ (3.12 recommended)
- `ffmpeg` and `ffprobe` on `PATH`
- Hugging Face token with access to the pyannote models (see below)
- NVIDIA GPU with recent drivers for practical speed on long files. CPU-only works but is much slower; whisper falls back to `int8` on CPU automatically.

Approximate VRAM (int8): `large-v3-turbo` ~1.5–2.5 GB, `large-v3` ~2.5–3.5 GB, plus ~1–2 GB for pyannote while it runs.

## Hugging Face setup

Diarization requires accepting the model terms for:

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Then create a token at <https://huggingface.co/settings/tokens>. The app looks for it in this order:

1. `HF_TOKEN` (or `HFTOKEN`) environment variable
2. `.hf_token` file in the repo root
3. `.hftoken` file in the repo root (this is what the installers write)

If no token is found, transcription still works but the diarization status indicator will show as unavailable and jobs will fail at the diarization step.

## Installation

### Linux (Ubuntu recommended)

```bash
git clone https://github.com/leobaldiga/interview-transcriber.git
cd interview-transcriber
bash setup.sh
```

The script checks Python, detects `nvidia-smi` to pick CUDA (cu128) or CPU torch wheels, creates both venvs, prompts for your HF token, and writes a default `.env`.

Start the app:

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python app.py
```

Open `http://localhost:8765`, or your Tailscale IP on port `8765`.

### Windows

```powershell
git clone https://github.com/leobaldiga/interview-transcriber.git
cd interview-transcriber
py install.py
```

Start the app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Or activate the venv first (`.\.venv\Scripts\Activate.ps1` in PowerShell, `.venv\Scripts\activate.bat` in cmd) and run `python app.py`.

If PowerShell blocks activation with an `Activate.ps1 cannot be loaded` message:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

or, for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Configuration

Environment variables (see `.env.example`):

| Var | Default | Notes |
|-----|---------|-------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8765` | HTTP port |
| `WHISPER_MODEL` | `large-v3-turbo` | Default model when none is chosen in the UI |
| `HF_TOKEN` | — | Hugging Face token (alternative to the token file) |

## Using the app

### Web UI

- **Upload panel** — pick a file, group, language (or auto-detect), optional secondary language, expected speaker count, and Whisper model. Expand *Advanced* to tune Whisper and diarization.
- **Status bar** — GPU name and VRAM usage, and whether whisper and diarization are ready.
- **Jobs list** — grouped, with live step text (`Queued — position 2 of 3`, `Diarizing — 41 speaker turns found...`). Cancel running jobs, rename, move between groups, delete.
- **Preview / download** — view or download `.txt`, `.srt`, `.json` for each job.
- **Editor** — open any completed job at `/editor/<job_id>` to listen and edit alongside the audio, rename speakers, or merge speaker labels. Speaker edits rewrite the `.json` source and regenerate `.txt` / `.srt`.

### Advanced settings

**Whisper**

- **Model** — `large-v3-turbo` (default, fastest large model), `large-v3` (max accuracy), or a Thai fine-tune
- **Beam size** — higher trades speed for accuracy (default 1)
- **Temperature** — raise slightly if Whisper loops or repeats
- **Chunk length** — only used in single-pass fallback mode
- **No-speech threshold** — raise to suppress hallucinated text in silence
- **Hotwords** — names, place names, and jargon you expect to hear
- **Condition on previous text** — keeps context in single-pass mode; off by default to limit runaway hallucination

**Diarization**

- **Preset** — Interview, Focus Group, Panel, Monologue (sets the four values below)
- **Min turn duration** — suppresses rapid speaker flipping
- **Min silence** — how aggressively pyannote splits turns
- **Min cluster size** — reduces spurious extra speakers
- **Min word count** — turns shorter than this and briefer than *min turn duration* are merged into a neighbor
- **Number of speakers** — if set, pyannote is constrained to it and extra clusters are folded into the most frequent speakers

### Correction dictionaries

`outputs/dictionary.json` (global) and `outputs/<group>/dictionary.json` (per group) map misrecognized tokens to corrections, e.g. `{"aggro": "agro", "Michigan state": "Michigan State"}`. Matching is case-insensitive on whole words; group entries override global ones. Edit through the UI or the `/api/dictionary` endpoint.

### Outputs

Each job writes to `outputs/<group>/<job_id>/`:

- `<name>.txt` — `[Speaker 0]: ...` paragraphs
- `<name>.srt` — subtitles broken at speaker changes, >1.5 s gaps, 5 s, or 20 words
- `<name>_words.json` — `{info, speakers, words[{text,start,end,speaker,probability}]}`; the source of truth for the editor
- `job.json` — job metadata and settings used

`outputs/jobs.json` is the index loaded at startup. Jobs that were mid-run when the server restarted are marked as errored and should be resubmitted.

### API

The UI is a thin client over a JSON API you can call directly:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/transcribe` | Multipart upload; returns `{job_id}` |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | List / inspect jobs |
| `POST` | `/api/jobs/{id}/cancel`, `/rename`, `/move` | Job management |
| `DELETE` | `/api/jobs/{id}` | Delete job and its outputs |
| `GET` | `/api/live/{id}` | SSE stream of segments during transcription |
| `GET` | `/api/download/{id}/{txt\|srt\|json}` | Download outputs |
| `GET` | `/api/preview/{id}/{type}` | Output as JSON |
| `GET` | `/api/audio/{id}`, `/api/words/{id}` | Editor data |
| `POST` | `/api/jobs/{id}/speakers/rename`, `/merge` | Speaker relabeling |
| `GET/POST` | `/api/dictionary` | Correction dictionaries |
| `GET` | `/api/status`, `/api/vram` | System / GPU status |
| `GET/POST` | `/api/groups` | Job groups |

### MCP tool server (optional)

`mcp_proxy.py` exposes `list_audio_files`, `list_whisper_models`, `transcribe_audio`, `get_job`, and `list_jobs` as MCP tools over Streamable HTTP on port `8002`, so a local LLM in Open WebUI can transcribe files dropped into `uploads/` and read the result. It does no GPU work itself — it only drives the running app's queue. Installation, the systemd unit, and Open WebUI registration are covered in [`mcp_README.md`](mcp_README.md).

## Troubleshooting

**Diarization fails but transcription works**  
Check that you accepted both pyannote model terms, the token is readable by the app (env var or `.hftoken`), and `ffmpeg` is on `PATH`. The diarization indicator in the status bar reflects whether the token and `.venv-pyannote` were found.

**"Whisper venv not found" / "pyannote venv not found"**  
Re-run `setup.sh` or `install.py`. Both venvs must sit in the repo root as `.venv` and `.venv-pyannote`.

**CUDA crash / SIGABRT during diarization**  
The worker reports this as an error and the server keeps running. Retry the job; if it recurs, check that the torch build in `.venv-pyannote` matches your driver and that VRAM isn't exhausted by another process (e.g. a local LLM). Consider a smaller Whisper model or waiting for the GPU to idle.

**Timing drift on long recordings**  
The VAD-chunked path in `whisper_worker.py` exists specifically to prevent cumulative drift. If VAD fails it logs a warning and falls back to single-pass mode, which can drift on multi-hour files. Test worker changes on a real long interview, not just short clips.

**Too many / too few speakers**  
Set *Number of speakers* explicitly if known. Otherwise raise *Min cluster size* to reduce spurious speakers, or lower it (Panel preset) if real speakers are being merged.

**Thai or other non-spaced scripts split mid-phrase**  
Speaker assignment is already segment-level for Thai/CJK. If turns still fragment, raise *Min turn duration*.

**PowerShell activation fails**  
See the execution-policy commands under Windows installation.

## Security note

The app has no authentication. It is designed to sit behind Tailscale or on `localhost`. Do not expose port `8765` (or `8002` for the MCP proxy) to an untrusted network, especially with human-subjects recordings.

## Current status

The full workflow — upload, transcription, diarization, grouping, editing, export, and MCP access — is in daily use. Active development is focused on worker environment compatibility, cross-platform reliability, and multilingual (Thai/English) diarization quality.

## License

Add your preferred license here if you plan to publish or distribute the project.
