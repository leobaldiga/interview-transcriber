# Interview Transcriber

Local FastAPI web app for transcribing, diarizing, reviewing, and organizing long-form interview audio.

**Stack:** FastAPI, faster-whisper, pyannote.audio, ffmpeg  
**Primary target:** Ubuntu 24.04 with NVIDIA GPU  
**Also supports:** Windows installation via `install.py`, with ongoing cross-platform worker hardening

## Overview

Interview Transcriber is built for research interviews, focus groups, oral histories, panels, podcasts, and other long recordings where speaker separation matters. It runs locally, keeps audio on your own machine, and provides a browser-based workflow for upload, transcription, diarization, review, editing, and export.

Main features:

- Drag-and-drop upload for `.mp3`, `.wav`, `.m4a`, `.ogg`, and `.flac`
- Whisper transcription with `faster-whisper`
- Live transcript streaming during active transcription
- Speaker diarization with `pyannote.audio`
- Job grouping and persistent history
- Built-in transcript editor with audio playback
- Downloadable `.txt`, `.srt`, and `.json` outputs
- Advanced settings for Whisper and diarization tuning

## Architecture

The app uses two separate Python environments to reduce dependency and CUDA conflicts.

- `.venv` — FastAPI app and `faster-whisper`
- `.venv-pyannote` — PyTorch and `pyannote.audio`

Typical job flow:

1. `app.py` receives the upload and creates a job.
2. `whisper_worker.py` runs transcription in a subprocess.
3. `diarize_worker.py` runs diarization in a separate subprocess.
4. The app merges timings and speaker labels, writes outputs, and exposes previews and downloads.

This design isolates Whisper from pyannote and makes worker failures easier to debug.

## Repository layout

```text
.
├── app.py
├── whisper_worker.py
├── diarize_worker.py
├── install.py
├── setup.sh
├── README.md
├── static/
│   ├── css/
│   ├── js/
│   └── favicon.svg
├── templates/
│   ├── index.html
│   └── editor.html
├── uploads/
└── outputs/
```

## Requirements

Recommended environment:

- Python 3.12
- `ffmpeg` and `ffprobe`
- Hugging Face token for pyannote model access
- NVIDIA GPU for best performance on long files

CPU-only use is possible, but much slower.

## Hugging Face setup

Before diarization will work, accept access for:

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Then generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Depending on your setup flow, the token may be provided during installation, placed in `.env`, or stored in a local token file read by `app.py`.

## Installation

### Linux (Ubuntu recommended)

Clone the repository:

```bash
git clone https://github.com/leobaldiga/interview-transcriber.git
cd interview-transcriber
```

Run the setup script:

```bash
bash setup.sh
```

Activate the main environment and start the app:

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python app.py
```

Then open:

- `http://localhost:8765`
- or your Tailscale IP on port `8765`

### Windows

Clone the repository:

```powershell
git clone https://github.com/leobaldiga/interview-transcriber.git
cd interview-transcriber
```

Run the installer:

```powershell
py install.py
```

Activate the main environment.

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

Start the app:

```powershell
python app.py
```

Then open `http://localhost:8765` in your browser.

### PowerShell execution policy note

If PowerShell blocks activation with an `Activate.ps1 cannot be loaded` message, allow scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or allow it only for the current shell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run:

```powershell
.\.venv\Scripts\Activate.ps1
```

A separate project-specific `.ps1` script is not required for basic installation. Python's built-in `venv` already creates `Activate.ps1` under `.venv\Scripts\`.

## Running the app

Once installed, the normal launch command is:

```bash
python app.py
```

The UI includes:

- file upload and advanced settings
- system status indicators
- live transcript streaming
- grouped job management
- transcript preview and downloads
- browser-based editing in `templates/editor.html`

## Outputs

Completed jobs can produce:

- `txt` — plain transcript
- `srt` — subtitle file
- `json` — word-level timestamps and structured output

Outputs are stored under `outputs/`, organized by group.

## Advanced settings

### Whisper

- **Beam size** — trade speed for accuracy
- **Temperature** — useful when Whisper loops or gets stuck
- **Chunk length** — useful for memory tuning
- **Hotwords** — improve recognition of expected terms and names
- **Condition on previous text** — keep context in single-pass mode

### Diarization

- **Preset** — Interview, Focus Group, Panel, Monologue
- **Min turn duration** — suppresses rapid speaker flipping
- **Min silence** — changes how aggressively turns split
- **Min cluster size** — reduces spurious speaker clusters
- **Min word count** — helps suppress tiny diarization fragments

## Transcript editor

Completed jobs can be opened in the editor for cleanup and review.

Editor workflow includes:

- transcript review alongside audio playback
- manual cleanup after segmentation or diarization errors
- in-browser revision without switching tools

## Troubleshooting

### Diarization fails but transcription works

Check:

- Hugging Face access was accepted for both required pyannote models
- the token is available to the app and subprocesses
- `.venv-pyannote` has compatible `torch`, `pytorch-lightning`, and `pyannote.audio` versions
- `ffmpeg` is installed and available on `PATH`

### Long recordings show timing drift

`whisper_worker.py` includes chunked / VAD-based logic intended to reduce cumulative timestamp drift on long interviews. If you change worker logic, test on a long real interview rather than only short clips.

### PowerShell activation fails

Use the execution-policy commands above, then activate `.venv\Scripts\Activate.ps1` again.

### GPU crashes or CUDA errors

Because Whisper and pyannote run in separate subprocesses and environments, restart the app after a worker crash and retry. Repeated failures usually indicate an environment or dependency mismatch.

## Current status

The main workflow is functional for upload, transcription, grouping, preview, editing, and export. The app is still under active development, especially around worker-level environment compatibility and cross-platform reliability.

## Intended use cases

- qualitative research interviews
- focus groups
- field recordings
- panel discussions
- podcast transcription
- multilingual local-first transcription workflows

## License

Add your preferred license here if you plan to publish or distribute the project.