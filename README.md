# Interview Transcriber v5

Local webapp for transcribing and diarizing research interview audio.

**Stack:** FastAPI + faster-whisper + pyannote.audio  
**Target:** Ubuntu 24.04 + NVIDIA GPU + Tailscale

## Install

```bash
bash setup.sh
```

## Start

```bash
source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && python app.py
```

Then open http://localhost:8765 (or your Tailscale IP).

## Features

- Drag-and-drop audio upload (.mp3, .wav, .m4a, .ogg, .flac)
- Whisper transcription (large-v3-turbo default, Thai models supported)
- pyannote diarization with advanced parameter tuning
- Live transcript streaming during transcription
- Full-page transcript editor with audio playback and word confidence colors
- Job groups and persistence
- Download .txt, .srt, .json outputs

## Architecture

Two isolated Python virtual environments prevent CUDA context conflicts:

- `.venv` — FastAPI + faster-whisper (ctranslate2, no PyTorch)
- `.venv-pyannote` — PyTorch + pyannote.audio (no ctranslate2)

Each transcription job runs Whisper first in `.venv`, then diarization in `.venv-pyannote`. Both workers are subprocesses with independent CUDA contexts.

## Advanced Settings

- **Beam size** — 1 (greedy, fast) to 5 (accurate, slower)
- **Temperature** — 0 for deterministic output; increase if Whisper gets stuck
- **Chunk length** — 30s default; reduce if VRAM is tight
- **Diarization presets** — Interview, Focus Group, Panel, Monologue
- **Min turn duration / min word count** — suppresses single-word speaker artefacts
- **Min cluster size** — higher values reduce spurious extra-speaker clusters

## HuggingFace Setup

pyannote requires accepting model terms at:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Then generate a token at https://huggingface.co/settings/tokens and enter it when running `setup.sh`.
