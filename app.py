"""
Interview Transcriber v5 — Local webapp for transcribing and diarizing
research interview audio.

Stack: FastAPI + faster-whisper + pyannote.audio
Target: Ubuntu 24.04 + NVIDIA GPU + Tailscale
"""

import asyncio
import json
import os
import re
import shutil
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress torchaudio MP3 codec warning (cosmetic only, doesn't affect processing)
warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*")

# NOTE: torch is intentionally NOT imported at module level in app.py.
# Both Whisper (ctranslate2) and pyannote (PyTorch) run in isolated subprocesses.
# Importing torch here and calling torch.cuda.* would initialize a CUDA context
# in the main process that conflicts with the subprocess workers.
# GPU detection uses nvidia-smi directly instead.

import multiprocessing
_cpu_cores = multiprocessing.cpu_count()

# Note: lightning_fabric and torchaudio patches are applied in diarize_worker.py
# (runs in .venv-pyannote subprocess). They are NOT needed in the main app.

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

JOBS_INDEX_PATH = OUTPUT_DIR / "jobs.json"

DEFAULT_GROUP = "Ungrouped"

# Whisper model — override with WHISPER_MODEL env var
# large-v3-turbo is the stable default; large-v3 causes SIGABRT on RTX 3060 + driver 570
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

# Detect GPU via nvidia-smi rather than torch.cuda to avoid initializing CUDA
def _has_gpu() -> bool:
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, timeout=3)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False

WHISPER_DEVICE = "cuda" if _has_gpu() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"

# HuggingFace token for pyannote (read from env or .hf_token file)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    hf_token_file = BASE_DIR / ".hf_token"
    if hf_token_file.exists():
        HF_TOKEN = hf_token_file.read_text().strip()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Interview Transcriber", version="5.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Job tracking
jobs: dict = {}

# Active subprocess PIDs per job: job_id -> list of ints
# Populated while workers are running so /api/jobs/{id}/cancel can kill them.
_active_pids: dict = {}

# Live segment buffer: job_id -> list of text lines received so far
_live_segments: dict = {}

# ---------------------------------------------------------------------------
# Job queue — single-worker to prevent GPU contention
# Jobs are processed one at a time. Multiple uploads are serialised here.
# ---------------------------------------------------------------------------
import asyncio as _asyncio
_job_queue: asyncio.Queue = None   # initialised in startup_event
_queue_order: list = []             # ordered list of queued job_ids

def _update_queue_steps():
    """Update the step text for all queued (waiting) jobs to show current position."""
    waiting = [jid for jid in _queue_order if jid in jobs and jobs[jid]["status"] == "queued"]
    total = len(waiting)
    for i, jid in enumerate(waiting, 1):
        jobs[jid]["step"] = f"Queued — position {i} of {total}"


async def _queue_worker():
    """Background coroutine that drains _job_queue one job at a time."""
    while True:
        job_id, audio_path, language, num_speakers, whisper_model = \
            await _job_queue.get()
        try:
            if job_id in _queue_order:
                _queue_order.remove(job_id)
            # Update step to show it's now running
            if job_id in jobs and jobs[job_id]["status"] == "queued":
                jobs[job_id]["step"] = "Starting..."
            await process_job(job_id, audio_path, language, num_speakers, whisper_model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
                _save_jobs_index()
        finally:
            _job_queue.task_done()
            _update_queue_steps()  # refresh position labels for remaining queued jobs


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_jobs_index():
    """Write the jobs index (lightweight summary) to outputs/jobs.json."""
    index = []
    for job in jobs.values():
        index.append({
            "id": job.get("id"),
            "filename": job.get("filename"),
            "group": job.get("group", DEFAULT_GROUP),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at"),
            "duration": job.get("duration"),
            "language": job.get("language"),
            "files": job.get("files", {}),
            "num_speakers": job.get("num_speakers"),
            "error": job.get("error"),
        })
    JOBS_INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _save_job_json(job_id: str):
    """Write full job metadata to outputs/<group>/<job_id>/job.json."""
    if job_id not in jobs:
        return
    job = jobs[job_id]
    group = job.get("group", DEFAULT_GROUP)
    job_dir = OUTPUT_DIR / group / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")


def _load_jobs_from_disk():
    """On startup, load jobs from outputs/jobs.json into the jobs dict."""
    global jobs
    if not JOBS_INDEX_PATH.exists():
        return
    try:
        index = json.loads(JOBS_INDEX_PATH.read_text(encoding="utf-8"))
        for entry in index:
            job_id = entry.get("id")
            if not job_id:
                continue
            group = entry.get("group", DEFAULT_GROUP)
            # Try to load full job.json if available
            job_dir = OUTPUT_DIR / group / job_id
            job_json_path = job_dir / "job.json"
            if job_json_path.exists():
                try:
                    full_job = json.loads(job_json_path.read_text(encoding="utf-8"))
                    jobs[job_id] = full_job
                except Exception:
                    jobs[job_id] = dict(entry)
            else:
                # Fall back to index entry
                jobs[job_id] = dict(entry)
            # Ensure required keys
            jobs[job_id].setdefault("step", "Done" if entry.get("status") == "complete" else entry.get("status", ""))
            jobs[job_id].setdefault("num_speakers", entry.get("num_speakers"))
            jobs[job_id].setdefault("error", entry.get("error"))
            # On server startup, _live_segments is empty. Any job that was mid-processing
            # when the server stopped is stuck — it cannot resume. Mark it as interrupted.
            in_flight = ("queued", "transcribing", "diarizing", "merging")
            if jobs[job_id].get("status") in in_flight:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["step"] = "Interrupted — server restarted mid-job. Please resubmit."
                jobs[job_id]["error"] = "Server restarted mid-job"
                try:
                    job_json_path.write_text(
                        json.dumps(jobs[job_id], indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] Failed to load jobs from disk: {e}")


def _get_job_output_dir(job_id: str) -> Path:
    """Return the output directory for a given job, based on its group."""
    job = jobs.get(job_id, {})
    group = job.get("group", DEFAULT_GROUP)
    return OUTPUT_DIR / group / job_id


def _get_all_groups() -> list:
    """Return sorted list of all group folder names under OUTPUT_DIR."""
    groups = set()
    groups.add(DEFAULT_GROUP)
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir() and item.name not in ("__pycache__",):
                groups.add(item.name)
    return sorted(groups)


# ---------------------------------------------------------------------------
# Startup: load persisted jobs
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global _job_queue
    _load_jobs_from_disk()
    # Start the single-worker queue processor
    _job_queue = asyncio.Queue()
    asyncio.create_task(_queue_worker())
    print("[INFO] Job queue worker started")
    print(f"[INFO] Loaded {len(jobs)} job(s) from disk.")


# ---------------------------------------------------------------------------
# Worker script paths
# ---------------------------------------------------------------------------

_VENV_PYTHON     = BASE_DIR / ".venv" / "bin" / "python"
_PYANNOTE_PYTHON = BASE_DIR / ".venv-pyannote" / "bin" / "python"
_WHISPER_SCRIPT  = BASE_DIR / "whisper_worker.py"
_WORKER_SCRIPT   = BASE_DIR / "diarize_worker.py"

# Only one diarization subprocess at a time — prevents CUDA context collision
import threading
_diarization_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, language: str = "en",
                     job_id: Optional[str] = None,
                     model_size: Optional[str] = None,
                     adv: dict = None) -> tuple[list, dict]:
    """
    Transcribe audio via whisper_worker.py subprocess.
    Runs in .venv so ctranslate2 has its own isolated CUDA context.
    Segments are written to a temp file line-by-line and pushed to _live_segments.
    """
    import subprocess as _sp
    import tempfile

    python_bin = str(_VENV_PYTHON)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        args_file = f.name
        _adv = adv or {}
        json.dump({
            "audio_path":            audio_path,
            "language":              language,
            "model_size":            model_size or WHISPER_MODEL_SIZE,
            "device":                WHISPER_DEVICE,
            "compute_type":          WHISPER_COMPUTE_TYPE,
            "beam_size":             _adv.get("beam_size", 1),
            "temperature":           _adv.get("temperature", 0.0),
            "chunk_length":          _adv.get("chunk_length", 30),
            "no_speech_threshold":   _adv.get("no_speech_threshold", 0.6),
            "hotwords":              _adv.get("hotwords", ""),
            "condition_on_prev":     _adv.get("condition_on_prev", False),
        }, f)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        result_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
        segments_file = f.name

    # Initialise live buffer
    if job_id:
        _live_segments[job_id] = []

    try:
        _whisper_env = os.environ.copy()
        _whisper_env.update({
            "PYTHONPATH": "",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "HOME": str(Path.home()),
            "PATH": f"{_VENV_PYTHON.parent}:/usr/local/bin:/usr/bin:/bin",
            "TOKENIZERS_PARALLELISM": "false",
        })
        _whisper_env.pop("LD_LIBRARY_PATH", None)  # ctranslate2 must use its own cuDNN

        proc = _sp.Popen(
            [python_bin, str(_WHISPER_SCRIPT), args_file, result_file, segments_file],
            stdout=_sp.DEVNULL,
            stderr=_sp.PIPE,
            start_new_session=True,
            env=_whisper_env,
            preexec_fn=lambda: __import__('resource').setrlimit(
                __import__('resource').RLIMIT_CORE, (0, 0)
            ),
        )
        # Track PID for cancellation
        if job_id:
            _active_pids.setdefault(job_id, []).append(proc.pid)

        read_pos = 0  # byte position in segments_file we've consumed

        def _drain_segments():
            nonlocal read_pos
            with open(segments_file, "r") as sf:
                sf.seek(read_pos)
                new_data = sf.read()
                read_pos += len(new_data.encode("utf-8"))
            for line in new_data.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    seg_dict = json.loads(line)
                    if job_id and job_id in _live_segments:
                        _live_segments[job_id].append(seg_dict)
                except json.JSONDecodeError:
                    pass

        # Poll segments file while process runs, drain stderr to prevent pipe deadlock
        while proc.poll() is None:
            _drain_segments()
            if proc.stderr:
                line = proc.stderr.readline()
                if line:
                    print(line.decode(errors="replace").rstrip())
            time.sleep(0.2)

        # Process exited — drain remaining stderr
        if proc.stderr:
            for line in proc.stderr:
                print(line.decode(errors="replace").rstrip())

        proc.wait()

        # Final drain of segments file
        _drain_segments()

        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            raise RuntimeError(f"Whisper worker crashed (exit {proc.returncode})")

        with open(result_file) as f:
            result = json.load(f)

        if result["status"] == "error":
            raise RuntimeError(f"Whisper failed: {result['payload']}")

        return result["payload"]["segments"], result["payload"]["info"]

    finally:
        if job_id and job_id in _active_pids:
            _active_pids[job_id] = [p for p in _active_pids[job_id] if p != proc.pid]
        for p in (args_file, result_file, segments_file):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def diarize_audio(audio_path: str, num_speakers: Optional[int] = None,
                  job: Optional[dict] = None,
                  adv: dict = None) -> list:
    """
    Run pyannote diarization via diarize_worker.py in .venv-pyannote.
    That venv has no ctranslate2, so pyannote gets a clean CUDA context.
    Returns list of {start, end, speaker} dicts.
    """
    import subprocess as _sp
    import tempfile

    audio_duration = job.get("duration", 0) if job else 0

    python_bin = str(_PYANNOTE_PYTHON)
    if not os.path.exists(python_bin):
        raise RuntimeError(
            f"pyannote venv not found at {_PYANNOTE_PYTHON}. "
            "Run setup.sh to install .venv-pyannote."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        args_file = f.name
        _adv = adv or {}
        json.dump({
            "audio_path":        audio_path,
            "hf_token":          HF_TOKEN,
            "num_speakers":      num_speakers,
            "min_silence":       _adv.get("min_silence", 0.5),
            "min_cluster_size":  _adv.get("min_cluster_size", 75),
            "seg_onset":         _adv.get("seg_onset", 0.6),
            "seg_offset":        _adv.get("seg_offset", 0.4),
        }, f)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        result_file = f.name

    with _diarization_lock:
        try:
            _worker_env = os.environ.copy()
            _worker_env.update({
                "PYTHONPATH": "",
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
                "HOME": str(Path.home()),
                "PATH": f"{_PYANNOTE_PYTHON.parent}:/usr/local/bin:/usr/bin:/bin",
                "TOKENIZERS_PARALLELISM": "false",
            })
            _worker_env.pop("LD_LIBRARY_PATH", None)

            proc = _sp.Popen(
                [python_bin, str(_WORKER_SCRIPT), args_file, result_file],
                stdout=_sp.DEVNULL,
                stderr=_sp.PIPE,
                start_new_session=True,
                env=_worker_env,
                preexec_fn=lambda: __import__('resource').setrlimit(
                    __import__('resource').RLIMIT_CORE, (0, 0)
                ),
            )
            _job_id = job.get("id") if job else None
            if _job_id:
                _active_pids.setdefault(_job_id, []).append(proc.pid)

            completed_turns = 0
            while True:
                line = proc.stderr.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue

                text = line.decode(errors="replace").strip()
                if not text:
                    continue

                if job is not None:
                    if "speaker turns found" in text:
                        try:
                            n = int(text.split()[1])
                            completed_turns = n
                        except (ValueError, IndexError):
                            pass
                        dur_str = f" of ~{audio_duration/60:.0f}-min audio" if audio_duration else ""
                        job["step"] = f"Diarizing{dur_str} — {completed_turns} speaker turns found..."
                    elif "loaded on" in text:
                        device = "GPU" if "cuda" in text.lower() else "CPU"
                        job["step"] = f"Pyannote pipeline loaded on {device}, running inference..."
                    elif "loading" in text.lower():
                        job["step"] = "Loading pyannote pipeline..."

            proc.wait()
            returncode = proc.returncode

            if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
                raise RuntimeError(
                    f"Diarization worker crashed (exit code {returncode}). "
                    "This is usually a CUDA context issue. "
                    "Try restarting the server and resubmitting the job."
                )

            with open(result_file) as f:
                result = json.load(f)

            if result["status"] == "error":
                raise RuntimeError(f"Diarization failed:\n{result['payload']}")

            return result["payload"]

        finally:
            for p in (args_file, result_file):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


def assign_speakers_to_words(segments: list, diarization_turns: list) -> list:
    """
    Assign speaker labels to each word based on diarization overlap.
    Returns flat list of {text, start, end, speaker}.
    """
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            words.append({
                "text": w["text"],
                "start": w["start"],
                "end": w["end"],
                "speaker": None,
            })

    for word in words:
        mid = (word["start"] + word["end"]) / 2
        best_speaker = "UNKNOWN"
        best_overlap = 0
        for turn in diarization_turns:
            if turn["start"] <= mid <= turn["end"]:
                overlap = min(word["end"], turn["end"]) - max(word["start"], turn["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]
        word["speaker"] = best_speaker

    return words


def _is_cjk_or_thai(text: str) -> bool:
    """Return True if text contains Thai or CJK characters (no spaces between words)."""
    for ch in text:
        cp = ord(ch)
        if 0x0E00 <= cp <= 0x0E7F:  # Thai
            return True
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified
            return True
        if 0x3000 <= cp <= 0x303F:  # CJK Symbols
            return True
    return False


def _join_words(words: list) -> str:
    """Join word texts. For Thai/CJK, omit spaces; for Latin scripts, use spaces."""
    if not words:
        return ""
    combined = "".join(words)
    if _is_cjk_or_thai(combined):
        return combined
    return " ".join(words)


def _merge_short_turns(labeled_words: list, min_turn_duration: float = 1.5,
                        min_word_count: int = 4,
                        max_speakers: Optional[int] = None) -> list:
    """
    Merge very short speaker turns into adjacent turns. This reduces
    over-segmentation from pyannote — especially the common failure mode where
    single words are attributed to a spurious third "speaker" cluster.

    A turn is considered short if EITHER:
      - its total duration < min_turn_duration seconds, OR
      - it contains fewer than min_word_count words

    If max_speakers is set, only the top-N speakers by total word count are
    kept. All words from minority speakers are redistributed to the nearest
    (previous, else next) majority-speaker turn.

    Finally a same-speaker coalescence pass is run so that adjacent turns that
    ended up with the same speaker label are collapsed into one.

    Pass 1: same-speaker-only merge
    Pass 2: excess-speaker relabeling only when max_speakers exceeded
    Pass 3: coalescence
    """
    if not labeled_words:
        return labeled_words

    def _group(words):
        t = []
        cs, cw = None, []
        for w in words:
            if w["speaker"] != cs:
                if cw:
                    t.append((cs, cw))
                cs, cw = w["speaker"], []
            cw.append(w)
        if cw:
            t.append((cs, cw))
        return t

    def _flatten(turns):
        result = []
        for spk, wds in turns:
            for w in wds:
                w2 = dict(w)
                w2["speaker"] = spk
                result.append(w2)
        return result

    # Pass 1: same-speaker-only merge
    turns = _group(labeled_words)

    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(turns):
            spk, wds = turns[i]
            duration = (wds[-1]["end"] - wds[0]["start"]) if wds else 0
            is_short = (duration < min_turn_duration) and (len(wds) < min_word_count)
            if is_short and len(turns) > 1:
                if merged and merged[-1][0] == spk:
                    merged[-1] = (spk, merged[-1][1] + wds)
                    changed = True
                elif i + 1 < len(turns) and turns[i + 1][0] == spk:
                    next_spk, next_wds = turns[i + 1]
                    merged.append((spk, wds + next_wds))
                    i += 2
                    changed = True
                    continue
                else:
                    merged.append((spk, wds))
            else:
                merged.append((spk, wds))
            i += 1
        turns = merged

    # Pass 2: minority-speaker relabeling
    spk_counts: dict = {}
    for spk, wds in turns:
        spk_counts[spk] = spk_counts.get(spk, 0) + len(wds)

    n_unique = len(spk_counts)

    if max_speakers and n_unique > max_speakers:
        top_n = sorted(spk_counts, key=spk_counts.get, reverse=True)[:max_speakers]
        majority = set(top_n)

        flat = _flatten(turns)
        last_maj = None
        for w in flat:
            if w["speaker"] in majority:
                last_maj = w["speaker"]
            elif last_maj is not None:
                w["speaker"] = last_maj
        last_maj = None
        for w in reversed(flat):
            if w["speaker"] in majority:
                last_maj = w["speaker"]
            elif last_maj is not None:
                w["speaker"] = last_maj
        turns = _group(flat)

    # Pass 3: same-speaker coalescence
    coalesced = []
    for spk, wds in turns:
        if coalesced and coalesced[-1][0] == spk:
            coalesced[-1] = (spk, coalesced[-1][1] + wds)
        else:
            coalesced.append((spk, wds))
    turns = coalesced

    return _flatten(turns)

def absorb_backchannels(
    labeled_words: list,
    max_words: int = 2,
    max_duration: float = 0.9,
) -> list:
    """
    Relabel only true backchannel turns (e.g. 'yeah', 'mm-hmm') to match the
    surrounding speaker when sandwiched between two turns from the same speaker.

    This is intentionally conservative: it should improve readability without
    swallowing genuine short replies.
    """
    if not labeled_words:
        return labeled_words

    BACKCHANNELS = {
        "yeah", "yes", "yep", "no", "ok", "okay", "right",
        "mm", "mm-hmm", "mhm", "mhm.", "uh-huh", "uh huh",
        "hmm", "huh", "sure"
    }

    def _normalize_text(words):
        text = " ".join(w.get("text", "").strip().lower() for w in words)
        text = re.sub(r"[^\w\s-]", "", text).strip()
        return text

    # Group consecutive words by speaker into turns
    turns = []
    current_speaker = None
    current_words = []

    for w in labeled_words:
        spk = w.get("speaker")
        if spk != current_speaker:
            if current_words:
                turns.append((current_speaker, current_words))
            current_speaker = spk
            current_words = [w]
        else:
            current_words.append(w)

    if current_words:
        turns.append((current_speaker, current_words))

    # Absorb only genuine backchannels
    for i in range(1, len(turns) - 1):
        mid_speaker, mid_words = turns[i]
        prev_speaker, prev_words = turns[i - 1]
        next_speaker, next_words = turns[i + 1]

        if prev_speaker != next_speaker:
            continue
        if mid_speaker == prev_speaker:
            continue
        if not mid_words:
            continue

        duration = mid_words[-1]["end"] - mid_words[0]["start"]
        n_words = len(mid_words)
        mid_text = _normalize_text(mid_words)

        if (
            n_words <= max_words
            and duration <= max_duration
            and mid_text in BACKCHANNELS
        ):
            for w in mid_words:
                w["speaker"] = prev_speaker

    return labeled_words

def build_diarized_transcript(words: list) -> str:
    """Build a plain-text diarized transcript from speaker-labeled words."""
    if not words:
        return ""

    lines = []
    current_speaker = None
    current_words = []

    for word in words:
        if word["speaker"] != current_speaker:
            if current_words:
                lines.append(f"[{current_speaker}]: {_join_words(current_words)}")
                current_words = []
            current_speaker = word["speaker"]
        current_words.append(word["text"])

    if current_words:
        lines.append(f"[{current_speaker}]: {_join_words(current_words)}")

    return "\n\n".join(lines)


def build_srt(words: list, max_segment_duration: float = 5.0,
              max_words_per_segment: int = 20) -> str:
    """Build SRT subtitle content from speaker-labeled words."""
    segments = []
    current = {"words": [], "speaker": None, "start": None, "end": None}

    for word in words:
        should_break = False
        if current["speaker"] is None:
            pass
        elif word["speaker"] != current["speaker"]:
            should_break = True
        elif (word["start"] - current["end"]) > 1.5:
            should_break = True
        elif (word["start"] - current["start"]) > max_segment_duration:
            should_break = True
        elif len(current["words"]) >= max_words_per_segment:
            should_break = True

        if should_break and current["words"]:
            segments.append(current)
            current = {"words": [], "speaker": None, "start": None, "end": None}

        if current["start"] is None:
            current["start"] = word["start"]
        current["end"] = word["end"]
        current["speaker"] = word["speaker"]
        current["words"].append(word["text"])

    if current["words"]:
        segments.append(current)

    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = _format_srt_time(seg["start"])
        end_ts = _format_srt_time(seg["end"])
        text = _join_words(seg["words"])
        if not _is_cjk_or_thai(text):
            text = re.sub(r"  +", " ", text)
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"[{seg['speaker']}]: {text}")
        lines.append("")

    return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def process_job(job_id: str, audio_path: str, language: str,
                      num_speakers: Optional[int], whisper_model: str = ""):
    """Run the full transcription + diarization pipeline as a background task."""
    job = jobs[job_id]
    if whisper_model:
        job["whisper_model"] = whisper_model
    stem = Path(audio_path).stem
    group = job.get("group", DEFAULT_GROUP)
    job_output_dir = OUTPUT_DIR / group / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 0: Convert to WAV if needed.
        wav_path = audio_path
        original_ext = Path(audio_path).suffix.lower()
        needs_wav = original_ext in (".m4a", ".mp4", ".webm", ".aac", ".wma", ".opus")
        if needs_wav:
            job["step"] = "Converting audio to WAV for diarization..."
            wav_out = job_output_dir / (stem + "_diarize.wav")
            if not wav_out.exists():
                import subprocess as _sp
                r = _sp.run(
                    ["ffmpeg", "-y", "-i", audio_path,
                     "-ar", "16000", "-ac", "1",
                     str(wav_out)],
                    capture_output=True, timeout=300
                )
                if r.returncode != 0:
                    raise RuntimeError(f"ffmpeg conversion failed: {r.stderr.decode()[:300]}")
            wav_path = str(wav_out)

        # Step 1: Transcribe
        job["status"] = "transcribing"
        job["step"] = "Running Whisper transcription..."
        _save_jobs_index()
        loop = asyncio.get_event_loop()
        _model = job.get("whisper_model") or WHISPER_MODEL_SIZE
        _th_models = {
            "Vinxscribe/biodatlab-whisper-th-large-v3-faster",
            "nectec/Pathumma-whisper-th-large-v3",
            "nectec/Pathumma-whisper-th-medium",
        }
        _lang = None if language in ("auto", "", None) else language
        if _model in _th_models and _lang is None:
            _lang = "th"

        _secondary = job.get("secondary_language", "").strip()
        if _secondary and _secondary != _lang:
            _lang = None
            job["step"] = f"Transcribing (multilingual: {language}+{_secondary})..."

        adv = job.get("adv", {})
        segments, info = await loop.run_in_executor(
            None, transcribe_audio, audio_path, _lang, job_id, _model, adv
        )
        job["duration"] = info["duration"]
        job["language"] = info["language"]
        _save_jobs_index()

        # Apply custom dictionary corrections to word-level data
        group = job.get("group", DEFAULT_GROUP)
        dictionary = _load_dictionary(group)
        if dictionary:
            for seg in segments:
                seg["words"] = _apply_dictionary(seg.get("words", []), dictionary)
            job["step"] = f"Applied {len(dictionary)} dictionary corrections..."

        # Step 2: Diarize
        job["status"] = "diarizing"
        job["step"] = "Starting diarization subprocess (GPU)..."
        _save_jobs_index()
        diarization_turns = await loop.run_in_executor(
            None, diarize_audio, wav_path, num_speakers, job, adv
        )

        # Step 3: Merge speakers with words
        job["status"] = "merging"
        job["step"] = "Assigning speakers to transcript..."
        labeled_words = assign_speakers_to_words(segments, diarization_turns)

        _min_turn  = adv.get("min_turn_duration", 1.5)
        _min_words = adv.get("min_word_count", 4)
        _max_spk   = num_speakers
        labeled_words = _merge_short_turns(
            labeled_words,
            min_turn_duration=_min_turn,
            min_word_count=_min_words,
            max_speakers=_max_spk,
        )
        # absorb very short backchannel interjections into surrounding speaker
        labeled_words = absorb_backchannels(
            labeled_words,
            max_words=2,
            max_duration=1.0,
        )
        

        # Normalize speaker labels to Speaker 0, Speaker 1, etc.
        speaker_map = {}
        speaker_counter = 0
        for w in labeled_words:
            if w["speaker"] not in speaker_map:
                speaker_map[w["speaker"]] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            w["speaker"] = speaker_map[w["speaker"]]

        job["num_speakers"] = len(speaker_map)

        # Step 4: Build outputs
        job["status"] = "building_outputs"
        job["step"] = "Generating transcript and subtitle files..."

        transcript_text = build_diarized_transcript(labeled_words)
        srt_text = build_srt(labeled_words)

        txt_path  = job_output_dir / f"{stem}.txt"
        srt_path  = job_output_dir / f"{stem}.srt"
        json_path = job_output_dir / f"{stem}_words.json"

        txt_path.write_text(transcript_text, encoding="utf-8")
        srt_path.write_text(srt_text, encoding="utf-8")
        json_path.write_text(json.dumps({
            "info": info,
            "speakers": list(speaker_map.values()),
            "words": labeled_words,
        }, indent=2), encoding="utf-8")

        job["files"] = {
            "txt":   str(txt_path),
            "srt":   str(srt_path),
            "json":  str(json_path),
            "audio": str(audio_path),
        }

        job["status"] = "complete"
        job["step"] = "Done"
        job["completed_at"] = datetime.now().isoformat()
        _live_segments.pop(job_id, None)

        _save_job_json(job_id)
        _save_jobs_index()

    except Exception as e:
        job["status"] = "error"
        job["step"] = f"Error: {str(e)[:200]}"
        job["error"] = str(e)[:500]
        _live_segments.pop(job_id, None)
        import traceback
        traceback.print_exc()
        _save_job_json(job_id)
        _save_jobs_index()


# ---------------------------------------------------------------------------
# API routes — Core
# ---------------------------------------------------------------------------

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(BASE_DIR / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/status")
async def system_status():
    """Return system status: GPU availability, Whisper ready, Diarization ready."""
    # Use nvidia-smi for GPU info — never touch torch.cuda in the main process
    gpu_available = False
    gpu_name = None
    gpu_vram_total_mb = 0
    gpu_vram_used_mb = 0
    try:
        import subprocess as _sp
        r = _sp.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.decode().strip().split(",")
            if len(parts) >= 3:
                gpu_available = True
                gpu_name = parts[0].strip()
                gpu_vram_total_mb = int(parts[1].strip())
                gpu_vram_used_mb  = int(parts[2].strip())
    except Exception:
        pass
    gpu_vram_total = f"{gpu_vram_total_mb / 1024:.1f} GB" if gpu_vram_total_mb else None
    gpu_vram_used  = f"{gpu_vram_used_mb  / 1024:.1f} GB" if gpu_vram_total_mb else None
    gpu_vram_pct   = round(gpu_vram_used_mb / gpu_vram_total_mb * 100) if gpu_vram_total_mb else 0

    whisper_loaded = _WHISPER_SCRIPT.exists() and _VENV_PYTHON.exists()
    hf_token_set   = bool(HF_TOKEN)

    return {
        "gpu": gpu_available,
        "gpu_detail": {
            "name": gpu_name,
            "vram_total": gpu_vram_total,
            "vram_used":  gpu_vram_used,
            "vram_pct":   gpu_vram_pct,
        },
        "whisper": whisper_loaded or gpu_available,
        "diarization": hf_token_set,
    }


@app.get("/api/vram")
async def vram_status():
    """Return current GPU VRAM usage via nvidia-smi."""
    try:
        import subprocess as _sp
        r = _sp.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.decode().strip().split(",")
            if len(parts) >= 2:
                total_mb = int(parts[0].strip())
                used_mb  = int(parts[1].strip())
                return {
                    "vram_total": f"{total_mb / 1024:.1f} GB",
                    "vram_used":  f"{used_mb  / 1024:.1f} GB",
                    "vram_pct":   round(used_mb / total_mb * 100) if total_mb else 0,
                }
    except Exception:
        pass
    return {"vram_total": None, "vram_used": None, "vram_pct": 0}


@app.post("/api/transcribe")
async def start_transcription(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    secondary_language: str = Form(""),
    num_speakers: Optional[int] = Form(None),
    group: str = Form(DEFAULT_GROUP),
    whisper_model: str = Form(""),
    # Transcription advanced params
    adv_beam_size: int = Form(1),
    adv_temperature: float = Form(0.0),
    adv_chunk_length: int = Form(20),
    adv_no_speech_threshold: float = Form(0.6),
    adv_hotwords: str = Form(""),
    adv_condition_on_prev: bool = Form(False),
    # Diarization advanced params
    adv_diarization_preset: str = Form(""),
    adv_min_turn_duration: float = Form(1.2),
    adv_min_silence: float = Form(0.7),
    adv_min_cluster_size: int = Form(35),
    adv_min_word_count: int = Form(3),
):
    """Upload an audio file and start transcription + diarization."""
    allowed_ext = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    job_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    group = group.strip() or DEFAULT_GROUP
    group_dir = OUTPUT_DIR / group
    group_dir.mkdir(parents=True, exist_ok=True)

    # Apply diarization preset overrides
    preset_map = {
        "interview":   {"min_turn": 1.5, "min_silence": 0.5, "min_cluster": 75, "min_words": 4},
        "focus_group": {"min_turn": 0.8, "min_silence": 0.3, "min_cluster": 50, "min_words": 3},
        "panel":       {"min_turn": 0.5, "min_silence": 0.2, "min_cluster": 30, "min_words": 2},
        "monologue":   {"min_turn": 2.0, "min_silence": 1.5, "min_cluster": 120, "min_words": 6},
    }
    if adv_diarization_preset and adv_diarization_preset in preset_map:
        p = preset_map[adv_diarization_preset]
        adv_min_turn_duration = p["min_turn"]
        adv_min_silence       = p["min_silence"]
        adv_min_cluster_size  = p["min_cluster"]
        adv_min_word_count    = p.get("min_words", adv_min_word_count)

    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "group": group,
        "status": "queued",
        "step": "Waiting to start...",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "duration": None,
        "language": language,
        "secondary_language": secondary_language.strip(),
        "num_speakers": None,
        "files": {},
        "error": None,
        "whisper_model": whisper_model.strip(),
        "adv": {
            "beam_size":             adv_beam_size,
            "temperature":           adv_temperature,
            "chunk_length":          adv_chunk_length,
            "no_speech_threshold":   adv_no_speech_threshold,
            "hotwords":              adv_hotwords.strip(),
            "condition_on_prev":     adv_condition_on_prev,
            "min_turn_duration":     adv_min_turn_duration,
            "min_silence":           adv_min_silence,
            "min_cluster_size":      adv_min_cluster_size,
            "min_word_count":        adv_min_word_count,
            "diarization_preset":    adv_diarization_preset,
        },
    }

    _save_jobs_index()

    _queue_order.append(job_id)
    queue_pos = len(_queue_order)
    if queue_pos > 1:
        jobs[job_id]["step"] = f"Queued — position {queue_pos} of {queue_pos}"
    print(f"[INFO] Job {job_id} enqueued (pos {queue_pos}): model={whisper_model!r} language={language!r}")
    _job_queue.put_nowait((
        job_id, str(upload_path), language, num_speakers, whisper_model.strip()
    ))
    _update_queue_steps()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job by killing its worker subprocesses."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] not in ("queued", "transcribing", "diarizing", "merging"):
        raise HTTPException(400, "Job is not running")

    killed = []
    pids = _active_pids.get(job_id, [])
    for pid in list(pids):
        try:
            import signal as _sig
            os.killpg(os.getpgid(pid), _sig.SIGTERM)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass

    _active_pids.pop(job_id, None)
    _live_segments.pop(job_id, None)
    if job_id in _queue_order:
        _queue_order.remove(job_id)
    job["status"] = "error"
    job["step"] = "Cancelled by user."
    job["error"] = "Cancelled"
    _save_job_json(job_id)
    _save_jobs_index()
    _update_queue_steps()
    return {"ok": True, "killed_pids": killed}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from memory and disk."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs.pop(job_id)
    _live_segments.pop(job_id, None)
    group = job.get("group", DEFAULT_GROUP)
    job_dir = OUTPUT_DIR / group / job_id
    if job_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(job_dir, ignore_errors=True)
    _save_jobs_index()
    return {"ok": True, "deleted": job_id}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return list(jobs.values())


@app.get("/api/live/{job_id}")
async def live_transcript(job_id: str):
    """
    Server-Sent Events stream of live transcription segments.
    Each event: data: {"index": N, "start": X, "end": Y, "text": "..."}
    A final event with done=True signals transcription complete.
    """
    async def event_generator():
        sent_count = 0
        while True:
            if job_id not in jobs:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                return

            job = jobs[job_id]
            live_segs = _live_segments.get(job_id, [])

            while sent_count < len(live_segs):
                seg = live_segs[sent_count]
                payload = {
                    "index": sent_count,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                sent_count += 1

            if job["status"] not in ("queued", "transcribing"):
                yield f"data: {json.dumps({'done': True, 'total': sent_count})}\n\n"
                return

            yield ": keepalive\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download a result file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if file_type not in job.get("files", {}):
        raise HTTPException(404, f"No {file_type} file for this job")
    file_path = job["files"][file_type]
    return FileResponse(
        file_path,
        filename=Path(file_path).name,
        media_type="application/octet-stream",
    )


@app.get("/api/preview/{job_id}/{file_type}")
async def preview_file(job_id: str, file_type: str):
    """Get file content for in-browser preview."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if file_type not in job.get("files", {}):
        raise HTTPException(404, f"No {file_type} file for this job")
    content = Path(job["files"][file_type]).read_text(encoding="utf-8")
    return {"content": content, "filename": Path(job["files"][file_type]).name}


# ---------------------------------------------------------------------------
# API routes — Groups
# ---------------------------------------------------------------------------

@app.get("/api/groups")
async def list_groups():
    """Return list of all group names."""
    return _get_all_groups()


@app.post("/api/groups")
async def create_group(name: str = Form(...)):
    """Create a new group folder."""
    name = name.strip()
    if not name:
        raise HTTPException(400, "Group name cannot be empty")
    name = re.sub(r"[/\\<>:\"|?*]", "_", name)
    group_dir = OUTPUT_DIR / name
    group_dir.mkdir(parents=True, exist_ok=True)
    return _get_all_groups()


@app.post("/api/jobs/{job_id}/move")
async def move_job(job_id: str, group: str = Form(...)):
    """Move a job to a different group folder."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    group = group.strip()
    if not group:
        raise HTTPException(400, "Group name cannot be empty")
    group = re.sub(r"[/\\<>:\"|?*]", "_", group)

    old_group = job.get("group", DEFAULT_GROUP)
    if old_group == group:
        return {"ok": True, "group": group}

    old_dir = OUTPUT_DIR / old_group / job_id
    new_group_dir = OUTPUT_DIR / group
    new_group_dir.mkdir(parents=True, exist_ok=True)
    new_dir = new_group_dir / job_id

    if old_dir.exists():
        shutil.move(str(old_dir), str(new_dir))

    job["group"] = group
    new_files = {}
    for ftype, fpath in job.get("files", {}).items():
        p = Path(fpath)
        try:
            rel = p.relative_to(OUTPUT_DIR / old_group / job_id)
            new_files[ftype] = str(new_dir / rel)
        except ValueError:
            new_files[ftype] = fpath
    job["files"] = new_files

    _save_job_json(job_id)
    _save_jobs_index()
    return {"ok": True, "group": group}


# ---------------------------------------------------------------------------
# Transcript Editor API
# ---------------------------------------------------------------------------

def _get_dictionary_path(group: str = None, global_dict: bool = False) -> Path:
    """Return the path to the dictionary file for a group or the global fallback."""
    if global_dict or not group:
        return OUTPUT_DIR / "dictionary.json"
    return OUTPUT_DIR / group / "dictionary.json"


def _load_dictionary(group: str = None) -> dict:
    """Load dictionary with global fallback → group override merge."""
    result = {}
    global_path = _get_dictionary_path(global_dict=True)
    if global_path.exists():
        try:
            result.update(json.loads(global_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    if group:
        group_path = _get_dictionary_path(group=group)
        if group_path.exists():
            try:
                result.update(json.loads(group_path.read_text(encoding="utf-8")))
            except Exception:
                pass
    return result


def _apply_dictionary(words: list, dictionary: dict) -> list:
    """Apply dictionary corrections to a word list (case-insensitive matching)."""
    if not dictionary:
        return words
    lower_dict = {k.lower(): v for k, v in dictionary.items()}
    for w in words:
        key = w.get("text", "").strip().lower().strip(".,!?;:\"'")
        if key in lower_dict:
            orig = w["text"]
            stripped = orig.strip()
            leading  = orig[: len(orig) - len(orig.lstrip())]
            trailing = orig[len(orig.rstrip()):]
            w["text"] = leading + lower_dict[key] + trailing
    return words


@app.get("/editor/{job_id}", response_class=HTMLResponse)
async def transcript_editor(request: Request, job_id: str):
    """Full-page transcript editor."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse(request, "editor.html", {"job_id": job_id, "job": jobs[job_id]})


@app.get("/api/audio/{job_id}")
async def stream_audio(job_id: str):
    """Stream the original audio file for the editor player."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    audio_path = job.get("audio_path") or job.get("files", {}).get("audio")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(audio_path, media_type="audio/mpeg", filename=Path(audio_path).name)


@app.get("/api/words/{job_id}")
async def get_words(job_id: str):
    """Return word-level data (timestamps + confidence) for the editor."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    words_path = job.get("files", {}).get("json")
    if not words_path or not Path(words_path).exists():
        raise HTTPException(404, "Word data not found")
    data = json.loads(Path(words_path).read_text(encoding="utf-8"))
    return JSONResponse(content=data)


@app.post("/api/transcript/{job_id}")
async def save_transcript(job_id: str, content: str = Form(...)):
    """Save edited transcript text back to disk."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    txt_path = job.get("files", {}).get("txt")
    if not txt_path:
        raise HTTPException(400, "No transcript file for this job")
    Path(txt_path).write_text(content, encoding="utf-8")
    return {"ok": True}


@app.get("/api/dictionary")
async def get_dictionary(group: str = "", scope: str = "group"):
    """Get dictionary entries."""
    if scope == "global":
        path = _get_dictionary_path(global_dict=True)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return {"scope": "global", "entries": data}
    merged = _load_dictionary(group or None)
    group_path = _get_dictionary_path(group=group) if group else None
    group_only = json.loads(group_path.read_text(encoding="utf-8")) if (group_path and group_path.exists()) else {}
    return {"scope": "group", "group": group, "merged": merged, "group_entries": group_only}


@app.post("/api/dictionary")
async def save_dictionary(entries: str = Form(...), group: str = Form(""), scope: str = Form("group")):
    """Save dictionary entries."""
    try:
        data = json.loads(entries)
    except Exception:
        raise HTTPException(400, "Invalid JSON for entries")
    if scope == "global" or not group:
        path = _get_dictionary_path(global_dict=True)
    else:
        path = _get_dictionary_path(group=group)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": True, "saved": len(data), "scope": scope or "group"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )
