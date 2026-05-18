"""
Interview Transcriber — Local webapp for transcribing and diarizing
research interview audio.

Stack: FastAPI + faster-whisper + pyannote.audio
Target: Ubuntu 24.04 + NVIDIA GPU + Tailscale
"""

import asyncio
import json
import mimetypes
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*")

_cpu_cores = multiprocessing.cpu_count()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS_INDEX_PATH = OUTPUT_DIR / "jobs.json"
DEFAULT_GROUP = "Ungrouped"
IS_WINDOWS = os.name == "nt"

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3-turbo").strip() or "large-v3-turbo"


def _has_gpu() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


WHISPER_DEVICE = "cuda" if _has_gpu() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"


def _load_hf_token() -> str:
    env_candidates = (
        os.environ.get("HF_TOKEN", "").strip(),
        os.environ.get("HFTOKEN", "").strip(),
    )
    for token in env_candidates:
        if token:
            return token

    file_candidates = (
        BASE_DIR / ".hf_token",
        BASE_DIR / ".hftoken",
    )
    for token_file in file_candidates:
        try:
            if token_file.exists():
                token = token_file.read_text(encoding="utf-8").strip()
                if token:
                    return token
        except Exception:
            pass
    return ""


HF_TOKEN = _load_hf_token()


def _venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if IS_WINDOWS else venv_dir / "bin" / "python"


def _build_worker_env(py_exe: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "HOME": str(Path.home()),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    path_parts = [str(py_exe.parent)]
    if IS_WINDOWS:
        path_parts.extend([str(Path(sys.executable).parent), env.get("PATH", "")])
    else:
        path_parts.extend(["/usr/local/bin", "/usr/bin", "/bin", env.get("PATH", "")])
    env["PATH"] = os.pathsep.join([p for p in path_parts if p])

    if not IS_WINDOWS:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def _popen_kwargs() -> dict:
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _spawn_worker(cmd: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env, **_popen_kwargs())


def _terminate_pid(pid: int) -> bool:
    try:
        if IS_WINDOWS:
            proc = subprocess.Popen(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.wait(timeout=10)
            return proc.returncode == 0
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired, OSError):
        return False


def _guess_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(str(path))
    return media_type or "application/octet-stream"


_VENV_PYTHON = _venv_python_path(BASE_DIR / ".venv")
_PYANNOTE_PYTHON = _venv_python_path(BASE_DIR / ".venv-pyannote")
_WHISPER_SCRIPT = BASE_DIR / "whisper_worker.py"
_WORKER_SCRIPT = BASE_DIR / "diarize_worker.py"

app = FastAPI(title="Interview Transcriber", version="5.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

jobs: dict = {}
_active_pids: dict = {}
_live_segments: dict = {}
_job_queue: asyncio.Queue | None = None
_queue_order: list = []
_diarization_lock = threading.Lock()


def _update_queue_steps():
    waiting = [jid for jid in _queue_order if jid in jobs and jobs[jid].get("status") == "queued"]
    total = len(waiting)
    for i, jid in enumerate(waiting, 1):
        jobs[jid]["step"] = f"Queued — position {i} of {total}"


async def _queue_worker():
    while True:
        job_id, audio_path, language, num_speakers, whisper_model = await _job_queue.get()
        try:
            if job_id in _queue_order:
                _queue_order.remove(job_id)
            if job_id in jobs and jobs[job_id].get("status") == "queued":
                jobs[job_id]["step"] = "Starting..."
            await process_job(job_id, audio_path, language, num_speakers, whisper_model)
        except Exception as e:
            import traceback

            traceback.print_exc()
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["step"] = f"Error: {str(e)[:200]}"
                jobs[job_id]["error"] = str(e)[:500]
                _save_job_json(job_id)
                _save_jobs_index()
        finally:
            _job_queue.task_done()
            _update_queue_steps()


def _to_rel_path(path: str | Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(BASE_DIR.resolve()))
    except Exception:
        return str(p)


def _resolve_job_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


def _normalize_job_paths(job: dict) -> dict:
    files = job.get("files", {})
    if isinstance(files, dict):
        job["files"] = {
            key: _to_rel_path(value) if isinstance(value, (str, Path)) else value for key, value in files.items()
        }
    for key in ("audiopath", "audio_path"):
        if isinstance(job.get(key), (str, Path)):
            job[key] = _to_rel_path(job[key])
    return job


def _save_jobs_index():
    index = []
    for job in jobs.values():
        j = _normalize_job_paths(dict(job))
        index.append(
            {
                "id": j.get("id"),
                "filename": j.get("filename"),
                "group": j.get("group", DEFAULT_GROUP),
                "status": j.get("status"),
                "created_at": j.get("created_at"),
                "completed_at": j.get("completed_at"),
                "duration": j.get("duration"),
                "language": j.get("language"),
                "files": j.get("files", {}),
                "num_speakers": j.get("num_speakers"),
                "error": j.get("error"),
            }
        )
    JOBS_INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _save_job_json(job_id: str):
    if job_id not in jobs:
        return
    job = _normalize_job_paths(dict(jobs[job_id]))
    group = job.get("group", DEFAULT_GROUP)
    job_dir = OUTPUT_DIR / group / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")


def _load_jobs_from_disk():
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
            job_json_path = OUTPUT_DIR / group / job_id / "job.json"
            if job_json_path.exists():
                try:
                    full_job = json.loads(job_json_path.read_text(encoding="utf-8"))
                    jobs[job_id] = _normalize_job_paths(full_job)
                except Exception:
                    jobs[job_id] = _normalize_job_paths(dict(entry))
            else:
                jobs[job_id] = _normalize_job_paths(dict(entry))

            jobs[job_id].setdefault("step", "Done" if entry.get("status") == "complete" else entry.get("status", ""))
            jobs[job_id].setdefault("num_speakers", entry.get("num_speakers"))
            jobs[job_id].setdefault("error", entry.get("error"))

            if jobs[job_id].get("status") in ("queued", "transcribing", "diarizing", "merging", "building_outputs"):
                jobs[job_id]["status"] = "error"
                jobs[job_id]["step"] = "Interrupted — server restarted mid-job. Please resubmit."
                jobs[job_id]["error"] = "Server restarted mid-job"
                try:
                    _save_job_json(job_id)
                except Exception:
                    pass

        _save_jobs_index()
        for job_id in list(jobs.keys()):
            try:
                _save_job_json(job_id)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to load jobs from disk: {e}")


def _get_all_groups() -> list:
    groups = {DEFAULT_GROUP}
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir() and item.name not in ("__pycache__",):
                groups.add(item.name)
    return sorted(groups)


@app.on_event("startup")
async def startup_event():
    global _job_queue
    _load_jobs_from_disk()
    _job_queue = asyncio.Queue()
    asyncio.create_task(_queue_worker())
    print("[INFO] Job queue worker started")
    print(f"[INFO] Loaded {len(jobs)} job(s) from disk.")


def transcribe_audio(audio_path: str, language: str = "en", job_id: Optional[str] = None,
                     model_size: Optional[str] = None, adv: dict = None) -> tuple[list, dict]:
    import tempfile

    python_bin = str(_VENV_PYTHON)
    if not os.path.exists(python_bin):
        raise RuntimeError(f"Whisper venv not found at {_VENV_PYTHON}.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        args_file = f.name
        _adv = adv or {}
        json.dump(
            {
                "audio_path": audio_path,
                "language": language,
                "model_size": model_size or WHISPER_MODEL_SIZE,
                "device": WHISPER_DEVICE,
                "compute_type": WHISPER_COMPUTE_TYPE,
                "beam_size": _adv.get("beam_size", 1),
                "temperature": _adv.get("temperature", 0.0),
                "chunk_length": _adv.get("chunk_length", 30),
                "no_speech_threshold": _adv.get("no_speech_threshold", 0.6),
                "hotwords": _adv.get("hotwords", ""),
                "condition_on_prev": _adv.get("condition_on_prev", False),
            },
            f,
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        result_file = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False, encoding="utf-8") as f:
        segments_file = f.name

    if job_id:
        _live_segments[job_id] = []

    proc = None
    try:
        proc = _spawn_worker(
            [python_bin, str(_WHISPER_SCRIPT), args_file, result_file, segments_file],
            env=_build_worker_env(_VENV_PYTHON),
        )

        if job_id:
            _active_pids.setdefault(job_id, []).append(proc.pid)

        def _drain_segments():
            try:
                with open(segments_file, "r", encoding="utf-8") as sf:
                    for line in sf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            seg_dict = json.loads(line)
                            if job_id and job_id in _live_segments:
                                if not _live_segments[job_id] or _live_segments[job_id][-1] != seg_dict:
                                    _live_segments[job_id].append(seg_dict)
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                pass

        while proc.poll() is None:
            _drain_segments()
            if proc.stderr:
                line = proc.stderr.readline()
                if line:
                    print(line.decode(errors="replace").rstrip())
            time.sleep(0.2)

        if proc.stderr:
            for line in proc.stderr:
                print(line.decode(errors="replace").rstrip())

        proc.wait()
        _drain_segments()

        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            raise RuntimeError(f"Whisper worker crashed (exit {proc.returncode})")

        with open(result_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        if result.get("status") == "error":
            raise RuntimeError(f"Whisper failed: {result.get('payload')}")

        payload = result.get("payload", {})
        return payload.get("segments", []), payload.get("info", {})

    finally:
        if job_id and job_id in _active_pids and proc is not None:
            _active_pids[job_id] = [p for p in _active_pids[job_id] if p != proc.pid]
            if not _active_pids[job_id]:
                _active_pids.pop(job_id, None)
        for p in (args_file, result_file, segments_file):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def diarize_audio(audio_path: str, num_speakers: Optional[int] = None, job: Optional[dict] = None, adv: dict = None) -> list:
    import tempfile

    audio_duration = job.get("duration", 0) if job else 0
    python_bin = str(_PYANNOTE_PYTHON)
    if not os.path.exists(python_bin):
        raise RuntimeError(
            f"pyannote venv not found at {_PYANNOTE_PYTHON}. Install the .venv-pyannote environment first."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        args_file = f.name
        _adv = adv or {}
        json.dump(
            {
                "audio_path": audio_path,
                "hf_token": HF_TOKEN,
                "num_speakers": num_speakers,
                "min_silence": _adv.get("min_silence", 0.5),
                "min_cluster_size": _adv.get("min_cluster_size", 75),
                "seg_onset": _adv.get("seg_onset", 0.6),
                "seg_offset": _adv.get("seg_offset", 0.4),
            },
            f,
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        result_file = f.name

    proc = None
    with _diarization_lock:
        try:
            proc = _spawn_worker(
                [python_bin, str(_WORKER_SCRIPT), args_file, result_file],
                env=_build_worker_env(_PYANNOTE_PYTHON),
            )

            _job_id = job.get("id") if job else None
            if _job_id:
                _active_pids.setdefault(_job_id, []).append(proc.pid)

            completed_turns = 0
            while True:
                line = proc.stderr.readline() if proc.stderr else b""
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
            if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
                raise RuntimeError(
                    f"Diarization worker crashed (exit code {proc.returncode}). This is usually a CUDA context issue."
                )

            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("status") == "error":
                raise RuntimeError(f"Diarization failed:\n{result.get('payload')}")
            return result.get("payload", [])

        finally:
            _job_id = job.get("id") if job else None
            if _job_id and proc is not None and _job_id in _active_pids:
                _active_pids[_job_id] = [p for p in _active_pids[_job_id] if p != proc.pid]
                if not _active_pids[_job_id]:
                    _active_pids.pop(_job_id, None)
            for p in (args_file, result_file):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


def assign_speakers_to_words(segments: list, diarization_turns: list) -> list:
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            if "text" not in w or "start" not in w or "end" not in w:
                continue
            words.append({"text": w["text"], "start": w["start"], "end": w["end"], "speaker": None})

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
    for ch in text:
        cp = ord(ch)
        if 0x0E00 <= cp <= 0x0E7F or 0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F:
            return True
    return False


def _join_words(words: list) -> str:
    if not words:
        return ""
    combined = "".join(words)
    return combined if _is_cjk_or_thai(combined) else " ".join(words)


def _merge_short_turns(labeled_words: list, min_turn_duration: float = 1.5,
                       min_word_count: int = 4, max_speakers: Optional[int] = None) -> list:
    if not labeled_words:
        return labeled_words

    def _group(words):
        t, cs, cw = [], None, []
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

    spk_counts: dict = {}
    for spk, wds in turns:
        spk_counts[spk] = spk_counts.get(spk, 0) + len(wds)

    if max_speakers and len(spk_counts) > max_speakers:
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

    coalesced = []
    for spk, wds in turns:
        if coalesced and coalesced[-1][0] == spk:
            coalesced[-1] = (spk, coalesced[-1][1] + wds)
        else:
            coalesced.append((spk, wds))
    return _flatten(coalesced)


def absorb_backchannels(labeled_words: list, max_words: int = 2, max_duration: float = 0.9) -> list:
    if not labeled_words:
        return labeled_words

    backchannels = {"yeah", "yes", "yep", "no", "ok", "okay", "right", "mm", "mm-hmm", "mhm", "mhm.", "uh-huh", "uh huh", "hmm", "huh", "sure"}

    def _normalize_text(words):
        text = " ".join(w.get("text", "").strip().lower() for w in words)
        return re.sub(r"[^\w\s-]", "", text).strip()

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

    for i in range(1, len(turns) - 1):
        mid_speaker, mid_words = turns[i]
        prev_speaker, _ = turns[i - 1]
        next_speaker, _ = turns[i + 1]
        if prev_speaker != next_speaker or mid_speaker == prev_speaker or not mid_words:
            continue
        duration = mid_words[-1]["end"] - mid_words[0]["start"]
        if len(mid_words) <= max_words and duration <= max_duration and _normalize_text(mid_words) in backchannels:
            for w in mid_words:
                w["speaker"] = prev_speaker
    return labeled_words


def build_diarized_transcript(words: list) -> str:
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


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(words: list, max_segment_duration: float = 5.0, max_words_per_segment: int = 20) -> str:
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
        text = _join_words(seg["words"])
        if not _is_cjk_or_thai(text):
            text = re.sub(r"  +", " ", text)
        lines.extend([
            str(i),
            f"{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}",
            f"[{seg['speaker']}]: {text}",
            "",
        ])
    return "\n".join(lines)


def _get_dictionary_path(group: str = None, global_dict: bool = False) -> Path:
    if global_dict or not group:
        return OUTPUT_DIR / "dictionary.json"
    return OUTPUT_DIR / group / "dictionary.json"


def _load_dictionary(group: str = None) -> dict:
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
    if not dictionary:
        return words
    lower_dict = {k.lower(): v for k, v in dictionary.items()}
    for w in words:
        key = w.get("text", "").strip().lower().strip(".,!?;:\"'")
        if key in lower_dict:
            orig = w["text"]
            leading = orig[: len(orig) - len(orig.lstrip())]
            trailing = orig[len(orig.rstrip()):]
            w["text"] = leading + lower_dict[key] + trailing
    return words


async def process_job(job_id: str, audio_path: str, language: str, num_speakers: Optional[int], whisper_model: str = ""):
    job = jobs[job_id]
    if whisper_model:
        job["whisper_model"] = whisper_model

    audio_path = str(_resolve_job_path(audio_path) or audio_path)
    stem = Path(audio_path).stem
    group = job.get("group", DEFAULT_GROUP)
    job_output_dir = OUTPUT_DIR / group / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        wav_path = audio_path
        original_ext = Path(audio_path).suffix.lower()
        needs_wav = original_ext in (".m4a", ".mp4", ".webm", ".aac", ".wma", ".opus")
        if needs_wav:
            job["step"] = "Converting audio to WAV for diarization..."
            wav_out = job_output_dir / f"{stem}_diarize.wav"
            if not wav_out.exists():
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", str(wav_out)],
                    capture_output=True,
                    timeout=300,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"ffmpeg conversion failed: {r.stderr.decode(errors='replace')[:300]}")
            wav_path = str(wav_out)

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
        segments, info = await loop.run_in_executor(None, transcribe_audio, audio_path, _lang, job_id, _model, adv)
        job["duration"] = info.get("duration")
        job["language"] = info.get("language")
        _save_jobs_index()

        dictionary = _load_dictionary(group)
        if dictionary:
            for seg in segments:
                seg["words"] = _apply_dictionary(seg.get("words", []), dictionary)
            job["step"] = f"Applied {len(dictionary)} dictionary corrections..."

        job["status"] = "diarizing"
        job["step"] = "Starting diarization subprocess..."
        _save_jobs_index()

        diarization_turns = await loop.run_in_executor(None, diarize_audio, wav_path, num_speakers, job, adv)

        job["status"] = "merging"
        job["step"] = "Assigning speakers to transcript..."
        labeled_words = assign_speakers_to_words(segments, diarization_turns)
        labeled_words = _merge_short_turns(
            labeled_words,
            min_turn_duration=adv.get("min_turn_duration", 1.5),
            min_word_count=adv.get("min_word_count", 4),
            max_speakers=num_speakers,
        )
        labeled_words = absorb_backchannels(labeled_words, max_words=2, max_duration=1.0)

        speaker_map = {}
        speaker_counter = 0
        for w in labeled_words:
            if w["speaker"] not in speaker_map:
                speaker_map[w["speaker"]] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            w["speaker"] = speaker_map[w["speaker"]]

        job["num_speakers"] = len(speaker_map)
        job["status"] = "building_outputs"
        job["step"] = "Generating transcript and subtitle files..."

        txt_path = job_output_dir / f"{stem}.txt"
        srt_path = job_output_dir / f"{stem}.srt"
        json_path = job_output_dir / f"{stem}_words.json"
        txt_path.write_text(build_diarized_transcript(labeled_words), encoding="utf-8")
        srt_path.write_text(build_srt(labeled_words), encoding="utf-8")
        json_path.write_text(json.dumps({"info": info, "speakers": list(speaker_map.values()), "words": labeled_words}, indent=2), encoding="utf-8")

        job["files"] = {
            "txt": _to_rel_path(txt_path),
            "srt": _to_rel_path(srt_path),
            "json": _to_rel_path(json_path),
            "audio": _to_rel_path(audio_path),
        }
        job["audio_path"] = _to_rel_path(audio_path)
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(BASE_DIR / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/status")
async def system_status():
    gpu_available = False
    gpu_name = None
    gpu_vram_total_mb = 0
    gpu_vram_used_mb = 0
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.decode().strip().split(",")
            if len(parts) >= 3:
                gpu_available = True
                gpu_name = parts[0].strip()
                gpu_vram_total_mb = int(parts[1].strip())
                gpu_vram_used_mb = int(parts[2].strip())
    except Exception:
        pass

    return {
        "gpu": gpu_available,
        "gpu_detail": {
            "name": gpu_name,
            "vram_total": f"{gpu_vram_total_mb / 1024:.1f} GB" if gpu_vram_total_mb else None,
            "vram_used": f"{gpu_vram_used_mb / 1024:.1f} GB" if gpu_vram_total_mb else None,
            "vram_pct": round(gpu_vram_used_mb / gpu_vram_total_mb * 100) if gpu_vram_total_mb else 0,
        },
        "whisper": _WHISPER_SCRIPT.exists() and _VENV_PYTHON.exists(),
        "diarization": bool(HF_TOKEN) and _WORKER_SCRIPT.exists() and _PYANNOTE_PYTHON.exists(),
    }


@app.get("/api/vram")
async def vram_status():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.decode().strip().split(",")
            if len(parts) >= 2:
                total_mb = int(parts[0].strip())
                used_mb = int(parts[1].strip())
                return {
                    "vram_total": f"{total_mb / 1024:.1f} GB",
                    "vram_used": f"{used_mb / 1024:.1f} GB",
                    "vram_pct": round(used_mb / total_mb * 100) if total_mb else 0,
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
    adv_beam_size: int = Form(1),
    adv_temperature: float = Form(0.0),
    adv_chunk_length: int = Form(20),
    adv_no_speech_threshold: float = Form(0.6),
    adv_hotwords: str = Form(""),
    adv_condition_on_prev: bool = Form(False),
    adv_diarization_preset: str = Form(""),
    adv_min_turn_duration: float = Form(1.2),
    adv_min_silence: float = Form(0.7),
    adv_min_cluster_size: int = Form(35),
    adv_min_word_count: int = Form(3),
):
    allowed_ext = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    job_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    group = re.sub(r"[/\\\\<>:\"|?*]", "_", group.strip() or DEFAULT_GROUP)
    (OUTPUT_DIR / group).mkdir(parents=True, exist_ok=True)

    preset_map = {
        "interview": {"min_turn": 1.5, "min_silence": 0.5, "min_cluster": 75, "min_words": 4},
        "focus_group": {"min_turn": 0.8, "min_silence": 0.3, "min_cluster": 50, "min_words": 3},
        "panel": {"min_turn": 0.5, "min_silence": 0.2, "min_cluster": 30, "min_words": 2},
        "monologue": {"min_turn": 2.0, "min_silence": 1.5, "min_cluster": 120, "min_words": 6},
    }
    if adv_diarization_preset in preset_map:
        p = preset_map[adv_diarization_preset]
        adv_min_turn_duration = p["min_turn"]
        adv_min_silence = p["min_silence"]
        adv_min_cluster_size = p["min_cluster"]
        adv_min_word_count = p.get("min_words", adv_min_word_count)

    jobs[job_id] = {
        "id": job_id,
        "filename": Path(file.filename).name,
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
        "audio_path": _to_rel_path(upload_path),
        "adv": {
            "beam_size": adv_beam_size,
            "temperature": adv_temperature,
            "chunk_length": adv_chunk_length,
            "no_speech_threshold": adv_no_speech_threshold,
            "hotwords": adv_hotwords.strip(),
            "condition_on_prev": adv_condition_on_prev,
            "min_turn_duration": adv_min_turn_duration,
            "min_silence": adv_min_silence,
            "min_cluster_size": adv_min_cluster_size,
            "min_word_count": adv_min_word_count,
            "diarization_preset": adv_diarization_preset,
        },
    }

    _save_jobs_index()
    _save_job_json(job_id)

    _queue_order.append(job_id)
    queue_pos = len(_queue_order)
    if queue_pos > 1:
        jobs[job_id]["step"] = f"Queued — position {queue_pos} of {queue_pos}"

    print(f"[INFO] Job {job_id} enqueued (pos {queue_pos}): model={whisper_model!r} language={language!r}")
    _job_queue.put_nowait((job_id, _to_rel_path(upload_path), language, num_speakers, whisper_model.strip()))
    _update_queue_steps()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] not in ("queued", "transcribing", "diarizing", "merging", "building_outputs"):
        raise HTTPException(400, "Job is not running")

    killed = []
    for pid in list(_active_pids.get(job_id, [])):
        if _terminate_pid(pid):
            killed.append(pid)

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
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs.pop(job_id)
    _live_segments.pop(job_id, None)
    _active_pids.pop(job_id, None)
    job_dir = OUTPUT_DIR / job.get("group", DEFAULT_GROUP) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    _save_jobs_index()
    return {"ok": True, "deleted": job_id}


@app.get("/api/jobs")
async def list_jobs():
    return list(jobs.values())


@app.get("/api/live/{job_id}")
async def live_transcript(job_id: str):
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
                text = seg.get("text")
                start = seg.get("start")
                end = seg.get("end")
                if text is not None:
                    payload = {"index": sent_count, "start": start, "end": end, "text": text}
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if file_type not in job.get("files", {}):
        raise HTTPException(404, f"No {file_type} file for this job")
    file_path = _resolve_job_path(job["files"][file_type])
    if not file_path or not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, filename=file_path.name, media_type="application/octet-stream")


@app.get("/api/preview/{job_id}/{file_type}")
async def preview_file(job_id: str, file_type: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if file_type not in job.get("files", {}):
        raise HTTPException(404, f"No {file_type} file for this job")
    file_path = _resolve_job_path(job["files"][file_type])
    if not file_path or not file_path.exists():
        raise HTTPException(404, "File not found")
    return {"content": file_path.read_text(encoding="utf-8"), "filename": file_path.name}


@app.get("/api/groups")
async def list_groups():
    return _get_all_groups()


@app.post("/api/groups")
async def create_group(name: str = Form(...)):
    name = re.sub(r"[/\\\\<>:\"|?*]", "_", name.strip())
    if not name:
        raise HTTPException(400, "Group name cannot be empty")
    (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
    return _get_all_groups()


@app.post("/api/jobs/{job_id}/move")
async def move_job(job_id: str, group: str = Form(...)):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    group = re.sub(r"[/\\\\<>:\"|?*]", "_", group.strip())
    if not group:
        raise HTTPException(400, "Group name cannot be empty")

    old_group = job.get("group", DEFAULT_GROUP)
    if old_group == group:
        return {"ok": True, "group": group}

    old_dir = OUTPUT_DIR / old_group / job_id
    new_dir = OUTPUT_DIR / group / job_id
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    if old_dir.exists():
        shutil.move(str(old_dir), str(new_dir))

    job["group"] = group
    old_base = OUTPUT_DIR / old_group / job_id
    new_files = {}
    for ftype, fpath in job.get("files", {}).items():
        resolved = _resolve_job_path(fpath)
        if not resolved:
            continue
        try:
            rel = resolved.relative_to(old_base)
            new_files[ftype] = _to_rel_path(new_dir / rel)
        except ValueError:
            new_files[ftype] = _to_rel_path(resolved)
    job["files"] = new_files

    _save_job_json(job_id)
    _save_jobs_index()
    return {"ok": True, "group": group}


@app.get("/editor/{job_id}", response_class=HTMLResponse)
async def transcript_editor(request: Request, job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse(request, "editor.html", {"job_id": job_id, "job": jobs[job_id]})


@app.get("/api/audio/{job_id}")
async def stream_audio(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    resolved = _resolve_job_path(job.get("audio_path") or job.get("files", {}).get("audio"))
    if not resolved or not resolved.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(resolved, media_type=_guess_media_type(resolved), filename=resolved.name)


@app.get("/api/words/{job_id}")
async def get_words(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    words_path = _resolve_job_path(jobs[job_id].get("files", {}).get("json"))
    if not words_path or not words_path.exists():
        raise HTTPException(404, "Word data not found")
    return JSONResponse(content=json.loads(words_path.read_text(encoding="utf-8")))


@app.post("/api/transcript/{job_id}")
async def save_transcript(job_id: str, content: str = Form(...)):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    txt_path = _resolve_job_path(jobs[job_id].get("files", {}).get("txt"))
    if not txt_path:
        raise HTTPException(400, "No transcript file for this job")
    txt_path.write_text(content, encoding="utf-8")
    return {"ok": True}


@app.get("/api/dictionary")
async def get_dictionary(group: str = "", scope: str = "group"):
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
    try:
        data = json.loads(entries)
    except Exception:
        raise HTTPException(400, "Invalid JSON for entries")
    path = _get_dictionary_path(global_dict=True) if scope == "global" or not group else _get_dictionary_path(group=group)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": True, "saved": len(data), "scope": scope or "group"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8765)),
        reload=False,
        workers=1,
    )
