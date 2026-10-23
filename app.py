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

import compute_devices
import compute_settings
import worker_runtime

warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*")

_cpu_cores = multiprocessing.cpu_count()

BASE_DIR = Path(__file__).resolve().parent
# Storage locations can be redirected via environment variables (e.g. for tests
# or alternative deployments). UPLOADS_DIR/OUTPUTS_DIR mirror the names used
# by mcp_proxy.py so one environment configures both processes.
UPLOAD_DIR = Path(os.environ.get("UPLOADS_DIR") or (BASE_DIR / "uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUTS_DIR") or (BASE_DIR / "outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS_INDEX_PATH = OUTPUT_DIR / "jobs.json"
DEFAULT_GROUP = "Ungrouped"
IS_WINDOWS = os.name == "nt"

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3-turbo").strip() or "large-v3-turbo"

# Compute settings store — persistent processing-device selection (see
# compute_settings.py). Loaded lazily; a missing file means "unconfigured".
_compute_store = compute_settings.ComputeSettingsStore()

# Serializes ALL GPU-allocating work: pipeline jobs (via the job queue) and
# compatibility probes share this lock so a probe can never overlap a job's
# model allocations.
_gpu_work_lock = threading.Lock()


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


def _has_gpu() -> bool:
    try:
        return bool(compute_devices.list_cuda_devices())
    except compute_devices.DeviceDiscoveryError:
        return False


def _worker_env_for(role: str, compute: dict | None) -> tuple[Path, dict]:
    """Resolve the interpreter + environment for a worker launch.

    compute is the job's immutable snapshot (job["compute"]); when absent
    (legacy jobs) it is resolved from current settings. Never reads current
    settings when a snapshot exists — changing defaults must not affect
    already-accepted jobs.
    """
    snapshot = compute or _snapshot_from_settings()
    stage = "transcription" if role == worker_runtime.ROLE_WHISPER else "diarization"
    device = {"backend": snapshot[stage]["backend"], "id": snapshot[stage].get("device_id")}
    py_exe = worker_runtime.resolve_worker_python(role)
    return py_exe, worker_runtime.build_worker_env(py_exe, device)


def _snapshot_from_settings() -> dict:
    """Resolve a compute snapshot from current saved settings + hardware."""
    try:
        settings = _compute_store.load()
        return compute_settings.build_job_snapshot(settings, compute_devices.available_cuda_ids())
    except compute_devices.DeviceDiscoveryError as e:
        raise RuntimeError(f"DEVICE_DISCOVERY_FAILED: {e.message}") from e
    except compute_settings.ComputeSettingsError as e:
        raise RuntimeError(f"{e.code}: {e.message}") from e


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


_WHISPER_SCRIPT = BASE_DIR / "whisper_worker.py"
_WORKER_SCRIPT = BASE_DIR / "diarize_worker.py"


def _worker_python_or_none(role: str) -> Path | None:
    try:
        return worker_runtime.resolve_worker_python(role)
    except worker_runtime.WorkerRuntimeError:
        return None

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
                     model_size: Optional[str] = None, adv: dict = None,
                     compute: Optional[dict] = None) -> tuple[list, dict, dict]:
    import tempfile

    py_exe, env = _worker_env_for(worker_runtime.ROLE_WHISPER, compute)
    python_bin = str(py_exe)

    snapshot = compute or _snapshot_from_settings()
    device = snapshot["transcription"]["backend"]
    requested_compute_type = snapshot["transcription"].get("requested_compute_type", "auto")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        args_file = f.name
        _adv = adv or {}
        json.dump(
            {
                "audio_path": audio_path,
                "language": language,
                "model_size": model_size or WHISPER_MODEL_SIZE,
                "device": device,
                "compute_type": requested_compute_type,
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
            env=env,
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
            raise RuntimeError(f"WORKER_CRASHED: Whisper worker exited with code {proc.returncode} without a result")

        with open(result_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        if result.get("status") == "error":
            code = result.get("code") or "WHISPER_FAILED"
            raise RuntimeError(f"{code}: Whisper worker failed: {result.get('payload')}")

        payload = result.get("payload", {})
        return payload.get("segments", []), payload.get("info", {}), result.get("execution", {})

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


def diarize_audio(audio_path: str, num_speakers: Optional[int] = None, job: Optional[dict] = None,
                  adv: dict = None, compute: Optional[dict] = None) -> tuple[list, dict]:
    import tempfile

    audio_duration = job.get("duration", 0) if job else 0
    py_exe, env = _worker_env_for(worker_runtime.ROLE_PYANNOTE, compute)
    python_bin = str(py_exe)

    snapshot = compute or _snapshot_from_settings()
    device = snapshot["diarization"]["backend"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        args_file = f.name
        _adv = adv or {}
        json.dump(
            {
                "audio_path": audio_path,
                "hf_token": HF_TOKEN,
                "num_speakers": num_speakers,
                "device": device,
                # Harmonized with the API form defaults (see /api/transcribe) so
                # every code path uses the same, looser diarization params.
                "min_silence": _adv.get("min_silence", 0.7),
                "min_cluster_size": _adv.get("min_cluster_size", 35),
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
                env=env,
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
                    f"WORKER_CRASHED: Diarization worker exited with code {proc.returncode} without a result. "
                    f"Last worker output: {(proc.stderr.read().decode(errors='replace')[-500:] if proc.stderr else 'n/a')}"
                )

            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("status") == "error":
                code = result.get("code") or "DIARIZATION_FAILED"
                raise RuntimeError(f"{code}: Diarization failed:\n{result.get('payload')}")
            return result.get("payload", []), result.get("execution", {})

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


def _best_speaker_for_span(start: float, end: float, diarization_turns: list) -> str:
    """Return the speaker whose diarization turn(s) cover the most of [start, end].

    Uses total temporal overlap (summed across turns per speaker) rather than a
    single midpoint test. This is robust for both short word spans and long
    segment spans, and never depends on a single instant near a turn boundary.
    """
    if end < start:
        start, end = end, start
    overlap_by_speaker: dict = {}
    for turn in diarization_turns:
        ov = min(end, turn["end"]) - max(start, turn["start"])
        if ov > 0:
            overlap_by_speaker[turn["speaker"]] = overlap_by_speaker.get(turn["speaker"], 0.0) + ov
    if overlap_by_speaker:
        return max(overlap_by_speaker, key=overlap_by_speaker.get)

    # No overlap (word/segment falls in a diarization gap): snap to nearest turn.
    mid = (start + end) / 2
    best_speaker = "UNKNOWN"
    best_dist = None
    for turn in diarization_turns:
        if turn["start"] <= mid <= turn["end"]:
            dist = 0.0
        elif mid < turn["start"]:
            dist = turn["start"] - mid
        else:
            dist = mid - turn["end"]
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_speaker = turn["speaker"]
    return best_speaker


def assign_speakers_to_words(segments: list, diarization_turns: list) -> list:
    """Assign a speaker to every word.

    For space-delimited languages we keep per-word assignment (each Whisper word
    is a real word). For Thai/CJK, faster-whisper emits one "word" per character
    or tone-mark, so per-word assignment shatters a single utterance across
    speakers at every diarization boundary. For those scripts we instead pick ONE
    speaker for the whole Whisper segment (a coherent phrase/sentence) by maximum
    total overlap, and every word in the segment inherits it. Output shape is
    unchanged (a flat word list), so all downstream builders keep working.
    """
    words = []
    for seg in segments:
        seg_words = []
        for w in seg.get("words", []):
            if "text" not in w or "start" not in w or "end" not in w:
                continue
            seg_words.append({"text": w["text"], "start": w["start"], "end": w["end"], "speaker": None})
        if not seg_words:
            continue

        # Decide granularity per segment based on its script.
        seg_text = seg.get("text", "") or "".join(sw["text"] for sw in seg_words)
        if _is_cjk_or_thai(seg_text):
            seg_start = min(sw["start"] for sw in seg_words)
            seg_end = max(sw["end"] for sw in seg_words)
            seg_speaker = _best_speaker_for_span(seg_start, seg_end, diarization_turns)
            for sw in seg_words:
                sw["speaker"] = seg_speaker
        else:
            for sw in seg_words:
                sw["speaker"] = _best_speaker_for_span(sw["start"], sw["end"], diarization_turns)
        words.extend(seg_words)

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
                       min_word_count: int = 4, max_speakers: Optional[int] = None,
                       min_cjk_turn_chars: int = 6) -> list:
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
            # Script-aware short-turn test. For space-delimited text a "turn" is
            # short when it is both brief AND has few words. For Thai/CJK,
            # faster-whisper words are single characters, so word count is a poor
            # proxy; use a character-count gate instead (OR'd with duration) so
            # residual micro-turns from per-character timing still coalesce.
            turn_text = "".join(w.get("text", "") for w in wds)
            if _is_cjk_or_thai(turn_text):
                char_count = sum(len(w.get("text", "").strip()) for w in wds)
                is_short = (char_count < min_cjk_turn_chars) or (duration < min_turn_duration)
            else:
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

    # Hold the GPU-work lock for the entire pipeline so compatibility probes
    # can never overlap this job's model allocations. Acquired off the event
    # loop thread (a blocking acquire here would deadlock the API while a
    # probe or test holds the lock).
    await asyncio.to_thread(_gpu_work_lock.acquire)
    try:
        # Revalidate the compute snapshot at execution start — AFTER the lock,
        # so hardware changes made while the job waited are caught. Never
        # rewrite the snapshot to whatever the current default is.
        compute = job.get("compute")
        if compute is not None:
            try:
                available = compute_devices.available_cuda_ids()
            except compute_devices.DeviceDiscoveryError as e:
                job["status"] = "error"
                job["step"] = f"Error: device discovery failed"
                job["error"] = f"DEVICE_DISCOVERY_FAILED: {e.message}"
                _save_job_json(job_id)
                _save_jobs_index()
                return
            for stage in ("transcription", "diarization"):
                ref = compute.get(stage, {})
                if ref.get("backend") == "cuda" and ref.get("device_id") not in available:
                    job["status"] = "error"
                    job["step"] = "Error: selected GPU unavailable"
                    job["error"] = (
                        f"DEVICE_NOT_AVAILABLE: {stage} GPU {ref.get('device_id')} disappeared before the job started. "
                        "No fallback device was used."
                    )
                    _save_job_json(job_id)
                    _save_jobs_index()
                    return
        await _process_job_inner(job_id, audio_path, language, num_speakers, compute)
    finally:
        await asyncio.to_thread(_gpu_work_lock.release)


def _run_compute_test_with_lock(test_id: str, device_ref: dict, include_smoke: bool) -> dict:
    with _gpu_work_lock:
        return _run_compute_test_blocking(test_id, device_ref, include_smoke)


async def _process_job_inner(job_id: str, audio_path: str, language: str, num_speakers: Optional[int], compute: Optional[dict]):
    job = jobs[job_id]

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
        segments, info, transcription_execution = await loop.run_in_executor(
            None, transcribe_audio, audio_path, _lang, job_id, _model, adv, compute
        )
        job["duration"] = info.get("duration")
        job["language"] = info.get("language")
        job["execution"] = {"transcription": transcription_execution}
        _save_jobs_index()

        # Cancellation during transcription: the worker was killed and the
        # exception path already marked the job — do not start diarization.
        if job.get("status") not in ("transcribing", "diarizing"):
            return

        dictionary = _load_dictionary(group)
        if dictionary:
            for seg in segments:
                seg["words"] = _apply_dictionary(seg.get("words", []), dictionary)
            job["step"] = f"Applied {len(dictionary)} dictionary corrections..."

        job["status"] = "diarizing"
        job["step"] = "Starting diarization subprocess..."
        _save_jobs_index()

        diarization_turns, diarization_execution = await loop.run_in_executor(
            None, diarize_audio, wav_path, num_speakers, job, adv, compute
        )
        job.setdefault("execution", {})["diarization"] = diarization_execution

        job["status"] = "merging"
        job["step"] = "Assigning speakers to transcript..."
        labeled_words = assign_speakers_to_words(segments, diarization_turns)
        labeled_words = _merge_short_turns(
            labeled_words,
            min_turn_duration=adv.get("min_turn_duration", 1.5),
            min_word_count=adv.get("min_word_count", 4),
            max_speakers=num_speakers,
            min_cjk_turn_chars=adv.get("min_cjk_turn_chars", 6),
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
    try:
        cuda_devices = compute_devices.list_cuda_devices()
    except compute_devices.DeviceDiscoveryError:
        cuda_devices = []

    # Legacy fields keep their meaning: the first GPU. Per-device data and the
    # selected device are additive.
    first = cuda_devices[0] if cuda_devices else None
    selected_id = None
    try:
        settings = _compute_store.load()
        resolved = compute_settings.resolve_settings_device(
            settings, {d.id for d in cuda_devices}
        )
        selected_id = resolved.get("id") if resolved.get("backend") == "cuda" else "cpu"
    except compute_settings.ComputeSettingsError:
        selected_id = None

    return {
        "gpu": bool(cuda_devices),
        "gpu_detail": {
            "name": first.name if first else None,
            "id": first.id if first else None,
            "vram_total": f"{first.memory_total_mib / 1024:.1f} GB" if first and first.memory_total_mib else None,
            "vram_used": f"{first.memory_used_mib / 1024:.1f} GB" if first and first.memory_total_mib else None,
            "vram_pct": round(first.memory_used_mib / first.memory_total_mib * 100) if first and first.memory_total_mib else 0,
        },
        "devices": [d.to_api() for d in cuda_devices],
        "selected_device": selected_id,
        "whisper": _WHISPER_SCRIPT.exists() and _worker_python_or_none(worker_runtime.ROLE_WHISPER) is not None,
        "diarization": bool(HF_TOKEN) and _WORKER_SCRIPT.exists() and _worker_python_or_none(worker_runtime.ROLE_PYANNOTE) is not None,
    }


@app.get("/api/vram")
async def vram_status():
    try:
        cuda_devices = compute_devices.list_cuda_devices()
    except compute_devices.DeviceDiscoveryError:
        cuda_devices = []

    selected_id = None
    selected_backend = None
    try:
        settings = _compute_store.load()
        resolved = compute_settings.resolve_settings_device(
            settings, {d.id for d in cuda_devices}
        )
        selected_backend = resolved.get("backend")
        if selected_backend == "cuda":
            selected_id = resolved["id"]
    except compute_settings.ComputeSettingsError:
        pass

    # Legacy keys describe the SELECTED device only. When resolution fails
    # (unconfigured multi-GPU, or a configured GPU that is missing) there is
    # no selected device: the inventory is still returned, but no device is
    # misattributed — no first-GPU fallback. A valid CPU selection is
    # reported via selected_backend (no GPU VRAM to show).
    shown = next((d for d in cuda_devices if d.id == selected_id), None)
    if shown and shown.memory_total_mib:
        return {
            "vram_total": f"{shown.memory_total_mib / 1024:.1f} GB",
            "vram_used": f"{shown.memory_used_mib / 1024:.1f} GB",
            "vram_pct": round(shown.memory_used_mib / shown.memory_total_mib * 100) if shown.memory_total_mib else 0,
            "selected_device": selected_id,
            "selected_backend": selected_backend,
            "devices": [d.to_api() for d in cuda_devices],
        }
    return {
        "vram_total": None,
        "vram_used": None,
        "vram_pct": 0,
        "selected_device": selected_id,
        "selected_backend": selected_backend,
        "devices": [d.to_api() for d in cuda_devices],
    }


@app.get("/api/compute/devices")
async def compute_devices_endpoint():
    """Allocation-free CPU/CUDA inventory for the settings UI."""
    try:
        return compute_devices.device_inventory()
    except compute_devices.DeviceDiscoveryError as e:
        raise HTTPException(500, f"{e.message}: {e.detail}")


@app.get("/api/compute/settings")
async def get_compute_settings():
    try:
        settings = _compute_store.load()
    except compute_settings.SettingsFileError as e:
        return JSONResponse(
            status_code=409,
            content={"error": e.message, "code": e.code, "recoverable_by_save": True},
        )
    try:
        cuda_devices = compute_devices.list_cuda_devices()
    except compute_devices.DeviceDiscoveryError as e:
        cuda_devices = []
    resolved = None
    resolution_error = None
    try:
        resolved = compute_settings.resolve_settings_device(settings, {d.id for d in cuda_devices})
    except compute_settings.ComputeSettingsError as e:
        resolution_error = {"code": e.code, "message": e.message}
    device_map = {d.id: d for d in cuda_devices}
    selected_name = None
    if resolved and resolved.get("backend") == "cuda":
        selected_name = device_map[resolved["id"]].name if resolved["id"] in device_map else "(missing)"
    return {
        "settings": settings,
        "resolved": resolved,
        "resolved_device_name": selected_name,
        "resolution_error": resolution_error,
        "needs_selection": resolution_error is not None and resolution_error["code"] == "DEVICE_NOT_CONFIGURED",
    }


@app.put("/api/compute/settings")
async def put_compute_settings(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be JSON")
    try:
        saved = _compute_store.save(body)
    except compute_settings.ComputeSettingsError as e:
        raise HTTPException(400, f"{e.code}: {e.message}")
    return {"ok": True, "settings": saved}


# --- Compatibility tests (coordinated with the job pipeline) -------------------

_compute_tests: dict = {}
_compute_tests_order: list = []
_PROBE_TIMEOUT_S = 900


def _launch_probe(py_exe: Path, env: dict, probe_args: dict) -> dict:
    """Run one probe subprocess through the same launch path as production."""
    import tempfile

    probe_script = BASE_DIR / "scripts" / "probe_compute.py"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        args_file = f.name
        json.dump(probe_args, f)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        result_file = f.name
    try:
        proc = _spawn_worker([str(py_exe), str(probe_script), args_file, result_file], env=env)
        try:
            proc.wait(timeout=_PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
            return {"status": "error", "code": "PROBE_TIMEOUT", "message": f"probe exceeded {_PROBE_TIMEOUT_S}s"}
        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            return {
                "status": "error",
                "code": "WORKER_CRASHED",
                "message": f"probe exited with code {proc.returncode} without a result",
            }
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        for p in (args_file, result_file):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def _run_compute_test_blocking(test_id: str, device_ref: dict, include_smoke: bool) -> dict:
    """Sequential probe suite for one device. Must run under _gpu_work_lock."""
    results: dict = {"device": device_ref, "started_at": datetime.now().isoformat()}
    for role in (worker_runtime.ROLE_WHISPER, worker_runtime.ROLE_PYANNOTE):
        try:
            py_exe = worker_runtime.resolve_worker_python(role)
            env = worker_runtime.build_worker_env(py_exe, device_ref)
        except worker_runtime.WorkerRuntimeError as e:
            results[role] = {"status": "error", "code": e.code, "message": e.message}
            continue
        results[role] = {}
        results[role]["inventory"] = _launch_probe(py_exe, env, {"role": role, "device": device_ref, "mode": "inventory"})
        results[role]["torch"] = _launch_probe(py_exe, env, {"role": role, "device": device_ref, "mode": "torch"})
        if include_smoke:
            mode = "whisper_smoke" if role == worker_runtime.ROLE_WHISPER else "pyannote_smoke"
            smoke_args = {"role": role, "device": device_ref, "mode": mode}
            if role == worker_runtime.ROLE_PYANNOTE:
                smoke_args["hf_token"] = HF_TOKEN
            results[role]["smoke"] = _launch_probe(py_exe, env, smoke_args)
    results["completed_at"] = datetime.now().isoformat()

    def _ok(step):
        return isinstance(step, dict) and step.get("status") == "ok"

    results["summary"] = {
        "whisper_runtime": _ok(results.get(worker_runtime.ROLE_WHISPER, {}).get("torch")) if isinstance(results.get(worker_runtime.ROLE_WHISPER), dict) else False,
        "pyannote_runtime": _ok(results.get(worker_runtime.ROLE_PYANNOTE, {}).get("torch")) if isinstance(results.get(worker_runtime.ROLE_PYANNOTE), dict) else False,
    }
    return results


@app.post("/api/compute/tests")
async def run_compute_test(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be JSON")
    try:
        device_ref = compute_settings.parse_device_override(body.get("device", ""), field="device")
    except compute_settings.ComputeSettingsError as e:
        raise HTTPException(400, f"{e.code}: {e.message}")
    if device_ref is None:
        # No explicit device: probe the currently-selected device.
        try:
            settings = _compute_store.load()
            device_ref = compute_settings.resolve_settings_device(
                settings, compute_devices.available_cuda_ids()
            )
        except (compute_settings.ComputeSettingsError, compute_devices.DeviceDiscoveryError) as e:
            raise HTTPException(400, f"{getattr(e, 'code', 'DEVICE_NOT_CONFIGURED')}: {getattr(e, 'message', e)}")

    include_smoke = bool(body.get("include_smoke", False))
    test_id = str(uuid.uuid4())[:8]
    _compute_tests[test_id] = {
        "id": test_id,
        "status": "running",
        "device": device_ref,
        "include_smoke": include_smoke,
        "started_at": datetime.now().isoformat(),
    }
    while len(_compute_tests_order) >= 20:
        _compute_tests.pop(_compute_tests_order.pop(0), None)
    _compute_tests_order.append(test_id)

    async def _runner():
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, _run_compute_test_with_lock, test_id, device_ref, include_smoke
            )
            _compute_tests[test_id].update(results)
            _compute_tests[test_id]["status"] = "done"
            _persist_compute_report(test_id, _compute_tests[test_id])
        except Exception as e:
            _compute_tests[test_id]["status"] = "error"
            _compute_tests[test_id]["error"] = f"{type(e).__name__}: {e}"[:500]

    asyncio.create_task(_runner())
    return {"test_id": test_id, "status": "running"}


@app.get("/api/compute/tests/{test_id}")
async def get_compute_test(test_id: str):
    if test_id not in _compute_tests:
        raise HTTPException(404, "Test not found")
    return _compute_tests[test_id]


def _persist_compute_report(test_id: str, report: dict):
    """Persist the latest compatibility report next to the settings file."""
    try:
        path = _compute_store.path.parent / "compute-report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass  # report persistence is best-effort; the API response is authoritative


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
    transcription_device: str = Form(""),
    diarization_device: str = Form(""),
    whisper_compute_type: str = Form(""),
):
    allowed_ext = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Build the immutable compute snapshot at acceptance: saved defaults +
    # per-job overrides, resolved against currently available hardware.
    # Changing saved defaults later must never affect this job.
    try:
        settings = _compute_store.load()
        available = compute_devices.available_cuda_ids()
        compute_snapshot = compute_settings.build_job_snapshot(
            settings,
            available,
            transcription_override=compute_settings.parse_device_override(
                transcription_device, field="transcription_device"
            ),
            diarization_override=compute_settings.parse_device_override(
                diarization_device, allow_same=True, field="diarization_device"
            ),
            compute_type_override=(whisper_compute_type.strip() or None),
        )
    except compute_settings.ComputeSettingsError as e:
        raise HTTPException(400, f"{e.code}: {e.message}")
    except compute_devices.DeviceDiscoveryError as e:
        raise HTTPException(503, f"DEVICE_DISCOVERY_FAILED: {e.message}")

    job_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    group = re.sub(r"[/\\\\<>:\"|?*]", "_", group.strip() or DEFAULT_GROUP)
    (OUTPUT_DIR / group).mkdir(parents=True, exist_ok=True)

    preset_map = {
        "interview": {"min_turn": 1.2, "min_silence": 0.7, "min_cluster": 35, "min_words": 3},
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
        "compute": compute_snapshot,
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


@app.post("/api/jobs/{job_id}/rename")
async def rename_job(job_id: str, filename: str = Form(...)):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    filename = filename.strip()
    if not filename:
        raise HTTPException(400, "Filename cannot be empty")
    job = jobs[job_id]
    job["filename"] = filename
    _save_job_json(job_id)
    _save_jobs_index()
    return {"ok": True, "filename": filename}


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


def _load_words_payload(job_id: str) -> tuple[Path, dict]:
    """Load the {job_id}_words.json payload for a job. Raises HTTPException on failure."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    words_path = _resolve_job_path(jobs[job_id].get("files", {}).get("json"))
    if not words_path or not words_path.exists():
        raise HTTPException(404, "Word data not found")
    try:
        payload = json.loads(words_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(500, "Word data is corrupt")
    if not isinstance(payload.get("words"), list):
        raise HTTPException(500, "Word data missing 'words' list")
    return words_path, payload


def _rewrite_job_outputs(job_id: str, words_path: Path, payload: dict):
    """Persist an updated words payload back to disk and regenerate .txt/.srt from it.

    This is the single source of truth rewrite used by speaker rename/merge so that
    every future .txt/.srt export reflects the edited speaker labels permanently.
    """
    words = payload["words"]
    speakers = sorted({w.get("speaker") for w in words if w.get("speaker")})
    payload["speakers"] = speakers
    words_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    job = jobs[job_id]
    files = job.get("files", {})

    txt_path = _resolve_job_path(files.get("txt"))
    if txt_path:
        txt_path.write_text(build_diarized_transcript(words), encoding="utf-8")

    srt_path = _resolve_job_path(files.get("srt"))
    if srt_path:
        srt_path.write_text(build_srt(words), encoding="utf-8")

    job["num_speakers"] = len(speakers)
    _save_job_json(job_id)
    _save_jobs_index()


@app.post("/api/jobs/{job_id}/speakers/rename")
async def rename_speaker(job_id: str, old_name: str = Form(...), new_name: str = Form(...)):
    old_name = old_name.strip()
    new_name = new_name.strip()
    if not old_name or not new_name:
        raise HTTPException(400, "Speaker names cannot be empty")

    words_path, payload = _load_words_payload(job_id)
    words = payload["words"]

    if old_name == new_name:
        return {"ok": True, "speakers": sorted({w.get("speaker") for w in words if w.get("speaker")})}

    matched = False
    for w in words:
        if w.get("speaker") == old_name:
            w["speaker"] = new_name
            matched = True
    if not matched:
        raise HTTPException(404, f"Speaker '{old_name}' not found in this job")

    _rewrite_job_outputs(job_id, words_path, payload)
    return {"ok": True, "speakers": payload["speakers"]}


@app.post("/api/jobs/{job_id}/speakers/merge")
async def merge_speakers(job_id: str, speakers: str = Form(...), target_name: str = Form(...)):
    try:
        speaker_list = json.loads(speakers)
    except Exception:
        raise HTTPException(400, "'speakers' must be a JSON array of speaker names")
    if not isinstance(speaker_list, list) or len(speaker_list) < 2:
        raise HTTPException(400, "Provide at least 2 speaker names to merge")
    speaker_list = [s.strip() for s in speaker_list if isinstance(s, str) and s.strip()]
    target_name = target_name.strip()
    if not target_name:
        raise HTTPException(400, "Target speaker name cannot be empty")

    words_path, payload = _load_words_payload(job_id)
    words = payload["words"]

    source_set = set(speaker_list)
    matched = False
    for w in words:
        if w.get("speaker") in source_set:
            w["speaker"] = target_name
            matched = True
    if not matched:
        raise HTTPException(404, "None of the given speakers were found in this job")

    _rewrite_job_outputs(job_id, words_path, payload)
    return {"ok": True, "speakers": payload["speakers"]}


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
