"""
whisper-mcp proxy — exposes the Interview Transcriber app (FastAPI on :8765)
to Gemma / Open WebUI as an MCP (Streamable HTTP) tool server.

This is a thin proxy: it owns no models and does no GPU work. It discovers
audio files in the app's uploads/ dir, submits transcription jobs through the
app's existing /api/transcribe endpoint (so outputs land in
outputs/<group>/<job_id>/ exactly as the web UI produces them), polls job
status, and returns transcript content for completed jobs.

Transport: Streamable HTTP at /mcp on port 8002 (matches the SearXNG MCP
pattern already used in the gemma-stack Open WebUI setup). Note: with mcp >= 1.28
the endpoint is /mcp (NO trailing slash) — /mcp/ 307-redirects to /mcp.

Run:
    ./.venv/bin/pip install -r requirements-mcp.txt
    ./.venv/bin/python mcp_proxy.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# --- Configuration (env-overridable) -----------------------------------------
TRANSCRIBER_URL = os.getenv("TRANSCRIBER_URL", "http://localhost:8765").rstrip("/")
# Repo root of the interview-transcriber app (where uploads/ and outputs/ live).
TRANSCRIBER_DIR = Path(os.getenv("TRANSCRIBER_DIR", Path(__file__).resolve().parent))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", TRANSCRIBER_DIR / "uploads"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", TRANSCRIBER_DIR / "outputs"))
JOBS_INDEX = OUTPUTS_DIR / "jobs.json"

PORT = int(os.getenv("PORT", "8002"))
HOST = os.getenv("HOST", "0.0.0.0")
# Warn (non-blocking) when free VRAM drops below this many MiB.
VRAM_WARN_MB = int(os.getenv("VRAM_WARN_MB", "2048"))
# Truncate transcript text returned in get_job() to keep tool responses small.
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "60000"))

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm", ".aac"}

# Host header values the Streamable HTTP transport will accept (DNS-rebinding
# protection rejects anything not listed; no "*" wildcard, only exact matches
# and "host:*" port patterns). Defaults cover the Docker service name used by
# open-webui, host.docker.internal, localhost, and the storm1 tailscale IP.
# Override via comma-separated MCP_ALLOWED_HOSTS.
_DEFAULT_ALLOWED_HOSTS = [
    "whisper-mcp:*",
    "host.docker.internal:*",
    "localhost:*",
    "127.0.0.1:*",
    "[::1]:*",
    "100.98.84.83:*",
]
_allowed_hosts = (
    [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    or _DEFAULT_ALLOWED_HOSTS
)
mcp = FastMCP(
    "interview-transcriber",
    transport_security=TransportSecuritySettings(allowed_hosts=_allowed_hosts),
)


# --- Whisper model catalogue (static; VRAM figures are faster-whisper int8 ---
# --- on CUDA, which app.py forces for cuda device) ---------------------------

WHISPER_MODELS: list[dict[str, Any]] = [
    {
        "model": "large-v3-turbo",
        "params": "809M",
        "vram_gb": "~1.5-2.5",
        "speed": "~8x realtime on RTX 3060",
        "accuracy": "Near large-v3; ~3-4% WER",
        "note": "App default. Best co-resident choice alongside Gemma (low VRAM, fast).",
    },
    {
        "model": "large-v3",
        "params": "1550M",
        "vram_gb": "~2.5-3.5",
        "speed": "~8x realtime on RTX 3060",
        "accuracy": "Best (~2.7% WER)",
        "note": "Max accuracy. Use when Gemma is idle or VRAM headroom is ample.",
    },
    {
        "model": "large-v2",
        "params": "1550M",
        "vram_gb": "~2.5-3.5",
        "speed": "~8x realtime on RTX 3060",
        "accuracy": "~2.4% WER (EN)",
        "note": "Legacy large model; rarely preferred over large-v3.",
    },
    {
        "model": "distil-large-v3",
        "params": "~756M",
        "vram_gb": "~1.5",
        "speed": "~6x faster than large-v3",
        "accuracy": "Within ~1% WER of large-v3",
        "note": "Distilled; great speed/VRAM, English-centric.",
    },
    {
        "model": "medium",
        "params": "769M",
        "vram_gb": "~2.0",
        "speed": "~2x realtime",
        "accuracy": "~3.5% WER (EN)",
        "note": "Light fallback if VRAM is very tight.",
    },
    {
        "model": "small",
        "params": "244M",
        "vram_gb": "~1.0",
        "speed": "~4x realtime",
        "accuracy": "Good for clean audio / drafts",
        "note": "Very light; useful when Gemma is fully loaded.",
    },
    # Thai-specific models referenced by app.py (process_job _th_models set)
    {
        "model": "nectec/Pathumma-whisper-th-large-v3",
        "params": "large",
        "vram_gb": "~3.0",
        "speed": "large-class",
        "accuracy": "Tuned for Thai",
        "note": "Thai language; app forces language='th' when auto-detected.",
    },
    {
        "model": "nectec/Pathumma-whisper-th-medium",
        "params": "medium",
        "vram_gb": "~2.0",
        "speed": "medium-class",
        "accuracy": "Tuned for Thai",
        "note": "Lighter Thai option.",
    },
    {
        "model": "Vinxscribe/biodatlab-whisper-th-large-v3-faster",
        "params": "large",
        "vram_gb": "~3.0",
        "speed": "large-class",
        "accuracy": "Tuned for Thai (CTranslate2)",
        "note": "Thai language; faster-whisper native.",
    },
]
_VALID_MODELS = {m["model"] for m in WHISPER_MODELS}


# --- Helpers -----------------------------------------------------------------

def _probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stderr=subprocess.DEVNULL, timeout=15,
        ).decode().strip()
        return float(out)
    except Exception:
        return None


def _load_jobs_index() -> list[dict]:
    try:
        if JOBS_INDEX.exists():
            return json.loads(JOBS_INDEX.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


# 8-hex-char hash prefix the transcriber app prepends to every upload (can
# stack, e.g. "e36f1af0_09ae599a_<orig>"). Strip ALL leading such prefixes so
# sibling copies and web-UI-submitted (un-prefixed) jobs match the same base.
# Only strip prefixes that contain a hex letter a-f — pure-digit 8-char prefixes
# are dates (e.g. 20260416_), not hashes.
_HEX_PREFIX_RE = re.compile(r"^([0-9a-f]{8})_")


def _normalize_name(name: str) -> str:
    name = os.path.basename(name)
    while True:
        m = _HEX_PREFIX_RE.match(name)
        if not m or not any(c in "abcdef" for c in m.group(1)):
            break
        name = name[m.end():]
    return name


def _completed_jobs_by_base() -> dict[str, list[dict]]:
    """Map normalized base filename -> [{id, group}, ...] for completed jobs."""
    out: dict[str, list[dict]] = {}
    for j in _load_jobs_index():
        if j.get("status") == "complete" and j.get("filename"):
            base = _normalize_name(j["filename"])
            jid = j.get("id") or j.get("job_id")
            if jid:
                out.setdefault(base, []).append({"id": jid, "group": j.get("group")})
    return out


def _list_uploads(group: str | None = None) -> list[dict]:
    """List uploaded audio files, optionally filtered to those with a completed
    job in `group` (case-insensitive exact match on the job's group name).
    """
    if not UPLOADS_DIR.exists():
        return []
    done = _completed_jobs_by_base()  # base filename -> [{id, group}, ...]
    group_norm = group.strip().lower() if group else None
    out = []
    for p in sorted(UPLOADS_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        entries = done.get(_normalize_name(p.name), [])
        groups = sorted({e["group"] for e in entries if e.get("group")})
        if group_norm is not None:
            entries = [e for e in entries if (e.get("group") or "").strip().lower() == group_norm]
            if not entries:
                continue  # file has no completed job in the requested group — skip
        job_ids = [e["id"] for e in entries]
        out.append({
            "name": p.name,
            "path": str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
            "duration_s": _probe_duration(p),
            "transcribed": bool(job_ids),
            "job_ids": job_ids,
            "groups": groups,
        })
    return out


def _list_groups() -> list[dict]:
    """Distinct groups across all jobs with counts, for discovery."""
    counts: dict[str, int] = {}
    for j in _load_jobs_index():
        g = (j.get("group") or "").strip()
        if g:
            counts[g] = counts.get(g, 0) + 1
    return [{"group": g, "job_count": n} for g, n in sorted(counts.items(), key=lambda kv: -kv[1])]


async def _get_vram() -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{TRANSCRIBER_URL}/api/vram")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


def _parse_free_vram_mb(vram: dict | None) -> int | None:
    if not vram:
        return None
    try:
        total = float(vram.get("vram_total", "").split()[0]) if vram.get("vram_total") else None
        used = float(vram.get("vram_used", "").split()[0]) if vram.get("vram_used") else None
        if total is not None and used is not None:
            return int((total - used) * 1024)
    except Exception:
        pass
    return None


async def _get_compute_settings() -> dict[str, Any] | None:
    """Fetch the app's saved compute settings (selected device etc.)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{TRANSCRIBER_URL}/api/compute/settings")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


async def _selected_device_summary() -> dict[str, Any] | None:
    """Best-effort summary of the currently selected processing device."""
    settings = await _get_compute_settings()
    if not settings or settings.get("resolution_error") or settings.get("needs_selection"):
        return None
    resolved = settings.get("resolved") or {}
    if not resolved:
        return None
    return {
        "backend": resolved.get("backend"),
        "id": resolved.get("id"),
        "name": settings.get("resolved_device_name"),
    }


# --- Tools -------------------------------------------------------------------

@mcp.tool()
async def list_whisper_models() -> dict:
    """List available Whisper transcription models with VRAM / speed / accuracy notes.

    VRAM notes are indicative; the actual device is whatever the app's saved
    compute settings select (call get_compute_selection() to see it). A small
    card (e.g. a 4 GB GTX 1050 Ti) may not fit large-class models — prefer
    large-v3-turbo / medium / small there.

    IMPORTANT for the assistant: before calling transcribe_audio(), you MUST ask
    the user which model to use. Present these options (call this tool to fetch
    them) and let the user choose. Default recommendation when unsure:
    'large-v3-turbo' (low VRAM, fast, near-large-v3 accuracy).
    """
    selection = await _selected_device_summary()
    if selection and selection.get("backend") == "cuda":
        coresident_note = (
            f"Transcription runs on the selected GPU ({selection.get('name') or selection.get('id')}). "
            "Its free VRAM is what matters for model fit. On a small card (e.g. 4 GB), "
            "prefer large-v3-turbo / medium / small. Note: the app serializes its own "
            "workers, but it cannot coordinate with other processes on that GPU — an "
            "LLM server on the same card can still cause out-of-memory errors."
        )
    elif selection and selection.get("backend") == "cpu":
        coresident_note = (
            "Transcription resolves to CPU (either explicitly saved or because no GPU "
            "is available) — models will run much slower than on a GPU. Pick a GPU in "
            "the app's Compute settings if speed matters."
        )
    else:
        coresident_note = (
            "No transcription device is configured yet (or the saved one is unavailable). "
            "Submissions without a device override will be rejected until a device is "
            "selected in the app's Compute settings."
        )
    return {
        "default": "large-v3-turbo",
        "coresident_note": coresident_note,
        "selected_device": selection,
        "models": WHISPER_MODELS,
    }


@mcp.tool()
def list_audio_files(group: str = "") -> dict:
    """List audio files available for transcription from the app's uploads/ dir.

    Returns name, path, size (MB), duration (s), the group(s) any completed
    transcript belongs to, and whether the file has already been transcribed.
    Drop new files into the app's uploads/ directory on the host to make them
    appear here.

    IMPORTANT for efficiency: if you're looking for a file within a known
    research group/project (e.g. "ARINT", "MCP"), pass `group` to narrow the
    list instead of listing every file and scanning titles yourself. Call
    list_groups() first if you don't know the exact group name. For finding
    a file by topic/content rather than filename, use search_transcripts()
    instead — it searches inside transcripts, not just filenames.

    Args:
        group: Optional. Only return files with a completed transcript in this
            group (case-insensitive exact match). Leave empty to list all files.

    Pass any 'path' (or 'name') to transcribe_audio().
    """
    files = _list_uploads(group=group or None)
    return {
        "uploads_dir": str(UPLOADS_DIR),
        "group_filter": group or None,
        "count": len(files),
        "files": files,
        "note": "Pass a 'path' value to transcribe_audio(). Files with "
                "'transcribed': true already have a completed transcript — use "
                "get_job(job_id) with one of that file's 'job_ids' to retrieve it "
                "instead of re-transcribing. Use list_groups() to see available "
                "groups, or search_transcripts() to find files by content.",
    }


@mcp.tool()
async def get_compute_selection() -> dict:
    """Get the transcriber's saved compute selection, plus the per-device VRAM inventory.

    'selected' resolves the TRANSCRIPTION role (the diarization device follows
    its own saved setting — returned raw under 'diarization_setting'; a
    'same_as_transcription' entry follows this job's transcription device).
    Returns {selected: {backend, id, name} | null, needs_selection, devices: [...]}.
    If needs_selection is true, no default device is configured — the user must
    pick one in the app's Compute settings (or pass a device override to
    transcribe_audio()) before jobs will be accepted.
    """
    settings = await _get_compute_settings()
    vram = await _get_vram()
    if not settings:
        return {"error": "Could not reach the transcriber app's compute settings API."}
    resolved = settings.get("resolved") or {}
    return {
        "selected": {
            "backend": resolved.get("backend"),
            "id": resolved.get("id"),
            "name": settings.get("resolved_device_name"),
        } if resolved else None,
        "needs_selection": bool(settings.get("needs_selection")),
        "resolution_error": settings.get("resolution_error"),
        "diarization_setting": ((settings.get("settings") or {}).get("diarization") or {}).get("device"),
        "devices": (vram or {}).get("devices", []),
    }


@mcp.tool()
def list_groups() -> dict:
    """List the distinct output groups used across all transcription jobs, with
    job counts for each. Use this to discover valid `group` values before
    calling list_audio_files(group=...) or list_jobs(group=...) /
    search_transcripts(group=...) to narrow results, instead of listing every
    file/job in the whole system.
    """
    groups = _list_groups()
    return {"count": len(groups), "groups": groups}


@mcp.tool()
async def transcribe_audio(
    audio_path: str,
    whisper_model: str,
    language: str = "auto",
    num_speakers: int = 0,
    group: str = "MCP",
    transcription_device: str = "",
    diarization_device: str = "",
    whisper_compute_type: str = "",
) -> dict:
    """Submit an audio file for transcription + diarization through the Interview Transcriber app.

    IMPORTANT for the assistant: you MUST confirm the whisper model choice with
    the user before calling this tool. Call list_whisper_models() to show options.
    If whisper_model is empty or not in the supported list, this tool returns the
    model list and asks you to re-call with a valid model.

    The job runs on the app's existing async queue (one GPU job at a time), so it
    returns immediately with a job_id. Poll get_job(job_id) until status == 'done'.

    Args:
        audio_path: Path to the audio file. Absolute, or a name resolved under the
            app's uploads/ dir (e.g. "20260401_ARINT003.mp3"). Use list_audio_files().
        whisper_model: REQUIRED. One of the models from list_whisper_models().
        language: ISO 639-1 code (e.g. "en", "th") or "auto" to auto-detect.
            Set explicitly for best non-English accuracy.
        num_speakers: Optional hint for diarization (exact speaker count).
        group: Output group name (creates outputs/<group>/<job_id>/). Default "MCP".
        transcription_device: Optional per-job override — GPU UUID or "cpu".
            Empty = the app's saved default. The proxy never picks a GPU itself.
        diarization_device: Optional per-job override — GPU UUID, "cpu", or "same"
            (follow this job's transcription device). Empty = saved default.
        whisper_compute_type: Optional per-job precision override (e.g.
            "int8_float32"). Empty = saved default / auto policy.

    Returns:
        {job_id, status:"queued", vram, warning?} on success,
        {error, models} if whisper_model is invalid.
    """
    # Validate model choice — enforce the "ask the user first" contract.
    if not whisper_model or whisper_model not in _VALID_MODELS:
        return {
            "error": (
                f"whisper_model '{whisper_model}' is not in the supported list. "
                "Ask the user which model to use, then re-call with a valid value."
            ),
            "supported_models": WHISPER_MODELS,
            "default": "large-v3-turbo",
        }

    # Resolve the audio file (absolute path, or name under uploads/).
    p = Path(audio_path)
    if not p.is_absolute():
        p = UPLOADS_DIR / audio_path
    if not p.exists():
        return {
            "error": f"Audio file not found: {p}. Use list_audio_files() to see "
                     f"available files in {UPLOADS_DIR}.",
            "uploads_dir": str(UPLOADS_DIR),
        }

    # Pre-flight VRAM check — device-specific, never pooled across GPUs.
    # If a per-job override names a GPU, use that device's inventory entry; if
    # the override isn't in the inventory we deliberately do NOT fall back to
    # the saved default's numbers — the app's submission validation will
    # reject or explain the unknown device.
    vram = await _get_vram()
    free_mb = None
    if transcription_device and transcription_device not in ("cpu", "same", ""):
        for dev in (vram or {}).get("devices", []):
            if dev.get("id") == transcription_device:
                free_mb = dev.get("memory_free_mib")
                break
    elif not transcription_device:
        free_mb = _parse_free_vram_mb(vram)
    warning = None
    if transcription_device == "cpu":
        warning = (
            "This job overrides transcription to CPU — expect much slower transcription."
        )
    elif free_mb is not None and free_mb < VRAM_WARN_MB:
        warning = (
            f"Low free VRAM on the target device (~{free_mb} MiB free). Consider a "
            "lighter model (large-v3-turbo / medium / small) or move to a GPU with "
            "headroom. Job submitted anyway — the app's queue serializes its own "
            "GPU work, but cannot protect against other processes on that GPU."
        )

    # Submit via the app's existing multipart endpoint (loopback). Streams the file
    # from disk so multi-GB interviews don't load fully into memory.
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as c:
            with open(p, "rb") as fh:
                resp = await c.post(
                    f"{TRANSCRIBER_URL}/api/transcribe",
                    files={"file": (p.name, fh, "application/octet-stream")},
                    data={
                        "language": language,
                        "num_speakers": "" if num_speakers <= 0 else str(num_speakers),
                        "group": group,
                        "whisper_model": whisper_model,
                        "transcription_device": transcription_device,
                        "diarization_device": diarization_device,
                        "whisper_compute_type": whisper_compute_type,
                    },
                )
            resp.raise_for_status()
            job_id = resp.json().get("job_id")
    except httpx.HTTPStatusError as e:
        return {"error": f"Transcriber rejected the job: {e.response.status_code} "
                         f"{e.response.text[:300]}"}
    except Exception as e:
        return {"error": f"Failed to submit job: {type(e).__name__}: {e}"}

    if not job_id:
        return {"error": "Transcriber did not return a job_id."}

    return {
        "job_id": job_id,
        "status": "queued",
        "whisper_model": whisper_model,
        "language": language,
        "group": group,
        "filename": p.name,
        "vram": vram,
        "warning": warning,
        "poll": f"Call get_job(\"{job_id}\") to check progress.",
    }


@mcp.tool()
async def get_job(job_id: str) -> dict:
    """Get status + result of a transcription job. Poll until status == 'done'.

    Maps the app's job statuses (queued / transcribing / diarizing / merging /
    building_outputs / complete / error) to running / done / error.

    When done, includes:
      transcript        - diarized text (truncated to MAX_TRANSCRIPT_CHARS; full
                          text is at transcript_path)
      transcript_path   - outputs/<group>/<job_id>/<stem>.txt
      srt_path, json_path
      language, duration_s, num_speakers, whisper_model
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{TRANSCRIBER_URL}/api/jobs/{job_id}")
            if r.status_code == 404:
                return {"status": "error", "error": f"Unknown job_id: {job_id}"}
            r.raise_for_status()
            job = r.json()
    except Exception as e:
        return {"status": "error", "error": f"Failed to fetch job: {type(e).__name__}: {e}"}

    status = job.get("status", "")
    if status in ("queued", "transcribing", "diarizing", "merging", "building_outputs"):
        return {
            "job_id": job_id,
            "status": "running",
            "step": job.get("step"),
            "filename": job.get("filename"),
        }
    if status == "error":
        return {
            "job_id": job_id,
            "status": "error",
            "error": job.get("error") or "job failed",
            "step": job.get("step"),
        }

    # status == "complete" — fetch the transcript text.
    transcript = ""
    files = job.get("files", {}) or {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            pr = await c.get(f"{TRANSCRIBER_URL}/api/preview/{job_id}/txt")
            if pr.status_code == 200:
                transcript = pr.json().get("content", "")
    except Exception:
        pass

    truncated = False
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = (
            transcript[:MAX_TRANSCRIPT_CHARS]
            + f"\n...[truncated {len(transcript) - MAX_TRANSCRIPT_CHARS} chars; "
            f"full text at {files.get('txt')}]"
        )
        truncated = True

    duration_s = job.get("duration")
    execution = job.get("execution") or {}
    transcription_exec = execution.get("transcription") or {}
    diarization_exec = execution.get("diarization") or {}
    return {
        "job_id": job_id,
        "status": "done",
        "filename": job.get("filename"),
        "language": job.get("language"),
        "duration_s": duration_s,
        "num_speakers": job.get("num_speakers"),
        "whisper_model": job.get("whisper_model"),
        "group": job.get("group"),
        "execution": {
            "transcription": {
                k: transcription_exec.get(k)
                for k in ("device", "device_id", "device_name", "compute_type", "elapsed_seconds")
                if transcription_exec.get(k) is not None
            },
            "diarization": {
                k: diarization_exec.get(k)
                for k in ("device", "device_id", "device_name", "elapsed_seconds")
                if diarization_exec.get(k) is not None
            },
        },
        "transcript": transcript,
        "truncated": truncated,
        "transcript_path": files.get("txt"),
        "srt_path": files.get("srt"),
        "json_path": files.get("json"),
        "audio_path": job.get("audio_path") or files.get("audio"),
    }


@mcp.tool()
async def list_jobs(group: str = "", filename_contains: str = "") -> dict:
    """List transcription jobs known to the app (lightweight: no transcript bodies).

    Each entry has id, filename, group, status, language, duration, num_speakers,
    created_at, completed_at. Use get_job(id) for the full transcript of a completed job.

    IMPORTANT for efficiency: narrow with `group` and/or `filename_contains`
    instead of listing every job and scanning it yourself. Call list_groups()
    first if you don't know the exact group name. To find jobs by topic/content
    rather than filename, use search_transcripts() instead.

    Args:
        group: Optional. Only return jobs in this group (case-insensitive exact match).
        filename_contains: Optional. Only return jobs whose filename contains this
            substring (case-insensitive).
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{TRANSCRIBER_URL}/api/jobs")
            r.raise_for_status()
            jobs = r.json()
    except Exception as e:
        return {"error": f"Failed to list jobs: {type(e).__name__}: {e}"}

    group_norm = group.strip().lower() if group else None
    needle = filename_contains.strip().lower() if filename_contains else None
    if group_norm is not None:
        jobs = [j for j in jobs if (j.get("group") or "").strip().lower() == group_norm]
    if needle is not None:
        jobs = [j for j in jobs if needle in (j.get("filename") or "").lower()]

    return {
        "group_filter": group or None,
        "filename_filter": filename_contains or None,
        "count": len(jobs),
        "jobs": [
            {
                "id": j.get("id"),
                "filename": j.get("filename"),
                "group": j.get("group"),
                "status": j.get("status"),
                "language": j.get("language"),
                "duration": j.get("duration"),
                "num_speakers": j.get("num_speakers"),
                "created_at": j.get("created_at"),
                "completed_at": j.get("completed_at"),
            }
            for j in jobs
        ],
    }


@mcp.tool()
def search_transcripts(query: str, group: str = "", max_results: int = 15, context_lines: int = 1) -> dict:
    """Search completed interview filenames, titles, job IDs, and transcript text.

    Use this for BOTH known interview identifiers (for example arint008) and
    topics mentioned in interviews. Matching is case-insensitive substring
    matching, not regex. Filename/title/ID matches do not require the query
    to appear in the spoken transcript.

    Returns matching jobs, matched_fields, transcript hit_count, and up to
    five sample_hits with surrounding context. hit_count counts only text
    matches; metadata-only matches have hit_count=0 and sample_hits=[].
    After finding a match, call get_job(job_id) to retrieve the transcript.

    Args:
        query: Non-empty filename fragment, interview code, keyword, or phrase.
        group: Optional case-insensitive exact group filter.
        max_results: Maximum matching jobs returned; must be positive.
        context_lines: Non-negative context lines before and after text hits.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "query must be a non-empty keyword or phrase."}
    if max_results < 1 or context_lines < 0:
        return {"error": "max_results must be positive; context_lines must be non-negative."}
    if not OUTPUTS_DIR.exists():
        return {"count": 0, "matches": [], "note": f"outputs dir not found: {OUTPUTS_DIR}"}

    group_norm = group.strip().lower() if group else None
    needle = query.lower()
    jobs_by_id = {j.get("id"): j for j in _load_jobs_index() if j.get("id")}

    # Search group directories directly (cheaper than opening every .txt when
    # a group filter narrows the directory scan).
    group_dirs = (
        [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.strip().lower() == group_norm]
        if group_norm is not None
        else [d for d in OUTPUTS_DIR.iterdir() if d.is_dir()]
    )

    matches: list[dict] = []
    scanned = 0
    for gdir in sorted(group_dirs):
        for job_dir in sorted(gdir.iterdir()):
            if not job_dir.is_dir():
                continue
            txt_files = list(job_dir.glob("*.txt"))
            if not txt_files:
                continue
            scanned += 1
            job_id = job_dir.name
            meta = jobs_by_id.get(job_id, {})
            metadata = {
                "filename": meta.get("filename"),
                "title": meta.get("title"),
                "job_id": job_id,
                "transcript_filename": txt_files[0].name,
            }
            matched_fields = [
                field for field, value in metadata.items()
                if value is not None and needle in str(value).lower()
            ]
            try:
                lines = txt_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            hits = []
            for i, line in enumerate(lines):
                if needle in line.lower():
                    lo = max(0, i - context_lines)
                    hi = min(len(lines), i + context_lines + 1)
                    hits.append({
                        "line_no": i + 1,
                        "context": "\n".join(lines[lo:hi]).strip(),
                    })
            if hits:
                matched_fields.append("content")
            if matched_fields:
                matches.append({
                    "job_id": job_id,
                    "filename": meta.get("filename") or txt_files[0].name,
                    "title": meta.get("title"),
                    "matched_fields": matched_fields,
                    "group": meta.get("group", gdir.name),
                    "hit_count": len(hits),
                    "sample_hits": hits[:5],
                })
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    return {
        "query": query,
        "group_filter": group or None,
        "transcripts_scanned": scanned,
        "count": len(matches),
        "matches": matches,
        "note": "Call get_job(job_id) on a match to pull the full transcript into context.",
    }


@mcp.tool()
def read_transcript_chunk(
    job_id: str,
    start_char: int,
    max_chars: int = 12000,
) -> dict:
    """Read transcript text by job ID without get_job's preview truncation.

    Start at start_char=0. Continue using next_start_char until has_more=false.
    Offsets are zero-based Python character offsets, not bytes or tokens.
    Each response includes a SHA-256 fingerprint; stop if it changes between calls.
    Use this tool, not read_file or resource URIs, for long interview retrieval.
    Read all chunks before claiming full coverage. Earlier chunks may still leave
    the model context: retain an evidence table with quotes and offsets.
    """
    import hashlib

    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        return {"error": "Invalid job_id; use the exact ID returned by list_jobs."}
    if start_char < 0 or not 1 <= max_chars <= 20000:
        return {
            "error": "start_char must be nonnegative; max_chars must be 1..20000."
        }

    root = OUTPUTS_DIR.resolve()
    candidates = []
    for candidate in root.glob(f"*/{job_id}/*.txt"):
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            candidates.append(resolved)
    candidates = sorted(set(candidates))

    if len(candidates) != 1:
        return {
            "error": "Expected exactly one transcript file for this job.",
            "candidate_count": len(candidates),
        }

    path = candidates[0]
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
    except OSError as exc:
        return {
            "error": f"Cannot read transcript: {type(exc).__name__}: {exc}"
        }

    total = len(text)
    if start_char > total:
        return {
            "error": "start_char exceeds transcript length.",
            "total_chars": total,
        }

    end = min(start_char + max_chars, total)
    return {
        "job_id": job_id,
        "transcript_filename": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "start_char": start_char,
        "end_char_exclusive": end,
        "total_chars": total,
        "has_more": end < total,
        "next_start_char": end if end < total else None,
        "text": text[start_char:end],
    }



# --- Run as Streamable HTTP at /mcp (no trailing slash) ----------------------
# Matches the SearXNG MCP pattern Open WebUI already consumes:
#   Admin Panel -> Settings -> External Tools -> + -> Type: MCP (Streamable HTTP)
#   URL: http://<storm1-tailscale-ip>:8002/mcp   (NO trailing slash with mcp >= 1.28;
#   /mcp/ 307-redirects to /mcp and drops the POST body)

def _make_app():
    return mcp.streamable_http_app()


if __name__ == "__main__":
    import uvicorn
    print(f"[whisper-mcp] proxying {TRANSCRIBER_URL} | uploads={UPLOADS_DIR}")
    print(f"[whisper-mcp] MCP endpoint: http://{HOST}:{PORT}/mcp  (no trailing slash)")
    uvicorn.run(_make_app(), host=HOST, port=PORT, log_level="info")
