# whisper-mcp proxy

An MCP (Streamable HTTP) tool server that lets **Gemma** (via Open WebUI)
transcribe large audio files by driving your already-running **Interview
Transcriber** app (FastAPI on `:8765`). It owns no models and does no GPU work —
it discovers files, submits jobs through the app's existing queue, polls status,
and returns transcript text. Outputs land in `outputs/<group>/<job_id>/` exactly
as the web UI produces them.

This mirrors the SearXNG MCP pattern already in your `~/gemma-stack` Open WebUI
setup: Streamable HTTP at `/mcp`. **Note:** with `mcp >= 1.28` the endpoint is
`/mcp` (NO trailing slash) — `/mcp/` 307-redirects to `/mcp` and drops the POST
body.

## Files

| File | Purpose |
|------|---------|
| `mcp_proxy.py` | FastMCP server: 5 tools, pure `httpx` client to `:8765` |
| `requirements-mcp.txt` | `mcp` + `httpx` + `uvicorn` (installed into the app's `.venv`) |
| `whisper-mcp.service` | systemd user unit for storm1 |

## Tools exposed to Gemma

- **`list_whisper_models()`** — model table with VRAM/speed/accuracy (int8/CUDA
  footprints). The transcribe tool enforces choosing one.
- **`list_audio_files()`** — scans the app's `uploads/` dir; returns name, path,
  size, duration, and whether already transcribed.
- **`transcribe_audio(audio_path, whisper_model, language?, num_speakers?, group?)`**
  — **required** `whisper_model`. If empty/invalid, returns the model list and
  tells Gemma to ask you first. Pre-flight VRAM warn (non-blocking). Submits via
  the app's multipart `/api/transcribe`; returns `{job_id, status:"queued"}`.
- **`get_job(job_id)`** — poll until `done`; returns the diarized transcript
  (truncated, full text at `transcript_path`) + `txt/srt/json` paths.
- **`list_jobs()`** — lightweight job list (no transcript bodies).

## Install (on storm1, in the interview-transcriber repo)

```bash
cd ~/interview-transcriber
# copy mcp_proxy.py, requirements-mcp.txt, whisper-mcp.service into the repo root
./.venv/bin/pip install -r requirements-mcp.txt
```

## Run

Either directly:

```bash
cd ~/interview-transcriber
./.venv/bin/python mcp_proxy.py
# -> MCP endpoint: http://0.0.0.0:8002/mcp  (no trailing slash)
```

Or as a systemd user service (recommended — auto-restarts, survives logout):

```bash
mkdir -p ~/.config/systemd/user
cp whisper-mcp.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now whisper-mcp
journalctl --user -u whisper-mcp -f          # check it started
sudo loginctl enable-linger $USER            # run when not SSH'd in
```

## Register in Open WebUI (same pattern as SearXNG)

1. Open `http://<storm1-tailscale-ip>:3000`, log in as admin.
2. **Admin Panel → Settings → External Tools → +**.
3. Add a tool server:
   - **Type:** MCP (Streamable HTTP)
   - **URL:** `http://<storm1-tailscale-ip>:8002/mcp` (NO trailing slash with
     `mcp >= 1.28`; `/mcp/` 307-redirects and drops the POST body)
   - **Auth:** None
   - **ID:** `whisper`
   - **Name:** Whisper Transcription
4. Save. Then **Admin Panel → Settings → Models**, edit Gemma, enable the
   Whisper Transcription tool.

**If Open WebUI runs in Docker** (the `~/gemma-stack` compose) and can't reach
the host process directly, use `http://host.docker.internal:8002/mcp` and add to
the `open-webui` service in `docker-compose.yml`:

```yaml
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

## Verify the handshake

```bash
curl -i -X POST http://localhost:8002/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}'
```

You should get `200` with a JSON-RPC `result` whose `serverInfo.name` is
`"interview-transcriber"` (SSE-framed, plus an `mcp-session-id` header). If you
get a `307`, you used `/mcp/` (trailing slash) — drop it and use `/mcp`.

## Using it from a Gemma chat

Drop a file into the app's uploads dir:

```bash
cp /path/to/interview.mp3 ~/interview-transcriber/uploads/
```

Then in Open WebUI, with the Whisper tool enabled, ask Gemma:

> Transcribe `20260401_ARINT003.mp3` (Thai, 2 speakers) and summarize the themes.

Gemma will: ask which model (or you tell it `large-v3-turbo`), call
`transcribe_audio`, poll `get_job` until done, then read the returned transcript.
The `.txt`/`.srt`/`.json` also land on disk at
`~/interview-transcriber/outputs/MCP/<job_id>/`.

## Hardware notes (shared 12 GB VRAM with Gemma)

- The app already serializes GPU jobs via its async `_job_queue` and releases
  VRAM after each job (whisper + pyannote run as subprocesses that exit per job).
  **The proxy adds no GPU work and no second lock** — it only enqueues. Safe
  alongside Gemma by design.
- `transcribe_audio` returns the live VRAM reading (via `/api/vram`) and, if free
  VRAM < `VRAM_WARN_MB` (default 2048 MiB), a non-blocking warning suggesting a
  lighter model or waiting for Gemma to idle.
- Default `large-v3-turbo` (int8 ~1.5–2.5 GB) is the right co-resident choice.
  Use `large-v3` (int8 ~2.5–3.5 GB) for max accuracy when Gemma is idle.

## Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `TRANSCRIBER_URL` | `http://localhost:8765` | The transcriber app base URL |
| `TRANSCRIBER_DIR` | script dir | Repo root (where `uploads/` + `outputs/` live) |
| `UPLOADS_DIR` | `<TRANSCRIBER_DIR>/uploads` | Scanned by `list_audio_files` |
| `OUTPUTS_DIR` | `<TRANSCRIBER_DIR>/outputs` | Read `jobs.json` for transcribed-flag |
| `PORT` | `8002` | MCP listen port |
| `HOST` | `0.0.0.0` | Bind for Tailscale access |
| `VRAM_WARN_MB` | `2048` | Free-VRAM threshold for the non-blocking warning |
| `MAX_TRANSCRIPT_CHARS` | `60000` | Truncate transcript in `get_job` response |

## Optional streamlining (not implemented)

Multipart submission re-uploads each file over loopback and duplicates it in
`uploads/` (the app prefixes the name with `<job_id>_`). For very large files you
could add a ~20-line `POST /api/transcribe-path` endpoint to `app.py` that
accepts a path already in `uploads/` and skips the copy, then point the proxy at
it. Not needed for correctness — loopback copy is fast.
