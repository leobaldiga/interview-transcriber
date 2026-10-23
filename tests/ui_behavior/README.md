# UI behavior test (jsdom + real app handlers)

Exercises the real `static/js/app.js` in a jsdom DOM against the REAL app
handlers (only the GPU inventory is faked — no GPU needed). Covers selector
population, Save defaults round-trip, selected-device VRAM display (including
the no-valid-selection message), the compatibility Test button, and per-job
device overrides in the submission FormData.

Scope note: the upload step uses a stubbed `XMLHttpRequest` that captures the
`FormData` — it verifies what the UI would send, not a real
`POST /api/transcribe` round-trip (that endpoint contract is covered by
`tests/test_app_compute.py` via the ASGI test client).

Safety: `server.py` forces uploads/outputs/compute settings into a temp
scratch dir before importing the app — running this cannot touch real storage
or saved defaults.

Not part of the pytest run (needs Node + jsdom). To execute:

```bash
cd tests/ui_behavior
npm install jsdom            # once, local node_modules
python3 server.py &          # scratch dirs + faked GPUs, serves on :8791
node ui_behavior_test.mjs    # UI_TEST_PORT=... to change the port
kill %1
```
