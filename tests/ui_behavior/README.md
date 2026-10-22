# UI behavior test (jsdom + real app handlers)

Exercises the real `static/js/app.js` in a jsdom DOM against the REAL app
handlers (only the GPU inventory is faked — no GPU needed). Covers selector
population, Save defaults round-trip, selected-device VRAM display, the
compatibility Test button, and per-job device overrides on submission.

Not part of the pytest run (needs Node + jsdom). To execute:

```bash
cd tests/ui_behavior
npm install jsdom            # once, local node_modules
python3 server.py &         # or: COMPUTE_SETTINGS_PATH=/tmp/compute.json python3 server.py
node ui_behavior_test.mjs
kill %1
```

server.py expects the repo checkout at the repo root (edit REPO/BASE paths in
both files if running from elsewhere).
