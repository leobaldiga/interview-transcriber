# Pascal Worker Profile

Ready-made Python environments for running the transcription workers on a
Pascal-class GPU (GTX 10-series: 1050 Ti, 1060, 1070, 1080...).

## Why a separate profile

Pascal cards have no usable FP16 throughput, which changes two things:

1. **Compute type** — the app's `auto` policy is expected to resolve to
   `int8_float32` on Pascal, per the worker's probe order
   (`int8_float16 → int8_float32 → ...`) — the on-device probe confirms it.
   Nothing to configure; just don't force `int8_float16`.
2. **PyTorch build** — the pinned stack for Pascal workers is
   `torch 2.8.0+cu126` with `cudnn 9.10.2.21` (the cu126 wheel carries that
   cudnn natively — no manual cudnn install, and no touching the host NVIDIA
   driver).

The manifests in [requirements/whisper-pascal.txt](../requirements/whisper-pascal.txt)
and [requirements/pyannote-pascal.txt](../requirements/pyannote-pascal.txt)
pin that stack: `torch 2.8.0+cu126` with the cu126 wheel's native
`cudnn 9.10.2.21` (no manual cudnn install, no touching the host NVIDIA
driver), and the same faster-whisper / ctranslate2 / pyannote versions as
the host's existing environments. Dependency resolution for this exact
set has been verified; runtime behavior on the actual card is confirmed
by the compatibility test (pending a hardware run).

## Creating the worker environments

```bash
python3 scripts/setup_worker_envs.py            # creates .venv-whisper-pascal + .venv-pyannote-pascal
python3 scripts/setup_worker_envs.py --force    # rebuild from scratch (deletes the venvs first)
```

The script is non-destructive: it never touches `.venv`, `.venv-pyannote`,
or the host driver, refuses to run under Python < 3.9 or > 3.13 (the pinned
wheels only exist for cp39–cp313), verifies each install with `pip check`,
and freezes the resolved set into the venv for later audits.

## Pointing the app at them

Add to the app's launch environment (there is no dotenv support — export the
vars in the shell/systemd unit that starts `app.py`):

```bash
export WHISPER_WORKER_PYTHON=.venv-whisper-pascal/bin/python
export PYANNOTE_WORKER_PYTHON=.venv-pyannote-pascal/bin/python
```

Relative paths resolve against the repo root. Then select the Pascal GPU in
the UI's Compute panel (or `PUT /api/compute/settings`) and run the
compatibility Test.

## Verifying

- `python3 scripts/run_probe.py` — manual probe using the same launch path the
  app uses (inventory / torch / whisper smoke / pyannote smoke modes).
- The UI's **Test** button runs the same probe suite plus optional model
  smoke tests.
- After a job completes, `GET /api/jobs/<id>` reports the execution device and
  compute type — on Pascal expect `compute_type: "int8_float32"`.

## Rollback

```bash
unset WHISPER_WORKER_PYTHON PYANNOTE_WORKER_PYTHON
# restart the app; workers return to their previous interpreters
# (the legacy .venv / .venv-pyannote defaults)
```

Saved device settings are unaffected by this. The Pascal venvs can simply
be deleted (`rm -rf .venv-*-pascal`).
