"""Foundation tests: the app imports cleanly with redirected storage and never
writes job data outside the redirected directories."""

import app


def test_storage_redirected_from_production_paths():
    # The conftest redirects these before app import; the app must honor them.
    import os
    from pathlib import Path

    assert app.UPLOAD_DIR == Path(os.environ["UPLOADS_DIR"])
    assert app.OUTPUT_DIR == Path(os.environ["OUTPUTS_DIR"])
    assert app.JOBS_INDEX_PATH == app.OUTPUT_DIR / "jobs.json"
    # Redirected dirs exist (app creates them at import).
    assert app.UPLOAD_DIR.is_dir()
    assert app.OUTPUT_DIR.is_dir()


def test_app_has_no_gpu_allocation_in_parent():
    # The FastAPI parent process must not import torch/ctranslate2 — worker
    # subprocesses own all GPU work. This is a contract this feature relies on.
    import sys

    assert "torch" not in sys.modules
    assert "ctranslate2" not in sys.modules


def test_health_of_core_helpers():
    # Sanity: helpers used by the queue still behave.
    assert app._to_rel_path("uploads/x.mp3") == "uploads/x.mp3"
    assert app._normalize_job_paths({"files": {"txt": "outputs/g/j/a.txt"}, "audio_path": "uploads/x.mp3"}) == {
        "files": {"txt": "outputs/g/j/a.txt"},
        "audio_path": "uploads/x.mp3",
    }
