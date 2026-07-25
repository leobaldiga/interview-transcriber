"""faster-whisper backend — drives `whisper_worker.py` in the .venv subprocess.

This is a straight lift of the subprocess driver that used to live inline in
`app.transcribe_audio`. The worker script itself is untouched, so model loading,
VAD pre-segmentation, beam size and word_timestamps behave exactly as before.
This engine remains the default and the fallback.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .errors import EngineUnavailableError, TranscriptionError
from .process import build_worker_env, spawn_worker, venv_python_path
from .schema import TranscribeOptions, TranscriptResult

NAME = "faster_whisper"

_REPO_ROOT = Path(__file__).resolve().parent.parent


class FasterWhisperEngine:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else _REPO_ROOT
        self.python_bin = venv_python_path(self.base_dir / ".venv")
        self.script = self.base_dir / "whisper_worker.py"

    @property
    def name(self) -> str:
        return NAME

    def health(self) -> None:
        if not self.python_bin.exists():
            raise EngineUnavailableError(f"Whisper venv not found at {self.python_bin}.")
        if not self.script.exists():
            raise EngineUnavailableError(f"Whisper worker script not found at {self.script}.")

    def is_available(self) -> bool:
        try:
            self.health()
        except EngineUnavailableError:
            return False
        return True

    def transcribe(self, audio_path: str | Path, options: TranscribeOptions) -> TranscriptResult:
        self.health()
        hooks = options.hooks
        audio_path = str(audio_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            args_file = f.name
            json.dump(options.worker_args(audio_path), f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            result_file = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False, encoding="utf-8") as f:
            segments_file = f.name

        proc = None
        started = time.monotonic()
        try:
            proc = spawn_worker(
                [str(self.python_bin), str(self.script), args_file, result_file, segments_file],
                env=build_worker_env(self.python_bin),
            )
            if hooks.on_process_start:
                hooks.on_process_start(proc.pid)

            def _drain_segments():
                try:
                    with open(segments_file, "r", encoding="utf-8") as sf:
                        for line in sf:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                hooks.segment(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                except FileNotFoundError:
                    pass

            while proc.poll() is None:
                _drain_segments()
                if proc.stderr:
                    line = proc.stderr.readline()
                    if line:
                        hooks.log(line.decode(errors="replace").rstrip())
                time.sleep(0.2)

            if proc.stderr:
                for line in proc.stderr:
                    hooks.log(line.decode(errors="replace").rstrip())

            proc.wait()
            _drain_segments()

            if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
                raise TranscriptionError(f"Whisper worker crashed (exit {proc.returncode})")

            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)

            if result.get("status") == "error":
                raise TranscriptionError(f"Whisper failed: {result.get('payload')}")

            transcript = TranscriptResult.from_payload(result.get("payload", {}), engine=NAME)
            transcript.wall_time_s = time.monotonic() - started
            if transcript.segments and not transcript.has_word_timestamps:
                transcript.warnings.append("faster-whisper returned no word-level timestamps")
            return transcript

        finally:
            if proc is not None and hooks.on_process_end:
                hooks.on_process_end(proc.pid)
            for p in (args_file, result_file, segments_file):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
