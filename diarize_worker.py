"""
Diarization worker — runs as a standalone script using .venv-pyannote.
Called by app.py via subprocess.Popen with .venv-pyannote/bin/python.

Usage (internal — do not call directly):
    .venv-pyannote/bin/python diarize_worker.py <args_json_file> <result_json_file>

args_json_file contains:  { audio_path, hf_token, num_speakers }
result_json_file written: { status: "ok"|"error", payload: [...turns] | "error message" }

Progress is printed to stderr so the parent process can read it.
"""

# Force typing module to fully initialize before torch/lightning imports.
# Under systemd, partial module initialization can leave typing.Optional as None
# which crashes torch._numpy._ufuncs on import.
import typing
import typing_extensions
import json
import sys
import os
import warnings

warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*")

def main():
    if len(sys.argv) != 3:
        print("Usage: diarize_worker.py <args_file> <result_file>", file=sys.stderr)
        sys.exit(1)

    args_file   = sys.argv[1]
    result_file = sys.argv[2]

    with open(args_file) as f:
        args = json.load(f)

    audio_path        = args["audio_path"]
    hf_token          = args["hf_token"]
    num_speakers      = args.get("num_speakers")
    min_silence       = float(args.get("min_silence", 0.5))
    min_cluster_size  = int(args.get("min_cluster_size", 75))
    # onset/offset thresholds for the segmentation model.
    # Higher onset = pyannote needs louder/longer activity before declaring a
    # new speaker turn, which strongly reduces spurious single-word segments.
    # pyannote 3.1 default onset=0.5, offset=0.5.
    seg_onset         = float(args.get("seg_onset", 0.6))
    seg_offset        = float(args.get("seg_offset", 0.4))

    # Suppress CUDA core dump noise — worker crashes should be caught as exceptions,
    # but if the C++ runtime calls abort() (e.g. CUDA illegal memory access),
    # redirect the core dump message to stderr only, don't kill the parent.
    import signal
    def _sigabrt_handler(signum, frame):
        msg = "[diarize_worker] CUDA crash (SIGABRT) — illegal memory access in pyannote GPU inference"
        print(msg, file=sys.stderr, flush=True)
        with open(result_file, "w") as f:
            json.dump({"status": "error", "payload": msg}, f)
        sys.exit(1)
    signal.signal(signal.SIGABRT, _sigabrt_handler)

    try:
        import torch

        # PyTorch 2.6+ weights_only fix for pyannote checkpoints
        import lightning_fabric.utilities.cloud_io as _lf_io
        def _patched(path, map_location=None, **kwargs):
            kwargs["weights_only"] = False
            return torch.load(path, map_location=map_location, **kwargs)
        _lf_io._load = _patched

        # torchaudio compatibility for pyannote 3.x
        import torchaudio as _ta
        if not hasattr(_ta, "set_audio_backend"):
            _ta.set_audio_backend = lambda *a, **kw: None
        if not hasattr(_ta, "get_audio_backend"):
            _ta.get_audio_backend = lambda: "soundfile"

        from pyannote.audio import Pipeline
        from pyannote.core import Annotation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[diarize_worker] Loading pyannote on {device}...", file=sys.stderr, flush=True)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        pipeline.to(torch.device(device))

        # Reduce over-segmentation: require louder/longer activity before
        # declaring a new turn (onset/offset), longer silence gap before
        # ending a turn (min_duration_off), and a larger cluster minimum to
        # stop pyannote creating a spurious 3rd-speaker cluster from overlaps.
        # pyannote 3.1 defaults: onset=0.5, offset=0.5, min_duration_off=0.0,
        #   min_cluster_size=15 (very permissive).
        try:
            pipeline._segmentation.model.specifications  # check it's loaded
            pipeline.instantiate({
                "segmentation": {
                    "min_duration_off": min_silence,
                    "onset":            seg_onset,
                    "offset":           seg_offset,
                },
                "clustering": {
                    "min_cluster_size": min_cluster_size,
                },
            })
            print(
                f"[diarize_worker] Params: min_silence={min_silence} "
                f"onset={seg_onset} offset={seg_offset} "
                f"min_cluster={min_cluster_size}",
                file=sys.stderr, flush=True,
            )
        except Exception as _hp_err:
            print(f"[diarize_worker] Note: could not set hyperparams: {_hp_err}",
                  file=sys.stderr, flush=True)

        print(f"[diarize_worker] Pipeline loaded on {device}, starting inference...", file=sys.stderr, flush=True)

        import time as _time
        t_start = _time.time()
        completed_turns = [0]
        last_step = [""]

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            try:
                elapsed = _time.time() - t_start
                # Print each new pipeline step name so the user sees progress
                if step_name and step_name != last_step[0]:
                    last_step[0] = step_name
                    print(f"[diarize_worker] step: {step_name} ({elapsed:.0f}s elapsed)",
                          file=sys.stderr, flush=True)
                if isinstance(step_artifact, Annotation):
                    n = sum(1 for _ in step_artifact.itertracks())
                    if n > completed_turns[0]:
                        completed_turns[0] = n
                        print(f"[diarize_worker] {n} speaker turns found ({elapsed:.0f}s)",
                              file=sys.stderr, flush=True)
            except Exception:
                pass

        kwargs = {"hook": hook}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers

        diarization = pipeline(audio_path, **kwargs)

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": round(turn.start, 3),
                "end":   round(turn.end, 3),
                "speaker": speaker,
            })

        print(f"[diarize_worker] Done — {len(turns)} turns", file=sys.stderr, flush=True)

        with open(result_file, "w") as f:
            json.dump({"status": "ok", "payload": turns}, f)

    except Exception as e:
        import traceback
        msg = f"{e}\n{traceback.format_exc()}"
        print(f"[diarize_worker] ERROR: {msg}", file=sys.stderr, flush=True)
        with open(result_file, "w") as f:
            json.dump({"status": "error", "payload": msg}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()
