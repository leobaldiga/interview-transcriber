"""
Diarization worker — runs as a standalone script using .venv-pyannote.
Called by app.py via subprocess.Popen with the venv Python executable.

Usage (internal — do not call directly):
    python diarize_worker.py <args_json_file> <result_json_file>

args_json_file contains:  { audio_path, hf_token, num_speakers }
result_json_file written: { status: "ok"|"error", payload: [...turns] | "error message" }

Progress is printed to stderr so the parent process can read it.
"""

import json
import os
import sys
import typing
import typing_extensions
import warnings

warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*")


def main():
    if len(sys.argv) != 3:
        print("Usage: diarize_worker.py <args_file> <result_file>", file=sys.stderr)
        sys.exit(1)

    args_file = sys.argv[1]
    result_file = sys.argv[2]

    with open(args_file, "r", encoding="utf-8") as f:
        args = json.load(f)

    audio_path = args["audio_path"]
    hf_token = args["hf_token"]
    num_speakers = args.get("num_speakers")
    min_silence = float(args.get("min_silence", 0.5))
    min_cluster_size = int(args.get("min_cluster_size", 75))
    seg_onset = float(args.get("seg_onset", 0.6))
    seg_offset = float(args.get("seg_offset", 0.4))

    import signal

    def _write_error(msg: str):
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({"status": "error", "payload": msg}, f)

    def _sigabrt_handler(signum, frame):
        msg = "[diarize_worker] CUDA crash (SIGABRT) — illegal memory access in pyannote GPU inference"
        print(msg, file=sys.stderr, flush=True)
        _write_error(msg)
        sys.exit(1)

    if os.name != "nt":
        try:
            signal.signal(signal.SIGABRT, _sigabrt_handler)
        except Exception:
            pass

    try:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        import torch

        try:
            from torch.serialization import add_safe_globals
            from torch.torch_version import TorchVersion

            add_safe_globals([TorchVersion])
        except Exception:
            pass

        _real_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _real_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load

        try:
            import lightning_fabric.utilities.cloud_io as _lf_cloud_io

            def _lf_patched_load(path, map_location=None, **kwargs):
                kwargs["weights_only"] = False
                return torch.load(path, map_location=map_location, **kwargs)

            _lf_cloud_io._load = _lf_patched_load
        except Exception:
            pass

        try:
            import lightning.pytorch.utilities.cloud_io as _lp_cloud_io

            def _lp_patched_load(path, map_location=None, **kwargs):
                kwargs["weights_only"] = False
                return torch.load(path, map_location=map_location, **kwargs)

            if hasattr(_lp_cloud_io, "load"):
                _lp_cloud_io.load = _lp_patched_load
            if hasattr(_lp_cloud_io, "_load"):
                _lp_cloud_io._load = _lp_patched_load
        except Exception:
            pass

        try:
            import pytorch_lightning.utilities.cloud_io as _pl_cloud_io

            def _pl_cloud_patched_load(path, map_location=None, **kwargs):
                kwargs["weights_only"] = False
                return torch.load(path, map_location=map_location, **kwargs)

            if hasattr(_pl_cloud_io, "load"):
                _pl_cloud_io.load = _pl_cloud_patched_load
            if hasattr(_pl_cloud_io, "_load"):
                _pl_cloud_io._load = _pl_cloud_patched_load
        except Exception:
            pass

        try:
            import pytorch_lightning.core.saving as _pl_saving

            if hasattr(_pl_saving, "pl_load"):
                _orig_pl_load = _pl_saving.pl_load

                def _patched_pl_load(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return _orig_pl_load(*args, **kwargs)

                _pl_saving.pl_load = _patched_pl_load
        except Exception:
            pass

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

        try:
            pipeline._segmentation.model.specifications
            pipeline.instantiate(
                {
                    "segmentation": {
                        "min_duration_off": min_silence,
                        "onset": seg_onset,
                        "offset": seg_offset,
                    },
                    "clustering": {
                        "min_cluster_size": min_cluster_size,
                    },
                }
            )
            print(
                f"[diarize_worker] Params: min_silence={min_silence} onset={seg_onset} offset={seg_offset} min_cluster={min_cluster_size}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as hp_err:
            print(f"[diarize_worker] Note: could not set hyperparams: {hp_err}", file=sys.stderr, flush=True)

        print(f"[diarize_worker] Pipeline loaded on {device}, starting inference...", file=sys.stderr, flush=True)

        import time as _time

        t_start = _time.time()
        completed_turns = [0]
        last_step = [""]

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            try:
                elapsed = _time.time() - t_start
                if step_name and step_name != last_step[0]:
                    last_step[0] = step_name
                    print(
                        f"[diarize_worker] step: {step_name} ({elapsed:.0f}s elapsed)",
                        file=sys.stderr,
                        flush=True,
                    )
                if isinstance(step_artifact, Annotation):
                    n = sum(1 for _ in step_artifact.itertracks())
                    if n > completed_turns[0]:
                        completed_turns[0] = n
                        print(
                            f"[diarize_worker] {n} speaker turns found ({elapsed:.0f}s)",
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception:
                pass

        kwargs = {"hook": hook}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers

        diarization = pipeline(audio_path, **kwargs)

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                }
            )

        print(f"[diarize_worker] Done — {len(turns)} turns", file=sys.stderr, flush=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "payload": turns}, f)

    except Exception as e:
        import traceback

        msg = f"{e}\n{traceback.format_exc()}"
        print(f"[diarize_worker] ERROR: {msg}", file=sys.stderr, flush=True)
        _write_error(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
