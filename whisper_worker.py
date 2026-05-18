"""
Whisper transcription worker — runs as a standalone subprocess using .venv.
Called by app.py via subprocess.Popen with the venv Python executable.

Usage (internal):
    python whisper_worker.py <args_file> <result_file> <segments_file>

args_file:     { audio_path, language, model_size, device, compute_type, ... }
result_file:   written at end: { status, payload: {segments, info} } or { status, error }
segments_file: appended line-by-line as segments arrive (NDJSON), parent polls this file
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

TARGET_CHUNK_S = 20.0
MAX_CHUNK_S = 30.0
SPEECH_PAD_S = 0.15
MIN_SILENCE_S = 0.3
MIN_SPEECH_S = 0.1
IS_WINDOWS = os.name == "nt"


def _which_or_raise(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"Required executable not found on PATH: {name}")
    return exe


def _tool(name: str) -> str:
    return _which_or_raise(f"{name}.exe") if IS_WINDOWS else _which_or_raise(name)


def _get_audio_duration(audio_path: str) -> float:
    r = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _extract_chunk(audio_path: str, start: float, end: float, out_path: str):
    duration = end - start
    subprocess.run(
        [_tool("ffmpeg"), "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
        capture_output=True,
        timeout=120,
        check=True,
    )


def _run_silero_vad(audio_path: str, min_silence_s: float, min_speech_s: float, speech_pad_s: float) -> list:
    import numpy as np
    import torch

    sample_rate = 16000
    cmd = [_tool("ffmpeg"), "-y", "-i", audio_path, "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed: {result.stderr.decode(errors='replace')[:200]}")

    audio_np = np.frombuffer(result.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(audio_np)

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    speech_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        min_silence_duration_ms=int(min_silence_s * 1000),
        min_speech_duration_ms=int(min_speech_s * 1000),
        speech_pad_ms=int(speech_pad_s * 1000),
        return_seconds=True,
    )
    return [{"start": t["start"], "end": t["end"]} for t in speech_timestamps]


def _merge_speech_regions(regions: list, audio_duration: float) -> list:
    if not regions:
        return [{"start": 0.0, "end": audio_duration}]

    chunks = []
    current_start = regions[0]["start"]
    current_end = regions[0]["end"]

    for region in regions[1:]:
        merged_duration = region["end"] - current_start
        if merged_duration <= TARGET_CHUNK_S:
            current_end = region["end"]
        else:
            chunks.append({"start": current_start, "end": current_end})
            current_start = region["start"]
            current_end = region["end"]

    chunks.append({"start": current_start, "end": current_end})

    final_chunks = []
    for chunk in chunks:
        duration = chunk["end"] - chunk["start"]
        if duration <= MAX_CHUNK_S:
            final_chunks.append(chunk)
        else:
            pos = chunk["start"]
            while pos < chunk["end"]:
                seg_end = min(pos + TARGET_CHUNK_S, chunk["end"])
                final_chunks.append({"start": pos, "end": seg_end})
                pos = seg_end
    return final_chunks


def _detect_and_merge_chunks(audio_path: str) -> list:
    try:
        duration = _get_audio_duration(audio_path)
        print(f"[whisper_worker] VAD: detecting speech regions in {duration:.1f}s audio...", file=sys.stderr, flush=True)
        regions = _run_silero_vad(audio_path, MIN_SILENCE_S, MIN_SPEECH_S, SPEECH_PAD_S)
        print(f"[whisper_worker] VAD: found {len(regions)} speech regions", file=sys.stderr, flush=True)
        chunks = _merge_speech_regions(regions, duration)
        avg = sum(c["end"] - c["start"] for c in chunks) / len(chunks) if chunks else 0
        print(f"[whisper_worker] VAD: merged into {len(chunks)} transcription chunks (avg {avg:.1f}s each)", file=sys.stderr, flush=True)
        return chunks
    except Exception as e:
        print(f"[whisper_worker] VAD failed ({e}), falling back to single-chunk mode", file=sys.stderr, flush=True)
        duration = _get_audio_duration(audio_path) or 0.0
        return [{"start": 0.0, "end": duration or 9999.0}]


def main():
    if len(sys.argv) != 4:
        print("Usage: whisper_worker.py <args_file> <result_file> <segments_file>", file=sys.stderr)
        sys.exit(1)

    args_file, result_file, segments_file = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(args_file, "r", encoding="utf-8") as f:
        args = json.load(f)

    audio_path = args["audio_path"]
    language = args.get("language", None)
    model_size = args.get("model_size", "large-v3-turbo")
    device = args.get("device", "cuda")
    compute_type = args.get("compute_type", "float16")
    beam_size = int(args.get("beam_size", 1))
    temperature = float(args.get("temperature", 0.0))
    chunk_length = int(args.get("chunk_length", 20))
    no_speech_threshold = float(args.get("no_speech_threshold", 0.6))
    hotwords = args.get("hotwords", "").strip() or None
    condition_on_prev = bool(args.get("condition_on_prev", False))
    use_vad_preseg = bool(args.get("vad_preseg", True))

    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not IS_WINDOWS:
        os.environ.pop("LD_LIBRARY_PATH", None)

    try:
        _tool("ffmpeg")
        _tool("ffprobe")
        from faster_whisper import WhisperModel

        if device == "cuda":
            compute_type = "int8"

        print(f"[whisper_worker] Loading {model_size} on {device} ({compute_type})...", file=sys.stderr, flush=True)
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[whisper_worker] Model loaded.", file=sys.stderr, flush=True)

        _temperatures = (temperature,) if temperature > 0 else 0
        chunks = _detect_and_merge_chunks(audio_path) if use_vad_preseg else None
        if chunks and len(chunks) == 1 and chunks[0]["start"] == 0.0:
            use_vad_preseg = False

        all_segments = []
        detected_language = language
        detected_lang_prob = 1.0
        audio_duration = 0.0

        with open(segments_file, "w", buffering=1, encoding="utf-8") as seg_file:
            if use_vad_preseg and chunks:
                print(f"[whisper_worker] Transcribing {len(chunks)} VAD chunks...", file=sys.stderr, flush=True)
                tmp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
                lang_detected_from_first = False
                try:
                    for i, chunk in enumerate(chunks):
                        chunk_start = chunk["start"]
                        chunk_end = chunk["end"]
                        chunk_dur = chunk_end - chunk_start
                        if chunk_dur < 0.1:
                            continue

                        chunk_wav = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
                        try:
                            _extract_chunk(audio_path, chunk_start, chunk_end, chunk_wav)
                        except Exception as e:
                            print(f"[whisper_worker] chunk {i} extract failed: {e}", file=sys.stderr, flush=True)
                            continue

                        _lang = None if (not lang_detected_from_first and language is None) else language
                        try:
                            seg_gen, info = model.transcribe(
                                chunk_wav,
                                task="transcribe",
                                language=_lang,
                                beam_size=beam_size,
                                temperature=_temperatures,
                                word_timestamps=True,
                                condition_on_previous_text=False,
                                no_speech_threshold=no_speech_threshold,
                                hotwords=hotwords,
                                vad_filter=False,
                            )
                            chunk_segs = list(seg_gen)
                        except Exception as e:
                            print(f"[whisper_worker] chunk {i} transcription failed: {e}", file=sys.stderr, flush=True)
                            try:
                                os.unlink(chunk_wav)
                            except Exception:
                                pass
                            continue

                        if not lang_detected_from_first:
                            detected_language = info.language
                            detected_lang_prob = info.language_probability
                            lang_detected_from_first = True
                            print(f"[whisper_worker] Language detected: {detected_language} ({detected_lang_prob:.2f})", file=sys.stderr, flush=True)

                        audio_duration = max(audio_duration, chunk_end)
                        for seg in chunk_segs:
                            words = []
                            if seg.words:
                                for w in seg.words:
                                    words.append({
                                        "text": w.word.strip(),
                                        "start": round(w.start + chunk_start, 3),
                                        "end": round(w.end + chunk_start, 3),
                                        "probability": round(w.probability, 4) if hasattr(w, "probability") else 1.0,
                                    })
                            seg_dict = {
                                "start": round(seg.start + chunk_start, 3),
                                "end": round(seg.end + chunk_start, 3),
                                "text": seg.text.strip(),
                                "words": words,
                            }
                            all_segments.append(seg_dict)
                            seg_file.write(json.dumps(seg_dict) + "\n")
                            seg_file.flush()

                        try:
                            os.unlink(chunk_wav)
                        except Exception:
                            pass

                        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                            print(f"[whisper_worker] {i + 1}/{len(chunks)} chunks done ({chunk_end:.0f}s processed)", file=sys.stderr, flush=True)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                print("[whisper_worker] Transcribing in single-pass mode...", file=sys.stderr, flush=True)
                segments_gen, info = model.transcribe(
                    audio_path,
                    task="transcribe",
                    language=language,
                    beam_size=beam_size,
                    temperature=_temperatures,
                    word_timestamps=True,
                    condition_on_previous_text=condition_on_prev,
                    chunk_length=chunk_length,
                    no_speech_threshold=no_speech_threshold,
                    hotwords=hotwords,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
                )
                detected_language = info.language
                detected_lang_prob = info.language_probability
                audio_duration = info.duration

                for seg in segments_gen:
                    words = []
                    if seg.words:
                        for w in seg.words:
                            words.append({
                                "text": w.word.strip(),
                                "start": round(w.start, 3),
                                "end": round(w.end, 3),
                                "probability": round(w.probability, 4) if hasattr(w, "probability") else 1.0,
                            })
                    seg_dict = {
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "text": seg.text.strip(),
                        "words": words,
                    }
                    all_segments.append(seg_dict)
                    seg_file.write(json.dumps(seg_dict) + "\n")
                    seg_file.flush()

        info_dict = {
            "language": detected_language or "en",
            "language_probability": round(detected_lang_prob, 3),
            "duration": round(audio_duration, 2),
        }
        print(f"[whisper_worker] Done. {len(all_segments)} segments, {info_dict['duration']:.1f}s audio, lang={info_dict['language']}", file=sys.stderr, flush=True)

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "payload": {"segments": all_segments, "info": info_dict}}, f)
    except Exception as e:
        import traceback

        msg = f"{e}\n{traceback.format_exc()}"
        print(f"[whisper_worker] ERROR: {msg}", file=sys.stderr, flush=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({"status": "error", "payload": msg}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()
