#!/usr/bin/env python3
"""Compare FasterWhisperEngine and WhisperCppHttpEngine on one real recording.

Designed for a 60+ minute interview. Long-file timestamp drift — the kind that
accumulates across VAD chunk boundaries and only becomes visible an hour in — is
the headline metric, so drift is reported at fixed checkpoints spread across the
whole file rather than as a single average.

    python scripts/validate_engines.py interview.wav

Run it from the repo root with the app's .venv active. The whisper.cpp container
must be up (see docker/whisper-cpp/README.md); faster-whisper runs through the
usual .venv worker subprocess.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engines import EngineHooks, TranscribeOptions, TranscriptResult, get_engine  # noqa: E402

CHECKPOINTS = (0.10, 0.25, 0.50, 0.75, 0.90)
_PUNCT = re.compile(r"[^\w']+", re.UNICODE)


# ── metrics ─────────────────────────────────────────────────────────


def flat_words(result: TranscriptResult) -> list[tuple[str, float, float]]:
    return [(w.text, w.start, w.end) for seg in result.segments for w in seg.words]


def normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower())


def word_density(result: TranscriptResult) -> float:
    """Words per minute of audio."""
    minutes = result.duration / 60.0
    return len(flat_words(result)) / minutes if minutes > 0 else 0.0


def structural_issues(result: TranscriptResult) -> dict:
    """Monotonicity violations, zero-length and overlapping spans."""
    issues = {
        "segment_non_monotonic": 0,
        "segment_zero_length": 0,
        "segment_overlaps": 0,
        "word_non_monotonic": 0,
        "word_zero_length": 0,
        "word_outside_segment": 0,
        "examples": [],
    }

    prev_end = None
    for seg in result.segments:
        if seg.end < seg.start:
            issues["segment_non_monotonic"] += 1
            _example(issues, f"segment end {seg.end:.3f} < start {seg.start:.3f}")
        if seg.end == seg.start:
            issues["segment_zero_length"] += 1
        if prev_end is not None:
            if seg.start < prev_end - 1e-6:
                issues["segment_overlaps"] += 1
                _example(issues, f"segment starts {prev_end - seg.start:.3f}s before previous end at {seg.start:.1f}s")
        prev_end = seg.end

        prev_word_end = None
        for word in seg.words:
            if word.end < word.start:
                issues["word_non_monotonic"] += 1
                _example(issues, f"word {word.text!r} end < start at {word.start:.1f}s")
            if word.end == word.start:
                issues["word_zero_length"] += 1
            if prev_word_end is not None and word.start < prev_word_end - 1e-6:
                issues["word_non_monotonic"] += 1
            prev_word_end = word.end
            if word.start < seg.start - 0.05 or word.end > seg.end + 0.05:
                issues["word_outside_segment"] += 1

    return issues


def _example(issues: dict, message: str) -> None:
    if len(issues["examples"]) < 5:
        issues["examples"].append(message)


def anchor_drift(a: TranscriptResult, b: TranscriptResult) -> list[tuple[float, float]]:
    """Return (time_in_a, drift) pairs for words both engines agree on.

    Alignment uses difflib matching blocks over the normalised word sequences, so
    disagreements in wording do not shift every later anchor.
    """
    words_a = [(normalize(t), s) for t, s, _ in flat_words(a) if normalize(t)]
    words_b = [(normalize(t), s) for t, s, _ in flat_words(b) if normalize(t)]
    if not words_a or not words_b:
        return []

    matcher = difflib.SequenceMatcher(None, [w for w, _ in words_a], [w for w, _ in words_b], autojunk=False)
    drifts = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            t_a = words_a[block.a + offset][1]
            t_b = words_b[block.b + offset][1]
            drifts.append((t_a, t_b - t_a))
    return drifts


def drift_at_checkpoints(drifts: list[tuple[float, float]], duration: float, window: float = 60.0) -> list[dict]:
    """Median/max |drift| in a window around each checkpoint of the file."""
    rows = []
    for fraction in CHECKPOINTS:
        centre = duration * fraction
        local = [d for t, d in drifts if abs(t - centre) <= window / 2]
        if not local:
            local = [d for t, d in drifts if abs(t - centre) <= window * 2]
        rows.append(
            {
                "fraction": fraction,
                "at_s": centre,
                "n": len(local),
                "median": statistics.median(local) if local else None,
                "max_abs": max(abs(d) for d in local) if local else None,
            }
        )
    return rows


# ── downstream compatibility ────────────────────────────────────────


def synthetic_turns(duration: float, period: float = 30.0) -> list[dict]:
    turns, t, index = [], 0.0, 0
    while t < duration:
        turns.append({"start": t, "end": min(t + period, duration), "speaker": f"SPEAKER_{index % 2:02d}"})
        t += period
        index += 1
    return turns


def run_downstream(result: TranscriptResult, turns: list[dict]) -> dict:
    """Push a result through app.py's unmodified speaker-merge pipeline."""
    import app

    segments = result.to_payload()["segments"]
    labeled = app.assign_speakers_to_words(segments, turns)
    labeled = app._merge_short_turns(labeled, min_turn_duration=1.5, min_word_count=4)
    labeled = app.absorb_backchannels(labeled, max_words=2, max_duration=1.0)
    transcript = app.build_diarized_transcript(labeled)
    srt = app.build_srt(labeled)
    speakers = sorted({w["speaker"] for w in labeled})
    return {
        "words": len(labeled),
        "speakers": len(speakers),
        "turns": transcript.count("\n") + 1 if transcript else 0,
        "txt_chars": len(transcript),
        "srt_blocks": srt.count("-->"),
    }


# ── driver ──────────────────────────────────────────────────────────


def run_engine(engine_name: str, audio: Path, options: TranscribeOptions) -> TranscriptResult:
    engine = get_engine(engine_name)
    print(f"\n[{engine_name}] health check...", flush=True)
    engine.health()
    print(f"[{engine_name}] transcribing {audio.name}...", flush=True)
    started = time.monotonic()
    result = engine.transcribe(audio, options)
    print(f"[{engine_name}] done in {time.monotonic() - started:.1f}s "
          f"({len(result.segments)} segments, {result.word_count} words)", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", default=None, help="ISO code; omit for auto-detect")
    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper model size")
    parser.add_argument("--diarization", type=Path, help="JSON list of {start,end,speaker} turns; synthesised if absent")
    parser.add_argument("--max-drift", type=float, default=1.0, help="fail if |drift| at any checkpoint exceeds this")
    parser.add_argument("--min-density-ratio", type=float, default=0.5,
                        help="fail if whisper.cpp word density falls below this fraction of faster-whisper's")
    parser.add_argument("--json-out", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"FAIL: audio file not found: {args.audio}")
        return 2

    options = TranscribeOptions(
        language=args.language,
        model=args.model,
        hooks=EngineHooks(on_log=lambda line: None),
    )

    results: dict[str, TranscriptResult] = {}
    failures: list[str] = []
    for engine_name in ("faster_whisper", "whisper_cpp"):
        try:
            results[engine_name] = run_engine(engine_name, args.audio, options)
        except Exception as exc:
            failures.append(f"{engine_name} failed: {type(exc).__name__}: {exc}")
            print(f"[{engine_name}] ERROR {type(exc).__name__}: {exc}", flush=True)

    report: dict = {"audio": str(args.audio), "engines": {}}

    print("\n" + "=" * 72)
    print("1. WORD-LEVEL TIMESTAMPS")
    print("=" * 72)
    for name, result in results.items():
        density = word_density(result)
        report["engines"][name] = {
            "segments": len(result.segments),
            "words": result.word_count,
            "duration_s": result.duration,
            "words_per_minute": round(density, 1),
            "has_word_timestamps": result.has_word_timestamps,
            "warnings": result.warnings,
            "wall_time_s": round(result.wall_time_s, 1),
        }
        print(f"  {name:16s} words={result.word_count:<7d} density={density:7.1f}/min  "
              f"word_timestamps={result.has_word_timestamps}  duration={result.duration:.1f}s "
              f"wall={result.wall_time_s:.1f}s")
        for warning in result.warnings:
            print(f"  {'':16s} WARNING: {warning}")
        if not result.has_word_timestamps:
            failures.append(f"{name} produced no word-level timestamps")

    if len(results) == 2:
        ratio_base = word_density(results["faster_whisper"])
        ratio = word_density(results["whisper_cpp"]) / ratio_base if ratio_base else 0.0
        report["density_ratio"] = round(ratio, 3)
        print(f"  density ratio whisper_cpp/faster_whisper = {ratio:.2f}")
        if ratio < args.min_density_ratio:
            failures.append(f"whisper.cpp word density is {ratio:.2f}x faster-whisper's")

    print("\n" + "=" * 72)
    print("2. TIMESTAMP DRIFT ACROSS THE FILE  (whisper_cpp minus faster_whisper)")
    print("=" * 72)
    if len(results) == 2:
        drifts = anchor_drift(results["faster_whisper"], results["whisper_cpp"])
        duration = max(results["faster_whisper"].duration, results["whisper_cpp"].duration)
        if not drifts:
            failures.append("no common text anchors between engines — cannot measure drift")
            print("  no common anchors found")
        else:
            all_abs = [abs(d) for _, d in drifts]
            rows = drift_at_checkpoints(drifts, duration)
            report["drift"] = {
                "anchors": len(drifts),
                "median_abs": round(statistics.median(all_abs), 3),
                "p95_abs": round(sorted(all_abs)[int(0.95 * (len(all_abs) - 1))], 3),
                "max_abs": round(max(all_abs), 3),
                "checkpoints": rows,
            }
            print(f"  anchors={len(drifts)}  median|drift|={statistics.median(all_abs):.3f}s  "
                  f"max|drift|={max(all_abs):.3f}s")
            print(f"  {'pct of file':>12s} {'at':>10s} {'anchors':>8s} {'median':>10s} {'max|drift|':>12s}")
            for row in rows:
                median = f"{row['median']:+.3f}s" if row["median"] is not None else "  n/a"
                worst = f"{row['max_abs']:.3f}s" if row["max_abs"] is not None else "  n/a"
                print(f"  {row['fraction'] * 100:11.0f}% {row['at_s']:9.0f}s {row['n']:8d} {median:>10s} {worst:>12s}")
                if row["max_abs"] is not None and row["max_abs"] > args.max_drift:
                    failures.append(
                        f"drift {row['max_abs']:.3f}s at {row['fraction'] * 100:.0f}% of file "
                        f"exceeds --max-drift {args.max_drift}s"
                    )
            growth = [r for r in rows if r["median"] is not None]
            if len(growth) >= 2 and abs(growth[-1]["median"]) > abs(growth[0]["median"]) + 0.25:
                print(f"  NOTE: median drift grows {growth[0]['median']:+.3f}s -> {growth[-1]['median']:+.3f}s "
                      f"across the file — this is the accumulating-drift failure mode.")
    else:
        print("  skipped — need both engines")

    print("\n" + "=" * 72)
    print("3. STRUCTURAL INTEGRITY")
    print("=" * 72)
    for name, result in results.items():
        issues = structural_issues(result)
        report["engines"][name]["structure"] = issues
        hard = (issues["segment_non_monotonic"] + issues["word_non_monotonic"] + issues["segment_overlaps"])
        print(f"  {name:16s} non-monotonic seg={issues['segment_non_monotonic']} word={issues['word_non_monotonic']}  "
              f"overlaps={issues['segment_overlaps']}  zero-length seg={issues['segment_zero_length']} "
              f"word={issues['word_zero_length']}  word-outside-seg={issues['word_outside_segment']}")
        for example in issues["examples"]:
            print(f"  {'':16s} e.g. {example}")
        if hard:
            failures.append(f"{name} has {hard} monotonicity/overlap violations")

    print("\n" + "=" * 72)
    print("4. DOWNSTREAM SPEAKER-MERGE PIPELINE (unmodified app.py code)")
    print("=" * 72)
    if args.diarization and args.diarization.exists():
        turns = json.loads(args.diarization.read_text(encoding="utf-8"))
        print(f"  using {len(turns)} diarization turns from {args.diarization}")
    else:
        longest = max((r.duration for r in results.values()), default=0.0)
        turns = synthetic_turns(longest)
        print(f"  using {len(turns)} synthetic alternating turns (pass --diarization for the real thing)")

    downstream: dict[str, dict] = {}
    for name, result in results.items():
        try:
            downstream[name] = run_downstream(result, turns)
            report["engines"][name]["downstream"] = downstream[name]
            print(f"  {name:16s} OK  {downstream[name]}")
        except Exception as exc:
            failures.append(f"{name} broke the downstream pipeline: {type(exc).__name__}: {exc}")
            print(f"  {name:16s} FAILED  {type(exc).__name__}: {exc}")

    if len(downstream) == 2:
        a, b = downstream["faster_whisper"], downstream["whisper_cpp"]
        print(f"  delta: words {b['words'] - a['words']:+d}  turns {b['turns'] - a['turns']:+d}  "
              f"srt blocks {b['srt_blocks'] - a['srt_blocks']:+d}  speakers {a['speakers']} vs {b['speakers']}")

    print("\n" + "=" * 72)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} issue(s))")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("RESULT: PASS — whisper.cpp output is schema-compatible, word-timed, "
              "monotonic and drift-bounded against faster-whisper.")
    print("=" * 72)

    report["failures"] = failures
    report["result"] = "FAIL" if failures else "PASS"
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.json_out}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
