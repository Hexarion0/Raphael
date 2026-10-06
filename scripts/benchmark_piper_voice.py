"""Benchmark installed Piper voices offline, without playback or model downloads.

Run each voice in a fresh process. This measures the existing RAPHAEL synthesis
path; it is the baseline, not a multi-model voice-cloning benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    """Return a streaming SHA256 without loading model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Measure cold loading, first inference, and repeated warm synthesis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", required=True, help="Installed ONNX filename without suffix")
    parser.add_argument("--models", type=Path, default=ROOT / "models/tts")
    parser.add_argument("--suite", type=Path, default=ROOT / "docs/voice-evaluation.json")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--alignments",
        action="store_true",
        help="Include caption alignment overhead",
    )
    args = parser.parse_args()
    if not args.voice or Path(args.voice).name != args.voice or args.voice in {".", ".."}:
        parser.error("--voice must be a model basename")
    if not math.isfinite(args.speed) or not 0.2 <= args.speed <= 3.0 or args.repeats < 1:
        parser.error("--speed must be between 0.2 and 3.0; --repeats must be positive")
    model = args.models / f"{args.voice}.onnx"
    config = model.with_suffix(".onnx.json")
    if not config.is_file():
        config = model.with_suffix(".json")
    if not model.is_file() or not config.is_file():
        parser.error("Both local ONNX and config must exist; this script never downloads models")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    ids = [item["id"] for item in suite]
    if (
        not suite
        or len(set(ids)) != len(ids)
        or any(
            not item["text"].strip()
            or Path(item["id"]).name != item["id"]
            or item["id"] in {".", ".."}
            for item in suite
        )
    ):
        parser.error("Suite must have unique safe IDs and nonempty text")
    # Never overwrite earlier benchmark evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    model_sha = file_hash(model)
    config_sha = file_hash(config)
    dependency_started = time.perf_counter()
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(ROOT / "src"))
    from raphael.audio.tts import TextToSpeech

    dependency_seconds = time.perf_counter() - dependency_started
    load_started = time.perf_counter()
    # A direct existing model path bypasses download_voice entirely.
    tts = TextToSpeech(
        voice_name=str(model.resolve()),
        engine="piper",
        speed=args.speed,
        include_alignments=args.alignments,
        enabled=True,
    )
    load_seconds = time.perf_counter() - load_started
    if not tts.enabled or tts._voice is None:
        raise RuntimeError("Installed Piper model failed to load")
    ready_seconds = time.perf_counter() - PROCESS_STARTED
    original_synthesize = tts._voice.synthesize
    first_chunk: list[float | None] = [None]

    def measured_chunks(*positional, **keywords):
        """Observe native chunks without changing RAPHAEL's buffering behavior."""
        for chunk in original_synthesize(*positional, **keywords):
            if first_chunk[0] is None and chunk.audio_float_array is not None:
                if chunk.audio_float_array.size:
                    first_chunk[0] = time.perf_counter()
            yield chunk

    tts._voice.synthesize = measured_chunks
    rows = []
    for phase, sentences, repeats in [
        ("first_inference", suite[:1], 1),
        ("warm", suite, args.repeats),
    ]:
        for item in sentences:
            for repeat in range(1, repeats + 1):
                first_chunk[0] = None
                started = time.perf_counter()
                row = {
                    "phase": phase,
                    "id": item["id"],
                    "text": item["text"],
                    "repeat": repeat,
                    "error": None,
                }
                try:
                    result = tts.synthesize(item["text"])
                    finished = time.perf_counter()
                    if result is None:
                        raise RuntimeError("No audio generated")
                    audio, rate = result
                    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                        raise RuntimeError("Invalid generated audio")
                    duration = audio.size / rate
                    elapsed = finished - started
                    name = f"{phase}-{item['id']}-{repeat:02d}.wav"
                    sf.write(args.output / name, audio, rate, subtype="PCM_16")
                    row.update(
                        audio=name,
                        duration_seconds=duration,
                        sample_rate=rate,
                        generation_seconds=elapsed,
                        rtf=elapsed / duration,
                        native_first_chunk_seconds=(
                            first_chunk[0] - started if first_chunk[0] else None
                        ),
                        raphael_audio_ready_seconds=elapsed,
                        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        peak_vram_mib=None,
                        naturalness=None,
                        similarity=None,
                        consistency=None,
                    )
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                # Persist every observation so an interrupted run remains inspectable.
                (args.output / "measurements.json").write_text(
                    json.dumps(rows, indent=2) + "\n",
                    encoding="utf-8",
                )
    successful = [r for r in rows if r["phase"] == "warm" and not r["error"]]
    summary = {
        "voice": args.voice,
        "speed": args.speed,
        "alignments": args.alignments,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("piper-tts", "onnxruntime", "numpy", "soundfile")
        },
        "model_sha256": model_sha,
        "config_sha256": config_sha,
        "suite_sha256": file_hash(args.suite),
        "model_bytes": model.stat().st_size,
        "dependency_import_seconds": dependency_seconds,
        "cold_load_seconds": load_seconds,
        "script_to_ready_seconds": ready_seconds,
        "script_to_ready_note": (
            "Includes hashing, excludes Python interpreter startup; OS cache retained"
        ),
        "streaming": (
            "Native Piper chunks observed; RAPHAEL buffers the whole input before playback"
        ),
        "device": "cpu",
        "vram_note": "No CUDA allocation by Piper; peak GPU usage not sampled",
        "ram_note": "Linux process RSS high-water mark includes dependencies and accumulated audio",
        "playback": "Not measured; no sound device opened",
        "errors": sum(bool(row["error"]) for row in rows),
        "warm_median_seconds": (
            statistics.median(r["generation_seconds"] for r in successful) if successful else None
        ),
        "warm_median_rtf": (
            statistics.median(r["rtf"] for r in successful) if successful else None
        ),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "listening.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["audio", "naturalness_1_5", "similarity_1_5", "consistency_1_5", "notes"])
        writer.writerows([row["audio"], "", "", "", ""] for row in successful)
    cards = []
    for row in rows:
        if not row["error"]:
            cards.append(
                f"<p>{html.escape(row['phase'])} / {html.escape(row['text'])} / "
                f"take {row['repeat']}</p><audio controls preload='none' "
                f"src='{html.escape(row['audio'], quote=True)}'></audio>"
            )
    (args.output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Piper baseline</title>"
        f"<h1>{html.escape(args.voice)} — speed {args.speed}</h1>"
        "<p>Local baseline samples. Quality ratings require human listening.</p>"
        + "\n".join(cards),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
