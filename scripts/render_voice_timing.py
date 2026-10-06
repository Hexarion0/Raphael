"""Replay measured synthesis gaps in WAVs; these are timing simulations, not recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_cloning_voice import file_hash, write_json
from voice_pipeline_probe import playback_schedule


def main() -> None:
    """Preserve generated PCM and add only silence corresponding to observed arrivals."""
    import numpy as np
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentence", default="08_long")
    parser.add_argument("--paragraph", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Replay output already exists; preserve the existing evidence")
    rows = json.loads((args.directory / "measurements.json").read_text())
    if args.paragraph:
        selected = [
            r
            for r in rows
            if r["phase"] == "warm" and r["repeat"] == 1 and r["id"].startswith("24_long_chunk_")
        ]
        observed = json.loads((args.directory / "paced-playback.json").read_text())["events"]
        ready = {row["audio"]: row["ready_seconds"] for row in observed}
        origin = selected[0]["synthesis_start_clock_seconds"]
        arrivals = [ready[row["audio"]] - origin for row in selected]
        waves = [sf.read(args.directory / row["audio"], dtype="float32")[0] for row in selected]
    else:
        selected = [
            r
            for r in rows
            if r["id"] == args.sentence and r["phase"] == "warm" and r["repeat"] == 1
        ]
        if len(selected) != 1:
            raise ValueError("Expected exactly one fixed warm take")
        row = selected[0]
        wave, rate = sf.read(args.directory / row["audio"], dtype="float32")
        if rate != 24000:
            raise ValueError("Expected 24 kHz Qwen output")
        lengths = [round(duration * rate) for duration in row["chunk_duration_seconds"]]
        if sum(lengths) != len(wave):
            raise ValueError("Chunk durations do not account for the complete waveform")
        cuts = np.cumsum([0, *lengths])
        waves = [wave[start:end] for start, end in zip(cuts[:-1], cuts[1:])]
        arrivals = row["chunk_arrival_seconds"]
    durations = [len(wave) / 24000 for wave in waves]
    scheduled = playback_schedule(arrivals, durations)
    first = scheduled[0]["start"]
    result = np.zeros(round((scheduled[-1]["end"] - first) * 24000), dtype=np.float32)
    for event, wave in zip(scheduled, waves):
        start = round((event["start"] - first) * 24000)
        result[start : start + len(wave)] = wave
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, result, 24000, subtype="PCM_24")
    write_json(
        args.output.with_suffix(".json"),
        {
            "description": "Timing replay calculated from measured arrivals; no physical playback",
            "initial_wait_excluded_from_wav_seconds": first,
            "total_gap_seconds": sum(event["gap_before_seconds"] for event in scheduled),
            "request_to_playback_finish_seconds": scheduled[-1]["end"],
            "speech_duration_seconds": sum(durations),
            "schedule": scheduled,
            "source_audio_sha256": {
                r["audio"]: file_hash(args.directory / r["audio"]) for r in selected
            },
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
