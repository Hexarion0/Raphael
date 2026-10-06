"""Check generated words with cached ASR; subjective voice quality remains a listening task."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def tokens(text: str) -> list[str]:
    """Normalize punctuation and the evaluation suite's numeric spellings explicitly."""
    text = text.lower().replace("’", "'")
    text = re.sub(
        r"\b9[\s:.]+30\s*(a\.?\s*m\.?)?",
        lambda match: "nine thirty a m" if match[1] else "nine thirty",
        text,
    )
    text = re.sub(r"\bwork\s+station\b", "workstation", text)
    text = re.sub(r"\balright\b", "all right", text)
    for old, new in {
        "63": "sixty three",
        "4.2": "four point two",
        "9:30": "nine thirty",
        "9.30": "nine thirty",
        "gpu": "g p u",
        "g.p.u.": "g p u",
        "am": "a m",
        "a.m.": "a m",
    }.items():
        text = re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", new, text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def word_distance(expected: list[str], observed: list[str]) -> int:
    """Compute edit distance; retain ASR text so listeners can inspect disagreements."""
    previous = list(range(len(observed) + 1))
    for i, first in enumerate(expected, 1):
        current = [i]
        for j, second in enumerate(observed, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (first != second))
            )
        previous = current
    return previous[-1]


def main() -> None:
    """Run separately from timed synthesis, without silently fetching ASR models."""
    import numpy as np
    from faster_whisper import WhisperModel

    from raphael.audio.stt import _preload_cuda_libraries
    from raphael.audio.voice_references import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--checks-name", default="word_checks.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if Path(args.checks_name).name != args.checks_name or not args.checks_name.endswith(".json"):
        parser.error("--checks-name must be a JSON filename within each benchmark directory")
    if args.device == "cuda":
        _preload_cuda_libraries()
    model = WhisperModel(
        str(args.asr_model), device=args.device,
        compute_type="int8_float16" if args.device == "cuda" else "int8",
        local_files_only=True, cpu_threads=4,
    )
    for directory in args.directories:
        measurements = json.loads((directory / "measurements.json").read_text())
        destination = directory / args.checks_name
        existing = json.loads(destination.read_text()) if destination.exists() else []
        completed = {r["audio"] for r in existing}
        for row in measurements:
            if row["audio"] in completed:
                continue
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(directory / row["audio"]),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
            iterator, _ = model.transcribe(
                np.frombuffer(decoded.stdout, dtype=np.float32),
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            segments = list(iterator)
            transcript = " ".join(s.text.strip() for s in segments)
            spoken_text = row.get("spoken_text", row["text"])
            expected, observed = tokens(spoken_text), tokens(transcript)
            edits = word_distance(expected, observed)
            existing.append(
                {
                    "audio": row["audio"],
                    "asr_model": str(args.asr_model),
                    "asr_device": args.device,
                    "expected": spoken_text,
                    "input_text": row["text"],
                    "observed": transcript,
                    "word_edits": edits,
                    "expected_word_count": len(expected),
                    "wer_proxy": edits / max(1, len(expected)),
                    "interpretation": (
                        "ASR disagreement proxy, not human correctness or naturalness"
                    ),
                }
            )
            atomic_json(destination, existing)
            print(directory.name, row["audio"], f"word edits={edits}", flush=True)


if __name__ == "__main__":
    main()
