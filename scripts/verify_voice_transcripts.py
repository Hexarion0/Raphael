"""Cross-check provisional reference transcripts with a second cached local ASR model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Retain both decoding results and flag disagreements instead of inventing words."""
    import numpy as np
    from faster_whisper import WhisperModel

    from raphael.audio.stt import _preload_cuda_libraries
    from raphael.audio.voice_references import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    _preload_cuda_libraries()
    model = WhisperModel(
        str(args.model), device="cuda", compute_type="int8_float16", local_files_only=True
    )
    references = json.loads(args.references.read_text())
    for c in references:
        audio = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                c["reference_wav"],
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
            np.frombuffer(audio.stdout, dtype=np.float32),
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=False,
        )
        segments = list(iterator)
        text = " ".join(s.text.strip() for s in segments)
        agreed = re.findall(r"[a-z0-9]+", text.lower()) == re.findall(
            r"[a-z0-9]+",
            c["text"].lower(),
        )
        c.update(
            verification_asr_model=str(args.model),
            verification_asr_text=text,
            transcript_models_agree=agreed,
            verification_asr_avg_logprob=float(np.mean([s.avg_logprob for s in segments])),
            transcript_status=(
                "two ASR models agree; human verification pending"
                if agreed
                else "ASR disagreement; human transcript correction required"
            ),
        )
        if not agreed:
            c["reasons"].append("transcript_crosscheck_disagreement")
        atomic_json(Path(c["reference_wav"]).with_suffix(".json"), c)
        print(Path(c["reference_wav"]).name, "AGREE" if agreed else "DISAGREE", flush=True)
        print("primary:", c["text"], flush=True)
        print("second:", text, flush=True)
    atomic_json(args.references, references)


if __name__ == "__main__":
    main()
