"""Generate short, offline voice previews on CPU without taking the assistant's GPU."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    """Use the separate clone environment to save WAVs and measured synthesis metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expressiveness", choices=("expressive", "natural", "off"),
                        default="expressive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    from raphael.audio.speech_events import SpeechEventLimiter

    out = args.output or root / "data/voice/expressive-previews" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        model = ChatterboxTurboTTS.from_local(
            str(root / "data/voice/models/chatterbox-turbo"), device="cpu",
        )
        model.prepare_conditionals(
            str(root / "data/voice/references/raphael/reference-1.wav"), exaggeration=0.0,
        )
    report: dict = {"device": "cpu", "expressiveness": args.expressiveness,
                    "seed": 42, "load_seconds": round(time.monotonic() - started, 2),
                    "notes": "Offline synthesis only; speaker playback and quality not judged.",
                    "samples": []}
    for name, text in (
        ("chuckle", "[chuckle] You got me."),
        ("sigh", "[sigh] That's a relief."),
        ("moan-alias", "Oh! [moan] Not again."),
    ):
        normalized = SpeechEventLimiter(args.expressiveness).apply(text)
        started = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            audio = model.generate(normalized).squeeze().numpy()
        if not audio.size or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid preview waveform: {name}")
        sf.write(out / f"{name}.wav", audio, model.sr)
        row = {"name": name, "text": normalized, "sample_rate": model.sr,
               "duration_seconds": round(len(audio) / model.sr, 2),
               "synthesis_seconds": round(time.monotonic() - started, 2)}
        report["samples"].append(row)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
    print(f"Saved previews to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
