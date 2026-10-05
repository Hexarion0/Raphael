"""Run a selected experiment through real RAPHAEL chunking, IPC and playback.

Uses a fixture LLM stream and the unchanged production pipeline. Only this TTS
instance points at the benchmark worker. Does not update .env or source settings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
from benchmark_raphael_turbo_runtime import FixedStream
from benchmark_turbo_efficiency import write_json

from raphael.audio.streaming import stream_reply
from raphael.audio.tts import TextToSpeech
from raphael.providers.base import ChatMessage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "pace12", "pace16"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["RAPHAEL_BENCH_TURBO_MODE"] = args.mode
    os.environ["RAPHAEL_BENCH_TELEMETRY"] = str(out / "telemetry.json")
    tts = TextToSpeech()
    if tts.engine != "chatterbox_turbo":
        raise RuntimeError("Current configured production TTS must be Chatterbox Turbo")
    worker = tts._load_chatterbox_worker()
    worker.script = root / "scripts/turbo_efficiency_worker.py"
    rows, current = [], [None]
    original = tts.speak

    def save_speak(text, *args, _prepared=None, **kwargs):
        if _prepared is not None:
            index = len(current[0]["audio"])
            filename = f"{current[0]['name']}-{index}.wav"
            sf.write(out / filename, _prepared[0], _prepared[1], subtype="FLOAT")
            current[0]["audio"].append(filename)
        return original(text, *args, _prepared=_prepared, **kwargs)

    tts.speak = save_speak
    try:
        started = time.perf_counter()
        if not tts.warmup():
            raise RuntimeError("Turbo benchmark worker failed")
        startup = {"seconds_including_cooldown": time.perf_counter() - started,
                   "worker": worker.startup_metrics}
        for name, text in [
            ("short", "Of course."),
            ("multi", "Your server is online and everything looks normal. "
             "I checked the important services, and they are responding as expected. "
             "I will keep monitoring the machine and let you know if anything changes. "
             "You can continue with what you were doing."),
        ]:
            # Match every run's cooldown, without touching the production worker.
            time.sleep(15)
            row = {"name": name, "text": text, "audio": []}
            current[0] = row
            epoch = time.perf_counter()
            reply = stream_reply(FixedStream(text), [ChatMessage("user", "benchmark")],
                                 tts, lambda: True, max_tokens=300)
            row.update(first_audio_seconds=reply.first_audio_seconds,
                       total_seconds=time.perf_counter() - epoch,
                       pipeline=reply.speech_pipeline, fallback=tts._chatterbox_failed)
            for sentence, filename in zip(row["pipeline"]["sentences"], row["audio"]):
                duration = sf.info(out / filename).duration
                sentence["audio_seconds"] = duration
                sentence["rtf"] = sentence["synthesis_seconds"] / duration
            rows.append(row)
            write_json(out / "runtime.json", {"mode": args.mode, "startup": startup,
                                               "cases": rows})
            print(json.dumps(row), flush=True)
            if tts._chatterbox_failed:
                break
    finally:
        tts.close()


if __name__ == "__main__":
    main()
