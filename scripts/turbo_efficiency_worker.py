"""Benchmark-only worker shim. Production never launches this file."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from benchmark_turbo_efficiency import Telemetry, cool_down, experiment, write_json
from chatterbox_turbo_worker import main


def run() -> int:
    """Use the real IPC worker with temporary hooks after reference preparation."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with contextlib.redirect_stdout(sys.stderr):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

    mode = os.environ["RAPHAEL_BENCH_TURBO_MODE"]
    output = Path(os.environ["RAPHAEL_BENCH_TELEMETRY"])
    monitor = Telemetry(85)
    contexts = []
    original = ChatterboxTurboTTS.prepare_conditionals

    def prepare(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        context = experiment(model, mode, monitor)
        context.__enter__()
        contexts.append(context)
        cool_down(monitor, 57)
        return result

    ChatterboxTurboTTS.prepare_conditionals = prepare
    try:
        return main()
    finally:
        monitor.close()
        write_json(output, monitor.rows)
        for context in reversed(contexts):
            context.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(run())
