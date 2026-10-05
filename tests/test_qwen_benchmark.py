"""Protect benchmark evidence and check overlap scheduling without GPU/audio devices."""

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location(
    "voice_pipeline_probe", SCRIPTS / "voice_pipeline_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_slower_synthesis_has_gaps_despite_playback_overlap():
    result = probe.playback_schedule([2.0, 4.0, 6.0], [1.0, 1.0, 1.0])
    assert [part["start"] for part in result] == [2.0, 4.0, 6.0]
    assert sum(part["gap_before_seconds"] for part in result) == 2.0


def test_audio_ready_during_previous_playback_waits_in_queue():
    result = probe.playback_schedule([2.0, 2.5, 3.0], [2.0, 2.0, 1.0])
    assert [part["start"] for part in result] == [2.0, 4.0, 6.0]
    assert sum(part["gap_before_seconds"] for part in result) == 0


def test_invalid_chunk_timing_is_rejected():
    with pytest.raises(ValueError):
        probe.playback_schedule([1.0], [])
    with pytest.raises(ValueError):
        probe.playback_schedule([1.0], [0.0])


def test_paced_sink_cancels_duration_wait_and_releases_thread():
    sink = probe.PacedSink()
    sink.offer("long.wav", bytes(100), 1)
    # Cancellation is immediate even if a very long duration wait has begun.
    sink.stop()
    assert not sink.thread.is_alive()
    assert not any(t is sink.thread for t in threading.enumerate())


def test_existing_qwen_benchmark_stops_before_importing_torch(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("previous result")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "benchmark_qwen_practicality.py"),
            "--output",
            str(output),
            "--label",
            "test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "never overwrite" in result.stderr
    assert sentinel.read_text() == "previous result"
