"""Guard the isolated benchmark's thermal stops and incomplete-run reporting."""

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_turbo_efficiency.py"
SPEC = importlib.util.spec_from_file_location("turbo_efficiency_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize("rows", [[], [{"time": 0, "temperature_c": 40}]])
def test_guard_refuses_missing_or_stale_telemetry(rows):
    monitor = object.__new__(benchmark.Telemetry)
    monitor.rows, monitor.cutoff = rows, 85
    with pytest.raises(RuntimeError, match="telemetry unavailable"):
        monitor.guard()


def test_guard_stops_at_cutoff():
    monitor = object.__new__(benchmark.Telemetry)
    monitor.rows = [{"time": time.perf_counter(), "temperature_c": 85}]
    monitor.cutoff = 85
    with pytest.raises(RuntimeError, match="Thermal stop"):
        monitor.guard()


def test_failed_repeated_workload_keeps_partial_result(monkeypatch, tmp_path):
    calls = []

    def generate(*args):
        calls.append(args)
        if len(calls) > 1:
            raise RuntimeError("Thermal stop at 85 C")
        return {"audio_seconds": 0.01, "seconds": 0.002}

    monkeypatch.setattr(benchmark, "generate", generate)
    result = benchmark.repeat_pipeline(None, "baseline", SimpleNamespace(rows=[]), tmp_path, 1)
    assert result["error"] == "RuntimeError: Thermal stop at 85 C"
    assert len(result["rows"]) == 1
    assert result["audio_seconds"] == 0.01
    assert result["first_audio_ready_seconds"] is not None


def test_failed_before_audio_does_not_report_zero_rtf(monkeypatch, tmp_path):
    def generate(*_args):
        raise RuntimeError("Thermal stop at 85 C")

    monkeypatch.setattr(benchmark, "generate", generate)
    result = benchmark.repeat_pipeline(None, "baseline", SimpleNamespace(rows=[]), tmp_path, 1)
    assert result["rtf"] is None
    assert result["first_audio_ready_seconds"] is None
    assert result["error"]
