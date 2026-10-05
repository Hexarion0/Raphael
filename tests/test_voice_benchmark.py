"""Protect offline benchmark runs from accidental downloads and overwritten evidence."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_piper_voice.py"


def test_missing_model_stops_before_creating_output(tmp_path):
    output = tmp_path / "results"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--voice", "uninstalled", "--models", str(tmp_path),
         "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "this script never downloads models" in result.stderr
    assert not output.exists()


def test_existing_results_are_never_overwritten(tmp_path):
    # These invalid model placeholders should never reach model loading.
    (tmp_path / "existing.onnx").write_bytes(b"not a model")
    (tmp_path / "existing.onnx.json").write_text("{}")
    output = tmp_path / "results"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("previous benchmark")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--voice", "existing", "--models", str(tmp_path),
         "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert sentinel.read_text() == "previous benchmark"
    assert sorted(p.name for p in output.iterdir()) == ["summary.json"]
