"""Fill the eight previously unscheduled Turbo tag comparisons using documented events."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/voice/benchmarks/emotion-controls"


def documented_tags() -> set[str]:
    """Read the actual event list from the pinned official demo, without importing Gradio."""
    source = ROOT / "data/voice/vendor/chatterbox/gradio_tts_turbo_app.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EVENT_TAGS" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise ValueError("The pinned official Turbo demo has no event-tag list")


def baseline_tag_suite(
    baseline: list[dict] | None = None,
    allowed: set[str] | None = None,
    native: dict | None = None,
) -> list[dict]:
    """Keep every spoken word unchanged; vary only a documented event prefix."""
    if baseline is None:
        baseline = json.loads((ROOT / "docs/voice-evaluation.json").read_text())
    prefixes = ["chuckle", "sigh", "sigh", "sigh", "gasp", "chuckle", "sigh", "sigh"]
    if len(baseline) != len(prefixes):
        raise ValueError("Review tag assignments after changing the baseline suite")
    if allowed is None:
        allowed = documented_tags()
    if native is None:
        token_path = ROOT / "data/voice/models/chatterbox-turbo/added_tokens.json"
        native = json.loads(token_path.read_text())
    rows = []
    for item, prefix in zip(baseline, prefixes):
        tag = f"[{prefix}]"
        if tag not in allowed or tag not in native:
            raise ValueError(f"Not a documented native Turbo event: {tag}")
        rows.append({**item, "spoken_text": item["text"], "text": tag + " " + item["text"]})
    return rows


def main() -> None:
    """Benchmark once, preserving all existing tag and reference experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    reference = ROOT / "data/voice/references/raphael/reference-1.wav"
    previous = json.loads((OUTPUT / "turbo-emotion-tags/summary.json").read_text())
    if hashlib.sha256(reference.read_bytes()).hexdigest() != previous["reference_sha256"]:
        raise ValueError("The primary reference changed; do not silently change this comparison")
    suite = OUTPUT / "supported-tags-baseline-suite.json"
    suite.write_text(json.dumps(baseline_tag_suite(), indent=2) + "\n")
    destination = OUTPUT / "turbo-tags-baseline"
    if (destination / "summary.json").exists():
        summary = json.loads((destination / "summary.json").read_text())
        rows = json.loads((destination / "measurements.json").read_text())
        if not summary["errors"] and len([r for r in rows if r["phase"] == "warm"]) == 8:
            print("Already complete:", destination)
            return
        raise RuntimeError(f"Inspect incomplete attempt before resuming: {destination}")
    command = [
        str(ROOT / "data/voice/envs/clone/bin/python"),
        str(ROOT / "scripts/benchmark_cloning_voice.py"),
        "--candidate", "chatterbox-turbo",
        "--model", str(ROOT / "data/voice/models/chatterbox-turbo"),
        "--reference", str(reference), "--reference-text", str(reference.with_suffix(".txt")),
        "--suite", str(suite), "--output", str(destination), "--repeats", "1",
        "--label", "Turbo · documented events, original eight sentences",
    ]
    print(json.dumps(command), flush=True)
    if not args.plan_only:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
