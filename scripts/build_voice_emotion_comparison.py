"""Publish focused offline listening pages for the completed emotion experiment."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/voice/benchmarks/emotion-controls"


def combine_qwen() -> Path:
    """Link previous baseline takes and new emotion takes without copying audio."""
    directories = [
        ROOT / "data/voice/benchmarks/native-cloning/qwen06",
        OUTPUT / "qwen-emotions",
    ]
    destination = OUTPUT / "qwen-combined-view"
    destination.mkdir(exist_ok=True)
    rows, checks = [], []
    for directory in directories:
        rows.extend(
            {**r, "audio": str((directory / r["audio"]).resolve())}
            for r in json.loads((directory / "measurements.json").read_text())
        )
        if (directory / "word_checks.json").exists():
            checks.extend(
                {**r, "audio": str((directory / r["audio"]).resolve())}
                for r in json.loads((directory / "word_checks.json").read_text())
            )
    summary = json.loads((directories[-1] / "summary.json").read_text())
    warm = [r for r in rows if r["phase"] == "warm"]
    summary.update(
        display_name="Qwen Base · previous baseline + new emotion texts",
        warm_median_audio_ready_seconds=statistics.median(r["audio_ready_seconds"] for r in warm),
        warm_median_rtf=statistics.median(r["rtf"] for r in warm),
        view_only=True,
        source_runs=[str(d) for d in directories],
    )
    for filename, data in [
        ("summary.json", summary), ("measurements.json", rows), ("word_checks.json", checks),
    ]:
        (destination / filename).write_text(json.dumps(data, indent=2) + "\n")
    return destination


def main() -> None:
    """Preserve every take; keep separate experiments easy to compare."""
    primary = json.loads((ROOT / "data/voice/references/raphael/references.json").read_text())[:1]
    alternates = json.loads(
        (ROOT / "data/voice/references/raphael/emotion/references.json").read_text()
    )
    references = OUTPUT / "listening-references.json"
    references.write_text(json.dumps(primary + alternates, indent=2) + "\n")
    qwen = combine_qwen()
    groups = {
        "main": [
            "turbo-neutral", "turbo-reference-pleased", "turbo-emotion-tags",
            "original-default", "original-moderate", qwen.name,
        ],
        "original-levels": [f"original-{n}" for n in ["default", "subtle", "moderate", "strong"]],
        "turbo-references": [
            "turbo-neutral", *[f"turbo-reference-{n}" for n in ["pleased", "playful", "caring"]],
        ],
        "turbo-tags": ["turbo-tags-neutral", "turbo-tags"],
        "turbo-punctuation": ["turbo-punct-neutral", "turbo-punct-expressive"],
        "turbo-sampling": ["turbo-neutral", "turbo-temperature-0.6", "turbo-temperature-1.0"],
    }
    labels = {
        "main": "Main A/B comparison", "original-levels": "Original · four levels",
        "turbo-references": "Turbo · references", "turbo-tags": "Turbo · tags",
        "turbo-punctuation": "Turbo · punctuation", "turbo-sampling": "Turbo · temperature",
    }
    links = " · ".join(f'<a href="{name}.html">{labels[name]}</a>' for name in groups)
    explanation = (
        '<nav style="position:sticky;top:0;background:#101722;padding:15px;z-index:2">'
        + links + '</nav><p class="warning">Emotion names describe the requested delivery, '
        'not a proven result. Original profiles use exaggeration/CFG; Turbo does not support '
        'those controls. In the tags page, sigh/laugh/chuckle/gasp are documented events. '
        'Whispering/happy/sarcastic are tokenizer-only exploratory probes, not documented '
        'reliable emotion controls. [whisper] and [breath] were not used. Reference names '
        'describe the passage content; listen to judge its actual delivery. '
        'Qwen reuses earlier baseline takes and adds new emotion-text takes. Performance '
        'medians in each page reflect its available texts; use the report for matched-text '
        'model comparisons.</p>'
    )
    for name, directories in groups.items():
        output = OUTPUT / f"{name}.html"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/build_voice_comparison.py"),
            *[str(OUTPUT / d) for d in directories],
            "--references", str(references), "--output", str(output),
        ], check=True)
        page = output.read_text()
        page = page.replace("<h1>RAPHAEL", explanation + "<h1>RAPHAEL", 1)
        page = page.replace("<h1>RAPHAEL · native voice cloning</h1>",
                            "<h1>RAPHAEL · voice emotion listening</h1>")
        page = page.replace(
            "<title>RAPHAEL — native voice cloning comparison</title>",
            "<title>RAPHAEL — voice emotion listening</title>",
        )
        output.write_text(page)
    (OUTPUT / "index.html").write_text((OUTPUT / "main.html").read_text())
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
