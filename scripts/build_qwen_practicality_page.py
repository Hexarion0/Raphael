"""Assemble offline Qwen A/B pages and measured-gap listening replays."""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmark_cloning_voice import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def view(output: Path, sources: list[Path], label: str, ids: set[str] | None = None) -> Path:
    """Link fixed archived takes without presenting a combined view as a new GPU run."""
    output.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((source / "summary.json").read_text()) for source in sources]
    if len({s["reference_sha256"] for s in summaries}) != 1:
        raise ValueError("A/B view sources use different references")
    rows, checks = (
        [],
        {
            name: []
            for name in [
                "word_checks.json",
                "word_checks-medium.json",
                "similarity-chatterbox.json",
                "similarity-campplus.json",
            ]
        },
    )
    for source in sources:
        selected = [
            r
            for r in json.loads((source / "measurements.json").read_text())
            if (ids is None or r["id"] in ids)
            and (
                (r["phase"] == "warm" and r["repeat"] <= 2)
                or (source == sources[0] and r["phase"] == "first_inference")
            )
        ]
        names = {r["audio"] for r in selected}
        for row in selected:
            rows.append(row | {"audio": os.path.relpath(source / row["audio"], output)})
        for name, collected in checks.items():
            if not (source / name).exists():
                continue
            for row in json.loads((source / name).read_text()):
                if row["audio"] in names:
                    collected.append(
                        row | {"audio": os.path.relpath(source / row["audio"], output)}
                    )
    warm = [r for r in rows if r["phase"] == "warm"]
    summary = summaries[0] | {
        "display_name": label,
        "view_only": True,
        "source_runs": [str(p) for p in sources],
        "planned_sentence_ids": sorted({r["id"] for r in rows}),
        "warm_median_audio_ready_seconds": statistics.median(
            r["audio_ready_seconds"] for r in warm
        ),
        "warm_median_rtf": statistics.median(r["rtf"] for r in warm),
    }
    for key in ["sampled_process_vram_mib", "sampled_whole_gpu_mib", "peak_rss_mib"]:
        summary[key] = max(s[key] for s in summaries)
    write_json(output / "summary.json", summary)
    write_json(output / "measurements.json", rows)
    for name, collected in checks.items():
        if collected:
            write_json(output / name, collected)
    return output


def main() -> None:
    """Expose complete takes, separate text/precision experiments, and honest timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments", type=Path, default=ROOT / "data/voice/benchmarks/qwen-optimized"
    )
    args = parser.parse_args()
    folder = args.experiments.resolve()
    benchmark = ROOT / "data/voice/benchmarks"
    stock = view(
        folder / "stock-view",
        [benchmark / "native-cloning/qwen06", benchmark / "emotion-controls/qwen-emotions"],
        "Qwen stock BF16 / archived",
    )
    controls = {"01_of_course", "16_status_short", "20_happy_short", "21_playful_short"}
    short_default = view(
        folder / "short-default-view",
        [folder / "bf16-graph", folder / "short-chunks"],
        "Default conditioning / short controls",
        controls,
    )
    short_full = view(
        folder / "short-full-view",
        [folder / "full-text-mode", folder / "short-full-text"],
        "Official full-text mode / short controls",
        controls,
    )

    def page(filename: str, directories: list[Path], only_ids: list[str] | None = None):
        command = [
            sys.executable,
            str(ROOT / "scripts/build_voice_comparison.py"),
            *map(str, directories),
            "--references",
            str(ROOT / "data/voice/references/raphael/references.json"),
            "--output",
            str(folder / filename),
        ]
        if only_ids:
            command.extend(["--only-ids", *only_ids])
        subprocess.run(command, check=True)
        path = folder / filename
        content = path.read_text().replace(
            "RAPHAEL · native voice cloning", "RAPHAEL · optimized Qwen"
        )
        nav = '<p><a href="index.html">Main A/B</a> · <a href="short.html">Short replies</a> · '
        nav += '<a href="words.html">Text/sampling</a> · '
        nav += '<a href="short-controls.html">EOS controls</a> · '
        nav += '<a href="screens.html">Precision/attention screens</a></p>'
        content = content.replace("<h2>Source references</h2>", nav + "<h2>Source references</h2>")
        path.write_text(content)

    page(
        "index.html",
        [
            benchmark / "emotion-controls/turbo-neutral",
            stock,
            folder / "bf16-graph",
            folder / "full-text-mode",
            folder / "graph-stream12",
            folder / "graph-stream24",
        ],
    )
    page(
        "short.html",
        [folder / "short-chunks", folder / "short-full-text"],
        [
            f"{i}_{label}"
            for i, label in [
                (16, "status_short"),
                (17, "caring_short"),
                (18, "concerned_short"),
                (19, "serious_short"),
                (20, "happy_short"),
                (21, "playful_short"),
                (22, "sleepy_short"),
                (23, "annoyed"),
            ]
        ],
    )
    page(
        "words.html",
        [folder / "bf16-graph", folder / "text-normalized", folder / "temperature-08"],
        ["07_gpu", "08_long", "15_sleepy"],
    )
    page(
        "short-controls.html", [short_default, short_full, folder / "minimum-eos"], sorted(controls)
    )
    screens = [
        "screen-stock",
        "screen-compact",
        "screen-graph",
        "screen-eager",
        "screen-mixed",
        "screen-mixed-graph",
        "screen-fp32",
        "screen-codec-fp32",
        "screen-codec-fp16",
    ]
    page(
        "screens.html",
        [folder / name for name in screens],
        ["01_of_course", "02_server", "09_comforting"],
    )
    parts = [
        "<h2>Timing replays · hear the measured pauses</h2>",
        "<p>These replays insert silence from measured chunk arrivals. They omit the initial "
        "wait, which is shown below. They are calculated schedules; the overlap probe used a "
        "clock-paced CPU sink, without opening an audio device. All paragraph text was "
        "available to synthesis; later LLM text could add further delay.</p>",
    ]
    for name, source, paragraph in [
        ("stream12-long", "graph-stream12", False),
        ("stream24-long", "graph-stream24", False),
        ("sentence-pipeline-long", "short-chunks", True),
    ]:
        output = folder / "replays" / (name + ".wav")
        if not output.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts/render_voice_timing.py"),
                str(folder / source),
                "--output",
                str(output),
            ]
            if paragraph:
                command.append("--paragraph")
            subprocess.run(command, check=True)
        stats = json.loads(output.with_suffix(".json").read_text())
        label = html.escape(name.replace("-", " "))
        parts.append(
            f"<p><b>{label}</b> · first audio "
            f"{stats['initial_wait_excluded_from_wav_seconds']:.2f}s · gaps "
            f"{stats['total_gap_seconds']:.2f}s · finish "
            f"{stats['request_to_playback_finish_seconds']:.2f}s</p>"
            f'<audio controls preload="none" src="replays/{name}.wav"></audio>'
        )
    parts.append(
        "<p>The native BF16 predictor-graph and both corrected streaming paths were "
        "verified against the stock PCM sample for sample. Full-text, temperature, EOS, "
        "and precision variants change outputs: judge their emotion and identity by "
        "listening. Main medians cover different take counts shown in the table; "
        "the written report uses matched sentences/seeds for speed comparisons.</p>"
    )
    index = folder / "index.html"
    content = index.read_text().replace(
        "These WAVs are already generated, so playback here has no synthesis gaps.",
        "Main A/B WAVs play without generation waits. Timing replays below include pauses.",
    )
    index.write_text(content.replace("<script>", "\n".join(parts) + "<script>", 1))
    write_json(
        folder / "listening-page.json",
        {
            "pages": [
                "index.html",
                "short.html",
                "words.html",
                "short-controls.html",
                "screens.html",
            ],
            "reference_sha256": file_hash(ROOT / "data/voice/references/raphael/reference-1.wav"),
            "training": False,
            "physical_playback_measured": False,
        },
    )


if __name__ == "__main__":
    main()
