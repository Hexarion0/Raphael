"""Prepare a source/reference review bundle in one command using installed local tools."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cached_asr(name: str) -> Path:
    """Resolve an existing pinned HF cache snapshot without downloading anything."""
    hub = Path.home() / ".cache/huggingface/hub"
    directory = hub / ("models--Systran--" + name)
    main = directory / "refs/main"
    if main.is_file():
        snapshot = directory / "snapshots" / main.read_text().strip()
        if (snapshot / "model.bin").is_file():
            return snapshot
    candidates = [p.parent for p in (directory / "snapshots").glob("*/model.bin")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Specify a local ASR path: cannot uniquely resolve cached {name}")


def main() -> None:
    """Run stages sequentially so ASR frees the GPU before CPU reference scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/voice")
    parser.add_argument("--urls", type=Path)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--verification-model", type=Path)
    parser.add_argument("--analysis-python", type=Path)
    parser.add_argument("--yt-dlp", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    asr = args.asr_model or cached_asr("faster-distil-whisper-large-v3")
    verification = args.verification_model or cached_asr("faster-whisper-medium.en")
    python = args.analysis_python or root / "envs/clone/bin/python"
    urls = args.urls or root / "source_urls.txt"
    encoder = root / "models/chatterbox-turbo/ve.safetensors"
    panns = root / "models/analysis/panns-cnn6.pth"
    vendor = root / "vendor/panns"
    required = [urls, asr / "model.bin", verification / "model.bin", python, encoder, panns]
    if any(not p.is_file() for p in required) or not vendor.is_dir():
        parser.error(
            "Install the documented local preparation tools/models first; no auto-installs"
        )
    downloader = args.yt_dlp or root / "vendor/yt-dlp"
    prepare = [
        sys.executable,
        str(ROOT / "scripts/prepare_voice_references.py"),
        "--urls",
        str(urls),
        "--root",
        str(root),
        "--asr-model",
        str(asr),
    ]
    if downloader.is_file():
        prepare += ["--yt-dlp", str(downloader)]
    environment = {
        **os.environ,
        "HF_HOME": str(root / "cache/huggingface"),
        "NUMBA_CACHE_DIR": str(root / "cache/numba"),
    }
    subprocess.run(prepare, check=True, cwd=ROOT, env=environment)
    from raphael.audio.voice_references import atomic_json, read_source_urls

    clips = []
    for video_id, _ in read_source_urls(urls):
        manifest = root / "reference_candidates" / video_id / "shortlist.json"
        clips.extend(json.loads(manifest.read_text()))
    clips.sort(key=lambda clip: clip["signal_score"], reverse=True)
    shortlist = root / "reference_candidates/combined_shortlist.json"
    atomic_json(shortlist, clips[:50])
    output = root / "references/raphael"
    subprocess.run(
        [
            str(python),
            str(ROOT / "scripts/score_voice_references.py"),
            "--shortlist",
            str(shortlist),
            "--speaker-weights",
            str(encoder),
            "--panns-weights",
            str(panns),
            "--panns-source",
            str(vendor),
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_voice_transcripts.py"),
            "--references",
            str(output / "references.json"),
            "--model",
            str(verification),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )
    print("Reference listening page:", output / "index.html")


if __name__ == "__main__":
    main()
