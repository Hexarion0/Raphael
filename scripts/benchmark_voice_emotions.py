"""Run a bounded, resumable emotion listening experiment, one model at a time.

Uses cached models and references only. There is no training or runtime integration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data/voice/references/raphael/reference-1.wav"
EMOTION_REFS = ROOT / "data/voice/references/raphael/emotion"
OUTPUT = ROOT / "data/voice/benchmarks/emotion-controls"
ORIGINAL_PROFILES = {
    "default": {"exaggeration": 0.5, "cfg_weight": 0.5},
    "subtle": {"exaggeration": 0.6, "cfg_weight": 0.4},
    "moderate": {"exaggeration": 0.75, "cfg_weight": 0.3},
    "strong": {"exaggeration": 1.0, "cfg_weight": 0.3},
}
# The first four appear in the official Turbo demo's supported tag list.
# The other three are native tokenizer tokens; their delivery is exploratory.
TAG_CASES = [
    ("sigh", "Hey, it's okay. [sigh] You don't have to figure everything out tonight."),
    ("laugh", "You did it! [laugh] Everything is working again."),
    (
        "chuckle",
        "Oh, so now you're asking nicely? [chuckle] All right, I'll take care of it for you.",
    ),
    ("gasp", "[gasp] It's ready! Your project is finally running."),
    ("whispering", "[whispering] There we go. Everything can wait until morning."),
    ("happy", "[happy] You did it! Everything is working again. I'm really happy for you."),
    (
        "sarcastic",
        "[sarcastic] Oh, so now you're asking nicely? All right, I'll take care of it for you.",
    ),
]


def write_json(path: Path, value: object) -> Path:
    """Save the exact suite / settings used alongside its listening samples."""
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def build_suites(output: Path) -> dict[str, Path]:
    """Keep the existing eight texts exact; add seven semantic emotion scenarios."""
    import re

    baseline = json.loads((ROOT / "docs/voice-evaluation.json").read_text())
    emotions = json.loads((ROOT / "docs/voice-emotion-evaluation.json").read_text())
    neutral, tagged = [], []
    tokens = json.loads((ROOT / "data/voice/models/chatterbox-turbo/added_tokens.json").read_text())
    for tag, text in TAG_CASES:
        if f"[{tag}]" not in tokens:
            raise ValueError(f"Tag [{tag}] is absent from the installed model tokenizer")
        spoken = " ".join(re.sub(r"\[[^]]+\]", "", text).split())
        neutral.append({"id": "tag_" + tag, "category": "tag control", "text": spoken})
        tagged.append({
            "id": "tag_" + tag,
            "category": "documented tag" if tag in {"sigh", "laugh", "chuckle", "gasp"}
            else "tokenizer-only exploratory tag",
            "text": text,
            "spoken_text": spoken,
        })
    punctuation = [
        {
            "id": "punct_playful",
            "text": "Oh, so now you're asking nicely? All right, I'll take care of it for you.",
        },
        {
            "id": "punct_happy",
            "text": "You did it. Everything is working again. I'm really happy for you.",
        },
    ]
    changed = [
        {
            "id": "punct_playful",
            "text": "Oh... so NOW you're asking nicely? All right! I'll take care of it for you.",
        },
        {
            "id": "punct_happy",
            "text": "You DID it! Everything is working again! I'm really happy for you!",
        },
    ]
    emotion_tags = []
    for item in emotions:
        text = item["text"]
        if item["id"] == "09_comforting":
            text = text.replace("Hey, it's okay.", "Hey, it's okay. [sigh]")
        elif item["id"] == "10_concerned":
            text = text.replace("overwhelmed.", "overwhelmed. [sigh]")
        elif item["id"] == "12_happy":
            text = text.replace("You did it!", "You did it! [laugh]")
        elif item["id"] == "13_excited":
            text = "[gasp] " + text
        elif item["id"] == "14_playful":
            text = text.replace("nicely?", "nicely? [chuckle]")
        elif item["id"] == "15_sleepy":
            text = text.replace("There we go.", "There we go. [sigh]")
        emotion_tags.append({**item, "text": text, "spoken_text": item["text"]})
    return {
        name: write_json(output / f"{name}-suite.json", rows)
        for name, rows in {
            "all": baseline + emotions,
            "emotions": emotions,
            "tags-neutral": neutral,
            "tags": tagged,
            "punct-neutral": punctuation,
            "punct-expressive": changed,
            "sampling": [emotions[i] for i in [0, 3, 5]],
            "emotion-tags": emotion_tags,
        }.items()
    }


def main() -> None:
    """Resume only completed runs; retain failed attempts for diagnosis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=["chatterbox500", "chatterbox-turbo", "qwen06"])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suites = build_suites(OUTPUT)
    # (directory, label, reference, suite, repeats, generation overrides)
    if args.candidate == "chatterbox500":
        runs = [
            (
                f"original-{name}", f"Original 500M · {name}", PRIMARY, "all",
                2 if name == "default" else 1, config,
            )
            for name, config in ORIGINAL_PROFILES.items()
        ]
    elif args.candidate == "chatterbox-turbo":
        runs = [("turbo-neutral", "Turbo · primary reference", PRIMARY, "all", 1, {})]
        runs.extend(
            (
                f"turbo-reference-{name}", f"Turbo · {name} reference",
                EMOTION_REFS / f"reference-{name}.wav", "all" if name == "pleased"
                else "emotions", 1, {},
            )
            for name in ["pleased", "playful", "caring"]
        )
        runs.extend(
            (f"turbo-{name}", f"Turbo · {name}", PRIMARY, name, 1, {})
            for name in ["tags-neutral", "tags", "punct-neutral", "punct-expressive",
                         "emotion-tags"]
        )
        runs.extend(
            (
                f"turbo-temperature-{temperature}", f"Turbo · temperature {temperature}",
                PRIMARY, "sampling", 1, {"temperature": temperature},
            )
            for temperature in [0.6, 1.0]
        )
    else:
        runs = [("qwen-emotions", "Qwen Base · emotional texts", PRIMARY, "emotions", 2, {})]
    environment = "qwen" if args.candidate == "qwen06" else "clone"
    interpreter = ROOT / f"data/voice/envs/{environment}/bin/python"
    for directory, label, reference, suite, repeats, config in runs:
        destination = OUTPUT / directory
        if (destination / "summary.json").exists():
            summary = json.loads((destination / "summary.json").read_text())
            rows = json.loads((destination / "measurements.json").read_text())
            wanted = len(json.loads(suites[suite].read_text())) * repeats
            if not summary["errors"] and len([r for r in rows if r["phase"] == "warm"]) == wanted:
                print("Already complete:", directory, flush=True)
                continue
            raise RuntimeError(f"Inspect incomplete attempt before resuming: {destination}")
        config_path = write_json(OUTPUT / f"{directory}-settings.json", config)
        command = [
            str(interpreter), str(ROOT / "scripts/benchmark_cloning_voice.py"),
            "--candidate", args.candidate,
            "--model", str(ROOT / "data/voice/models" / args.candidate),
            "--reference", str(reference), "--reference-text", str(reference.with_suffix(".txt")),
            "--suite", str(suites[suite]), "--output", str(destination),
            "--repeats", str(repeats), "--label", label,
            "--generation-config", str(config_path),
        ]
        print(json.dumps(command), flush=True)
        if not args.plan_only:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
