"""Screen reference candidates with native speaker embeddings and PANNs sound tags.

Scores are diagnostic proxies. They cannot certify identity or absence of overlap.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Rank candidates and publish three provisional, non-overlapping references."""
    import librosa
    import numpy as np
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    from safetensors.torch import load_file

    from raphael.audio.voice_references import atomic_json, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--speaker-weights", type=Path, required=True)
    parser.add_argument("--panns-weights", type=Path, required=True)
    parser.add_argument("--panns-source", type=Path, default=ROOT / "data/voice/vendor/panns")
    parser.add_argument("--output", type=Path, default=ROOT / "data/voice/references/raphael")
    args = parser.parse_args()
    torch.set_num_threads(4)
    encoder = VoiceEncoder()
    encoder.load_state_dict(load_file(args.speaker_weights))
    encoder.eval()
    sys.path.insert(0, str(args.panns_source / "pytorch"))
    from models import Cnn6

    with (args.panns_source / "metadata/class_labels_indices.csv").open() as stream:
        labels = [row[2] for row in list(csv.reader(stream))[1:]]
    classifier = Cnn6(32000, 1024, 320, 64, 50, 14000, len(labels))
    classifier.load_state_dict(
        torch.load(args.panns_weights, map_location="cpu", weights_only=True)["model"]
    )
    classifier.eval()
    candidates = json.loads(args.shortlist.read_text())
    embeddings = []
    for candidate in candidates:
        audio, rate = librosa.load(candidate["wav"], sr=16000)
        with torch.inference_mode():
            embedding = encoder.embeds_from_wavs([audio], sample_rate=rate)[0]
            windows = [audio[i : i + 3 * rate] for i in range(0, len(audio) - 2 * rate, 2 * rate)]
            window_embeddings = encoder.embeds_from_wavs(windows, sample_rate=rate)
            sound_audio = librosa.resample(audio, orig_sr=rate, target_sr=32000)
            scores = classifier(torch.from_numpy(sound_audio)[None])["clipwise_output"][0].numpy()
        embeddings.append(embedding)
        candidate["speaker_window_min_cosine"] = float(np.min(window_embeddings @ embedding))
        candidate["sound_tags"] = {
            tag: float(scores[labels.index(tag)])
            for tag in [
                "Speech",
                "Music",
                "Whispering",
                "Laughter",
                "Shout",
                "Screaming",
                "Conversation",
                "Narration, monologue",
                "Silence",
            ]
            if tag in labels
        }
        candidate["sound_top_tags"] = {
            labels[i]: float(scores[i]) for i in np.argsort(scores)[-6:][::-1]
        }
        candidate["music_score"] = candidate["sound_tags"]["Music"]
        candidate["speaker_model_sha256"] = sha256(args.speaker_weights)
        candidate["background_model_sha256"] = sha256(args.panns_weights)
    embeddings = np.asarray(embeddings)
    centroid = np.mean(embeddings, axis=0)
    centroid /= np.linalg.norm(centroid)
    # Dominant-source consistency is not an independently enrolled identity check.
    for candidate, embedding in zip(candidates, embeddings):
        consistency = float(embedding @ centroid)
        tags = candidate["sound_tags"]
        atypical = max(tags.get(t, 0) for t in ["Whispering", "Laughter", "Shout", "Screaming"])
        score = (
            candidate["signal_score"]
            + 15 * consistency
            + 10 * candidate["speaker_window_min_cosine"]
            - 50 * tags["Music"]
            - 30 * atypical
        )
        candidate.update(
            reference_score=round(score, 3),
            source_speaker_consistency_cosine=consistency,
            speaker_confidence=None,
            overlap_score=None,
            grade="D" if candidate["grade"] == "D" else "C",
            verification_status="provisional; human identity/overlap check pending",
            selection_reason=(
                "High signal/transcription rank, low detected music/atypical sound "
                "tags, and consistent source-speaker embeddings; not identity proof"
            ),
        )
        if tags["Music"] >= 0.35:
            candidate["reasons"].append("music_classifier_flag")
        if atypical >= 0.35:
            candidate["reasons"].append("atypical_delivery_classifier_flag")
        if candidate["speaker_window_min_cosine"] < 0.65:
            candidate["reasons"].append("speaker_or_delivery_change_flag")
        text = candidate["text"]
        if text and not text[0].isupper():
            candidate["reasons"].append("possible_sentence_fragment")
            candidate["reference_score"] -= 20
        print(
            candidate["id"],
            candidate["reference_score"],
            {k: round(v, 3) for k, v in tags.items()},
            flush=True,
        )
    candidates.sort(key=lambda c: c["reference_score"], reverse=True)
    selected = []
    for candidate in candidates:
        if candidate["grade"] == "D" or any(
            reason in candidate["reasons"]
            for reason in [
                "music_classifier_flag",
                "atypical_delivery_classifier_flag",
                "speaker_or_delivery_change_flag",
                "possible_sentence_fragment",
            ]
        ):
            continue
        if any(
            candidate["video_id"] == c["video_id"]
            and candidate["start"] < c["end"]
            and candidate["end"] > c["start"]
            for c in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == 3:
            break
    args.output.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(selected, 1):
        target = args.output / f"reference-{index}.wav"
        shutil.copy2(candidate["wav"], target)
        target.with_suffix(".txt").write_text(candidate["text"] + "\n", encoding="utf-8")
        candidate["reference_wav"] = str(target.resolve())
        atomic_json(target.with_suffix(".json"), candidate)
    atomic_json(args.output / "references.json", selected)
    atomic_json(args.output / "ranked_candidates.json", candidates)
    for category, clips in {
        "accepted": [],
        "questionable": [c for c in candidates if c["grade"] != "D"],
        "rejected": [c for c in candidates if c["grade"] == "D"],
    }.items():
        (args.output / category).mkdir(exist_ok=True)
        atomic_json(args.output / category / "manifest.json", clips)
    rows = []
    for c in selected:
        name = Path(c["reference_wav"]).name
        url = f"{c['source_url']}&t={int(c['start'])}s"
        rows.append(
            f"<article><h2>{html.escape(name)}</h2><audio controls preload='none' src='{name}'>"
            f"</audio><p>{html.escape(c['text'])}</p><p>"
            f"{c['start']:.2f}–{c['end']:.2f}s; {c['duration']:.2f}s; "
            f"score {c['reference_score']:.1f}; music tag {c['music_score']:.3f}</p>"
            f"<p>{html.escape(c['verification_status'])}</p>"
            f"<p>{html.escape(c['selection_reason'])}</p><a href='{html.escape(url)}'>Source</a>"
            f"<p>Flags: {html.escape(', '.join(c['reasons']) or 'none from available checks')}</p>"
            "</article>"
        )
    (args.output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>RAPHAEL reference shortlist</title>"
        "<style>body{font:18px system-ui;max-width:960px;margin:32px auto;padding:20px;"
        "background:#10151d;color:#e0e8ef}article{background:#1c2633;padding:20px;"
        "margin-bottom:20px;border-radius:12px}a{color:#8ecbff}audio{width:100%}</style>"
        "<h1>Three provisional reference passages</h1><p>Original speech, no denoising. "
        "Transcript, identity, and absence of overlap are not manually certified. "
        "Sound tags and cosine similarities are diagnostic scores, not probabilities.</p>"
        + "".join(rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
