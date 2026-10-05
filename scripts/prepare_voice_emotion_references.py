"""Extract bounded alternate delivery references from the already verified source.

Content suggests a delivery to test; no automatic classifier certifies its emotion.
The original benchmark reference and training dataset remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data/voice/cache/huggingface"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "data/voice/cache/numba"))
sys.path.insert(0, str(ROOT / "src"))

# Word-aligned passages, with 120 ms before and 150 ms after spoken words.
# Turbo requires >5 s; original uses 6 s encoder / 10 s decoder conditioning.
PASSAGES = [
    ("playful", 637.91, 645.42, "Teasing question; compare delivery with the primary reference"),
    ("pleased", 780.42, 786.35, "Inviting question without the preceding five-second pause"),
    ("caring", 359.20, 370.62, "Reassurance followed by a warm personal statement"),
]


def main() -> None:
    """Preserve timestamps and score background / identity consistency on CPU."""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    from safetensors.torch import load_file

    from raphael.audio.voice_references import atomic_json, sha256, signal_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/voice/sources/PYmd20HsBj4")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/voice/references/raphael/emotion"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    models = ROOT / "data/voice/models"
    encoder = VoiceEncoder()
    encoder.load_state_dict(load_file(models / "chatterbox-turbo/ve.safetensors"))
    encoder.eval()
    panns = ROOT / "data/voice/vendor/panns"
    sys.path.insert(0, str(panns / "pytorch"))
    from models import Cnn6

    with (panns / "metadata/class_labels_indices.csv").open() as stream:
        labels = [row[2] for row in list(csv.reader(stream))[1:]]
    classifier = Cnn6(32000, 1024, 320, 64, 50, 14000, len(labels))
    classifier.load_state_dict(
        torch.load(models / "analysis/panns-cnn6.pth", map_location="cpu", weights_only=True)[
            "model"
        ]
    )
    classifier.eval()
    primary = ROOT / "data/voice/references/raphael/reference-1.wav"
    primary_audio, _ = librosa.load(primary, sr=16000)
    with torch.inference_mode():
        primary_embedding = encoder.embeds_from_wavs([primary_audio], sample_rate=16000)[0]
    asr = json.loads((args.source / "transcription.json").read_text())
    provenance = json.loads((args.source / "provenance.json").read_text())
    audio, rate = sf.read(args.source / "working.flac", dtype="float32", always_2d=True)
    references = []
    for name, start, end, reason in PASSAGES:
        segments = [s for s in asr["segments"] if s["end"] > start and s["start"] < end]
        words = [
            w for s in segments for w in s["words"] if start <= w["start"] < end
        ]
        text = "".join(w["word"] for w in words).strip()
        clip = audio[round(start * rate) : round(end * rate)].mean(axis=1)
        path = args.output / f"reference-{name}.wav"
        sf.write(path, clip, rate, subtype="PCM_24")
        path.with_suffix(".txt").write_text(text + "\n")
        resampled = librosa.resample(clip, orig_sr=rate, target_sr=16000)
        windows = [resampled[i : i + 48000] for i in range(0, len(resampled) - 32000, 32000)]
        with torch.inference_mode():
            embedding = encoder.embeds_from_wavs([resampled], sample_rate=16000)[0]
            window_embeddings = encoder.embeds_from_wavs(windows, sample_rate=16000)
            scores = classifier(
                torch.from_numpy(librosa.resample(clip, orig_sr=rate, target_sr=32000))[None]
            )["clipwise_output"][0].numpy()
        tags = {
            tag: float(scores[labels.index(tag)])
            for tag in ["Speech", "Music", "Whispering", "Laughter", "Shout", "Screaming"]
        }
        metrics = signal_metrics(clip, rate)
        cosine = float(embedding @ primary_embedding)
        minimum = float(np.min(window_embeddings @ primary_embedding))
        flags = []
        if cosine < 0.8 or minimum < 0.65:
            flags.append("speaker_or_delivery_change_review")
        if metrics["quiet_fraction"] > 0.35:
            flags.append("excess_silence")
        if tags["Music"] > 0.35:
            flags.append("music_classifier_flag")
        if metrics["clipping_fraction"] > 0.001:
            flags.append("clipping")
        ref = {
            **metrics,
            "id": name,
            "start": start,
            "end": end,
            "duration": len(clip) / rate,
            "video_id": "PYmd20HsBj4",
            "source_url": "https://www.youtube.com/watch?v=PYmd20HsBj4",
            "source_sha256": provenance.get("original_sha256"),
            "reference_wav": str(path.resolve()),
            "wav_sha256": sha256(path),
            "text": text,
            "word_confidence_mean": float(np.mean([w["probability"] for w in words])),
            "music_score": tags["Music"],
            "sound_tags": tags,
            "source_speaker_consistency_cosine": cosine,
            "speaker_window_min_cosine": minimum,
            "speaker_confidence": None,
            "overlap_score": None,
            "selection_reason": reason + "; emotion and identity require listening",
            "verification_status": "provisional; identity / overlap / delivery review pending",
            "transcript_status": "source word-aligned ASR; second-model check pending",
            "reasons": flags,
        }
        atomic_json(path.with_suffix(".json"), ref)
        references.append(ref)
        print(name, text, {"speaker_cosine": cosine, "music": tags["Music"], "flags": flags})
    atomic_json(args.output / "references.json", references)


if __name__ == "__main__":
    main()
