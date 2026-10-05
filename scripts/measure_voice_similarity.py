"""Compare two existing speaker-embedding proxies without claiming subjective similarity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "data/voice/cache/numba"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/voice/cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"


def main() -> None:
    """Use CPU only after timed inference; flag unreliable very short utterances."""
    import librosa
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--encoder", choices=["chatterbox", "campplus"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.encoder == "chatterbox":
        from chatterbox.models.voice_encoder import VoiceEncoder
        from safetensors.torch import load_file

        encoder = VoiceEncoder()
        encoder.load_state_dict(load_file(args.weights))
        encoder.eval()

        def embed(audio):
            with torch.inference_mode():
                return encoder.embeds_from_wavs([audio], sample_rate=16000)[0]
    else:
        import onnxruntime
        from torchaudio.compliance.kaldi import fbank

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        encoder = onnxruntime.InferenceSession(
            str(args.weights), sess_options=options, providers=["CPUExecutionProvider"]
        )

        def embed(audio):
            features = fbank(
                torch.from_numpy(audio)[None], num_mel_bins=80, dither=0, sample_frequency=16000
            )
            features -= features.mean(dim=0, keepdim=True)
            return encoder.run(None, {encoder.get_inputs()[0].name: features[None].numpy()})[
                0
            ].flatten()

    reference, _ = librosa.load(args.reference, sr=16000)
    target = embed(reference)
    target /= np.linalg.norm(target)
    fingerprint = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    for directory in args.directories:
        rows = json.loads((directory / "measurements.json").read_text())
        checked = []
        for row in rows:
            audio, _ = librosa.load(directory / row["audio"], sr=16000)
            vector = embed(audio)
            vector /= np.linalg.norm(vector)
            checked.append(
                {
                    "audio": row["audio"],
                    "encoder": args.encoder,
                    "encoder_sha256": fingerprint,
                    "reference_sha256": reference_hash,
                    "reference_cosine": float(vector @ target),
                    "short_clip_caution": len(audio) / 16000 < 3,
                    "interpretation": (
                        "Uncalibrated proxy. Chatterbox uses its encoder for conditioning; "
                        "CosyVoice uses CAMPPlus. Neither proves human-perceived similarity."
                    ),
                }
            )
        destination = directory / f"similarity-{args.encoder}.json"
        destination.write_text(json.dumps(checked, indent=2) + "\n")
        print(directory.name, args.encoder, "checked", len(checked), flush=True)


if __name__ == "__main__":
    main()
