"""Download original source audio and shortlist references; never train or install a voice."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Run resumable download, lossless decode, local ASR and reference extraction."""
    from raphael.audio.voice_references import (
        atomic_json,
        candidate_passages,
        download_source,
        rank_signal_candidate,
        read_source_urls,
        sha256,
        signal_metrics,
        split_asr_sentences,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", type=Path, default=ROOT / "data/voice/source_urls.txt")
    parser.add_argument("--root", type=Path, default=ROOT / "data/voice")
    parser.add_argument("--yt-dlp", type=Path, help="Optional downloaded yt-dlp Python zipapp")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--asr-model",
        type=Path,
        help="Existing local faster-whisper model directory",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock_stream = (args.root / "prepare.lock").open("a")
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = read_source_urls(args.urls)
    downloader = [sys.executable, str(args.yt_dlp)] if args.yt_dlp else ["yt-dlp"]
    states = []
    for video_id, url in sources:
        if args.skip_download:
            state = json.loads((args.root / "sources" / video_id / "provenance.json").read_text())
        else:
            state = download_source(video_id, url, args.root, downloader)
        states.append(state)
    if args.download_only:
        return
    if args.asr_model is None or not (args.asr_model / "model.bin").is_file():
        parser.error(
            "--asr-model must point to an existing local model; ASR downloads are disabled"
        )

    import soundfile as sf
    from faster_whisper import WhisperModel

    from raphael.audio.stt import _preload_cuda_libraries

    _preload_cuda_libraries()
    model = WhisperModel(
        str(args.asr_model),
        device=args.device,
        compute_type="int8_float16" if args.device == "cuda" else "int8",
        cpu_threads=4,
        local_files_only=True,
    )
    for state in states:
        directory = args.root / "sources" / state["video_id"]
        analysis = directory / "analysis16k.wav"
        if not analysis.exists():
            temporary = directory / "analysis16k.partial.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    state["working"],
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(temporary),
                ],
                check=True,
            )
            temporary.replace(analysis)
        asr_path = directory / "transcription.json"
        fingerprint = {
            "audio_sha256": sha256(analysis),
            "model": str(args.asr_model.resolve()),
            "language": "en",
            "beam_size": 5,
            "word_timestamps": True,
        }
        asr = json.loads(asr_path.read_text()) if asr_path.exists() else {}
        if asr.get("fingerprint") != fingerprint:
            print(f"Transcribing {state['video_id']} with local {args.asr_model.name}", flush=True)
            iterator, info = model.transcribe(
                sf.read(analysis, dtype="float32")[0],
                language="en",
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
            )
            segments = []
            for s in iterator:
                segments.append(
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "avg_logprob": s.avg_logprob,
                        "no_speech_prob": s.no_speech_prob,
                        "compression_ratio": s.compression_ratio,
                        "words": [
                            {
                                "start": w.start,
                                "end": w.end,
                                "word": w.word,
                                "probability": w.probability,
                            }
                            for w in (s.words or [])
                        ],
                    }
                )
                print(f"  {s.end:.1f}/{info.duration:.1f}s", flush=True)
            asr = {
                "fingerprint": fingerprint,
                "segments": segments,
                "duration": info.duration,
                "language_probability": info.language_probability,
            }
            atomic_json(asr_path, asr)
        audio, rate = sf.read(state["working"], dtype="float32", always_2d=True)
        candidates = []
        for passage in candidate_passages(split_asr_sentences(asr["segments"])):
            end = min(passage["end"], len(audio) / rate)
            clip = audio[round(passage["start"] * rate) : round(end * rate)]
            if not clip.size:
                continue
            passage["end"] = end
            passage["duration"] = len(clip) / rate
            candidate = rank_signal_candidate(passage, signal_metrics(clip, rate))
            candidate.update(
                id=f"{state['video_id']}-{round(passage['start'] * 1000):07d}",
                video_id=state["video_id"],
                source_url=state["source_url"],
                submitted_url=state["submitted_url"],
                source_sha256=state["original_sha256"],
                sample_rate=rate,
                channels=audio.shape[1],
            )
            candidates.append(candidate)
        candidates.sort(key=lambda c: c["signal_score"], reverse=True)
        # Export a limited shortlist, not every second of the source.
        shortlist = []
        for c in candidates:
            if c["grade"] == "D" or any(
                abs(c["start"] - other["start"]) < 12 for other in shortlist
            ):
                continue
            shortlist.append(c)
            if len(shortlist) == 30:
                break
        clips_dir = args.root / "reference_candidates" / state["video_id"]
        clips_dir.mkdir(parents=True, exist_ok=True)
        for candidate in shortlist:
            clip = audio[round(candidate["start"] * rate) : round(candidate["end"] * rate)]
            destination = clips_dir / f"{candidate['id']}.wav"
            temporary = destination.with_suffix(".partial.wav")
            sf.write(temporary, clip.mean(axis=1), rate, subtype="PCM_24")
            temporary.replace(destination)
            candidate.update(wav=str(destination.resolve()), wav_sha256=sha256(destination))
            destination.with_suffix(".txt").write_text(candidate["text"] + "\n", encoding="utf-8")
            atomic_json(destination.with_suffix(".json"), candidate)
        atomic_json(clips_dir / "shortlist.json", shortlist)
        atomic_json(directory / "all_candidates.json", candidates)
        print(
            f"{state['video_id']}: {len(candidates)} scored, {len(shortlist)} exported",
            flush=True,
        )
        del audio
    del model


if __name__ == "__main__":
    main()
