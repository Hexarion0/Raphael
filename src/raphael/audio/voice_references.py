"""Download licensed sources and shortlist reference passages without training.

Original media and source timestamps remain immutable. Automatic diagnostics do
not certify speaker identity, absence of overlap, or verbatim transcription.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

PREPARATION_VERSION = 1


def sha256(path: Path) -> str:
    """Hash an artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Publish metadata only after a complete write on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_source_urls(path: Path) -> list[tuple[str, str]]:
    """Deduplicate explicit YouTube videos; never expand a channel or playlist."""
    sources: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https":
            raise ValueError("Source URLs must use HTTPS")
        if host == "youtu.be":
            video_id = parsed.path.strip("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            if parsed.path.startswith(("/shorts/", "/embed/")):
                video_id = parsed.path.split("/")[2]
        else:
            raise ValueError(f"Unsupported source host: {host}")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Provide a URL for an individual YouTube video")
        sources.setdefault(video_id, url)
    if not sources:
        raise ValueError("No source URLs supplied")
    return list(sources.items())


def download_source(
    video_id: str,
    submitted_url: str,
    root: Path,
    downloader: list[str],
) -> dict:
    """Resume a best-original-English-track download and preserve original media."""
    directory = root / "sources" / video_id
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "provenance.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    originals = [
        p
        for p in directory.glob("original.*")
        if p.suffix in {".webm", ".m4a", ".mp4", ".opus", ".ogg"}
    ]
    verified = bool(len(originals) == 1 and state.get("original_sha256") == sha256(originals[0]))
    if originals and state.get("original_sha256") and not verified:
        raise ValueError("Original source changed or is corrupt; preserve it for inspection")
    if not verified:
        # Local verification takes precedence over a stale archive entry.
        archive = directory / "download-archive.txt"
        if archive.exists():
            archive.write_text("", encoding="utf-8")
        command = [
            *downloader,
            "--ignore-config",
            "--no-playlist",
            "--continue",
            "--write-info-json",
            "--download-archive",
            str(archive),
            "--format",
            "bestaudio[language^=en]/bestaudio/best",
            "--paths",
            str(directory),
            "--output",
            "original.%(ext)s",
            submitted_url,
        ]
        subprocess.run(command, check=True)
        originals = [
            p
            for p in directory.glob("original.*")
            if p.suffix in {".webm", ".m4a", ".mp4", ".opus", ".ogg"}
        ]
        if len(originals) != 1:
            raise ValueError("Expected one preserved original audio artifact")
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1",
                str(originals[0]),
            ],
            check=True,
            capture_output=True,
        )
        info = json.loads((directory / "original.info.json").read_text())
        if info["id"] != video_id:
            raise ValueError("Downloaded video ID differs from the requested source")
        language = info.get("language")
        if language and not language.startswith("en"):
            raise ValueError("Selected track is not English; inspect alternate/dubbed tracks")
        state = {
            "video_id": video_id,
            "submitted_url": submitted_url,
            "source_url": info["webpage_url"],
            "title": info.get("title"),
            "channel": info.get("channel"),
            "source_duration": info.get("duration"),
            "format_id": info.get("format_id"),
            "language": language,
            "format_note": info.get("format_note"),
            "codec": info.get("acodec"),
            "original": str(originals[0].resolve()),
            "original_sha256": sha256(originals[0]),
            "downloader_version": subprocess.check_output(
                [*downloader, "--version"],
                text=True,
            ).strip(),
        }
        atomic_json(state_path, state)
    working = directory / "working.flac"
    valid_working = bool(working.exists() and state.get("working_sha256") == sha256(working))
    if not valid_working:
        temporary = directory / "working.partial.flac"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(originals[0]),
                "-vn",
                "-c:a",
                "flac",
                str(temporary),
            ],
            check=True,
        )
        sf.info(temporary)  # Reject failed/invalid decodes before marking this stage complete.
        temporary.replace(working)
        state.update(
            working_sha256=sha256(working),
            working=str(working.resolve()),
            preparation_version=PREPARATION_VERSION,
        )
        atomic_json(state_path, state)
    return state


def signal_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    """Measure conservative signal diagnostics; these are not music/overlap detectors."""
    samples = np.asarray(audio, dtype=np.float32)
    if sample_rate <= 0 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Audio must be finite, nonempty, and have a positive sample rate")
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    frame_size = max(1, sample_rate // 50)
    frames = np.array(
        [np.sqrt(np.mean(mono[i : i + frame_size] ** 2)) for i in range(0, len(mono), frame_size)]
    )
    background = float(np.percentile(frames, 10))
    speech_level = float(np.percentile(frames, 75))
    return {
        "peak": float(np.max(np.abs(samples))),
        "clipping_fraction": float(np.mean(np.abs(samples) >= 0.995)),
        "rms_dbfs": float(20 * np.log10(max(float(np.sqrt(np.mean(mono**2))), 1e-9))),
        "quiet_fraction": float(np.mean(frames < max(0.003, speech_level * 0.05))),
        "background_proxy_dbfs": float(20 * np.log10(max(background, 1e-9))),
        "dynamic_contrast_db": float(
            20 * np.log10(max(speech_level, 1e-9) / max(background, 1e-9))
        ),
        "music_score": None,
        "overlap_score": None,
        "speaker_confidence": None,
    }


def split_asr_sentences(segments: list[dict]) -> list[dict]:
    """Use ASR word times to recover sentence boundaries inside long segments."""
    sentences = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            sentences.append(segment)
            continue
        pending = []
        for index, word in enumerate(words):
            pending.append(word)
            if word["word"].rstrip().endswith((".", "?", "!")) or index == len(words) - 1:
                sentences.append({
                    **segment, "start": pending[0]["start"], "end": pending[-1]["end"],
                    "text": "".join(w["word"] for w in pending).strip(), "words": pending,
                })
                pending = []
    return sentences


def candidate_passages(segments: list[dict], minimum: float = 10, maximum: float = 30) -> list:
    """Combine ASR utterances at their boundaries, preserving original timestamps."""
    passages = []
    for start_index, first in enumerate(segments):
        group = []
        for segment in segments[start_index:]:
            if group and segment["start"] - group[-1]["end"] > 4.0:
                break
            duration = segment["end"] - first["start"]
            if duration > maximum:
                break
            group.append(segment)
            text = " ".join(s["text"].strip() for s in group)
            if duration >= minimum and text.endswith((".", "?", "!")):
                passages.append(
                    {
                        "start": max(0, first["start"] - 0.12),
                        "end": segment["end"] + 0.15,
                        "text": text,
                        "duration": duration + 0.27,
                        "avg_logprob": float(np.mean([s["avg_logprob"] for s in group])),
                        "max_no_speech_prob": max(s["no_speech_prob"] for s in group),
                        "max_compression_ratio": max(s["compression_ratio"] for s in group),
                        "segments": group,
                    }
                )
                break
    return passages


def rank_signal_candidate(passage: dict, metrics: dict) -> dict:
    """Rank reference candidates while leaving missing identity evidence explicit."""
    score = 100.0
    reasons = []
    rejected = False
    if metrics["clipping_fraction"] > 0.001:
        reasons.append("clipping")
        score -= 35
        rejected = True
    if metrics["quiet_fraction"] > 0.35:
        reasons.append("excess_silence")
        score -= 25
    if passage["avg_logprob"] < -0.6:
        reasons.append("uncertain_transcription")
        score -= 25
    if passage["max_no_speech_prob"] > 0.5 or passage["max_compression_ratio"] > 2.4:
        reasons.append("possible_asr_hallucination")
        score -= 40
        rejected = True
    if re.search(r"\b(?:ha ha|hahaha|laughing|whispering|shouting)\b", passage["text"], re.I):
        reasons.append("atypical_delivery_text_marker")
        score -= 20
    return {
        **passage,
        **metrics,
        "signal_score": max(0, score),
        "grade": "D" if rejected else "C",
        "reasons": reasons,
        "verification_status": "provisional; speaker, overlap and transcript need review",
        "transcript_status": "automatic ASR, not manually verified",
    }
