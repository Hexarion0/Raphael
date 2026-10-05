"""Regression tests for provenance, conservative scoring and reference preparation."""

import json

import numpy as np
import pytest
import soundfile as sf

from raphael.audio.voice_references import (
    candidate_passages,
    download_source,
    rank_signal_candidate,
    read_source_urls,
    sha256,
    signal_metrics,
    split_asr_sentences,
)


def test_youtube_aliases_deduplicate_and_comments_are_ignored(tmp_path):
    path = tmp_path / "sources.txt"
    path.write_text(
        "# source\n\nhttps://youtu.be/PYmd20HsBj4?t=20\n"
        "https://www.youtube.com/watch?v=PYmd20HsBj4&list=ignored\n"
    )
    assert read_source_urls(path) == [("PYmd20HsBj4", "https://youtu.be/PYmd20HsBj4?t=20")]


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/playlist?list=abc",
        "https://youtube.com.evil.test/watch?v=PYmd20HsBj4",
        "http://youtu.be/PYmd20HsBj4",
        "https://youtu.be/../../outside",
    ],
)
def test_non_video_or_unsafe_source_urls_are_rejected(tmp_path, url):
    path = tmp_path / "sources.txt"
    path.write_text(url)
    with pytest.raises(ValueError):
        read_source_urls(path)


def test_verified_original_and_working_audio_resume_without_downloading(tmp_path, monkeypatch):
    directory = tmp_path / "sources" / "PYmd20HsBj4"
    directory.mkdir(parents=True)
    original = directory / "original.webm"
    original.write_bytes(b"preserved compressed original")
    working = directory / "working.flac"
    sf.write(working, np.zeros(16000), 16000)
    state = {"original_sha256": sha256(original), "working_sha256": sha256(working)}
    (directory / "provenance.json").write_text(json.dumps(state))

    def forbid_download(*args, **kwargs):
        pytest.fail("A verified cached source must not invoke external processing")

    monkeypatch.setattr("raphael.audio.voice_references.subprocess.run", forbid_download)
    assert download_source("PYmd20HsBj4", "https://youtu.be/PYmd20HsBj4", tmp_path, []) == state


def test_changed_original_is_preserved_and_rejected(tmp_path):
    directory = tmp_path / "sources" / "PYmd20HsBj4"
    directory.mkdir(parents=True)
    original = directory / "original.webm"
    original.write_bytes(b"changed")
    (directory / "provenance.json").write_text(json.dumps({"original_sha256": "old"}))
    with pytest.raises(ValueError, match="changed or is corrupt"):
        download_source("PYmd20HsBj4", "https://youtu.be/PYmd20HsBj4", tmp_path, [])
    assert original.read_bytes() == b"changed"


def test_passages_keep_original_utterance_boundaries_and_avoid_large_gaps():
    segments = [
        {
            "start": start,
            "end": end,
            "text": text,
            "avg_logprob": -0.1,
            "no_speech_prob": 0.01,
            "compression_ratio": 1.0,
        }
        for start, end, text in [
            (20, 25, "Hello."),
            (25.5, 32, "Welcome back."),
            (40, 45, "Another passage."),
        ]
    ]
    result = candidate_passages(segments)
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(19.88)
    assert result[0]["end"] == pytest.approx(32.15)
    assert result[0]["text"] == "Hello. Welcome back."


def test_missing_speaker_and_music_evidence_never_become_accepted():
    audio = np.sin(np.arange(160000) * 0.1).astype(np.float32) * 0.2
    metrics = signal_metrics(audio, 16000)
    passage = {
        "text": "A normal passage.",
        "avg_logprob": -0.1,
        "max_no_speech_prob": 0.01,
        "max_compression_ratio": 1.0,
    }
    result = rank_signal_candidate(passage, metrics)
    assert result["grade"] == "C"
    assert result["speaker_confidence"] is None
    assert result["music_score"] is None
    assert result["overlap_score"] is None


def test_clipped_audio_cannot_be_saved_by_high_transcription_confidence():
    metrics = signal_metrics(np.ones(160000), 16000)
    passage = {
        "text": "Clearly recognized words.",
        "avg_logprob": -0.01,
        "max_no_speech_prob": 0.01,
        "max_compression_ratio": 1.0,
    }
    result = rank_signal_candidate(passage, metrics)
    assert result["grade"] == "D"
    assert "clipping" in result["reasons"]


def test_long_asr_segments_split_on_punctuation_and_original_word_times():
    segment = {
        "start": 5.0,
        "end": 40.0,
        "text": "Hello there. How are you?",
        "avg_logprob": -0.2,
        "words": [
            {"start": 5.2, "end": 6.0, "word": " Hello"},
            {"start": 6.1, "end": 7.0, "word": " there."},
            {"start": 8.0, "end": 8.2, "word": " How"},
            {"start": 8.3, "end": 8.5, "word": " are"},
            {"start": 8.6, "end": 9.0, "word": " you?"},
        ],
    }
    sentences = split_asr_sentences([segment])
    assert [(s["start"], s["end"], s["text"]) for s in sentences] == [
        (5.2, 7.0, "Hello there."),
        (8.0, 9.0, "How are you?"),
    ]
    assert all(s["avg_logprob"] == -0.2 for s in sentences)
